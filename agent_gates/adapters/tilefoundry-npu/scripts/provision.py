#!/usr/bin/env python3
"""Provision the tilefoundry-npu container idempotently, and NEVER rebuild it.

Two modes, one contract:

  --check   (default) inspect only. Says whether the running deployment matches the
            tracked profile. Writes nothing, starts nothing, removes nothing.
  --apply   create the container ONLY if it does not exist. If it exists, --apply
            behaves like --check: it verifies and refuses on drift.

Why it never removes or recreates: the container is shared (other agents, other runs,
possibly a LIVE device job). A "converge" that rebuilds on any difference turns a typo
in the profile into an outage, and a rebuild is exactly how a verified image silently
becomes a different one. Drift is a NAMED verdict (DEPLOYMENT_DRIFT) plus the exact
differences; the repair is a human decision.

Verdicts / exit codes:
  0  PROVISIONED_NOOP   container matches the profile
  0  CREATED            (--apply only) it did not exist and was created
  3  DEPLOYMENT_DRIFT   it exists and differs; nothing was changed
  4  CONTAINER_ABSENT   --check and there is no such container
  5  IMAGE_UNRESOLVED   the profile image is not present locally
  2  CONFIG_ERROR       profile/env unreadable or incomplete
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACK = HERE.parent
PROFILE_PATH = PACK / "deployment_profile.json"
ENV_FILE = PACK / "env.sh"

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_DRIFT = 3
EXIT_ABSENT = 4
EXIT_IMAGE = 5


class ProvisionError(RuntimeError):
    def __init__(self, code: str, message: str, exit_code: int = EXIT_CONFIG):
        super().__init__("%s: %s" % (code, message))
        self.code = code
        self.exit_code = exit_code


def run(argv, timeout=120):
    """Bounded docker call: a hung daemon must not hang the check."""
    done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    return done.returncode, done.stdout, done.stderr


def load_profile(path: Path) -> dict:
    try:
        profile = json.loads(path.read_text())
    except OSError as exc:
        raise ProvisionError("PROFILE_UNREADABLE", str(exc)) from exc
    except ValueError as exc:
        raise ProvisionError("PROFILE_MALFORMED", str(exc)) from exc
    required = ("required_image_name", "required_image_digest", "required_devices",
                "required_binds")
    missing = [key for key in required if key not in profile]
    if missing:
        raise ProvisionError("PROFILE_INCOMPLETE", "missing keys: %r" % (missing,))
    return profile


def load_env_file(path: Path) -> dict:
    """The machine-local half: container name and image TAG. Deliberately not tracked."""
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ProvisionError("ENV_UNREADABLE", str(exc)) from exc
    values = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    for key in ("CONTAINER_NAME", "IMAGE"):
        if not values.get(key):
            raise ProvisionError("ENV_INCOMPLETE",
                                 "%s is empty or missing in %s" % (key, path))
    return values


def docker_inspect(name: str):
    rc, out, err = run(["docker", "inspect", name])
    if rc != 0:
        if "no such" in (err or "").lower():
            return None
        raise ProvisionError("DOCKER_INSPECT_FAILED", (err or out).strip()[:400])
    parsed = json.loads(out)
    return parsed[0] if parsed else None


def image_digest(profile: dict) -> str:
    rc, out, err = run(["docker", "image", "inspect", profile["required_image_name"],
                        "-f", "{{.Id}}"])
    if rc != 0:
        raise ProvisionError(
            "IMAGE_UNRESOLVED",
            "%s is not present locally: %s" % (profile["required_image_name"],
                                               (err or "").strip()[:300]),
            EXIT_IMAGE)
    return out.strip()


def binds_of(inspect: dict) -> dict:
    """destination -> {source, read_only}, from the Binds strings docker records."""
    found = {}
    for bind in (inspect.get("HostConfig") or {}).get("Binds") or []:
        parts = bind.split(":")
        source = parts[0]
        destination = parts[1] if len(parts) > 1 else ""
        mode = parts[2] if len(parts) > 2 else ""
        found[destination] = {"source": source, "read_only": "ro" in mode.split(",")}
    return found


def compare(inspect: dict, profile: dict) -> list:
    """Every difference between the running container and the tracked profile."""
    drift = []
    digest = inspect.get("Image")
    if digest != profile["required_image_digest"]:
        drift.append({"field": "image", "code": "IMAGE_DRIFT",
                      "expected": profile["required_image_digest"], "actual": digest})
    devices = [(device or {}).get("PathOnHost")
               for device in ((inspect.get("HostConfig") or {}).get("Devices") or [])]
    missing = [d for d in profile["required_devices"] if d not in devices]
    if missing:
        drift.append({"field": "devices", "code": "DEVICES_DRIFT",
                      "expected": profile["required_devices"], "actual": devices,
                      "missing": missing})
    binds = binds_of(inspect)
    for want in profile["required_binds"]:
        destination = want["destination"]
        have = binds.get(destination)
        if have is None:
            drift.append({"field": "mount", "code": "MOUNTS_DRIFT",
                          "destination": destination, "actual": None})
            continue
        # Host and container see the SAME path here (/home is a bind mount). A bind
        # whose source differs is a different filesystem even when it looks plausible.
        if profile.get("bind_source_equals_destination", True) and have["source"] != destination:
            drift.append({"field": "mount", "code": "MOUNTS_DRIFT",
                          "destination": destination, "expected_source": destination,
                          "actual_source": have["source"]})
        if bool(want.get("read_only")) != bool(have["read_only"]):
            drift.append({"field": "mount_mode", "code": "MOUNTS_DRIFT",
                          "destination": destination,
                          "expected_read_only": bool(want.get("read_only")),
                          "actual_read_only": have["read_only"]})
    want_user = profile.get("required_container_user", "root")
    actual_user = ((inspect.get("Config") or {}).get("User") or "root").strip() or "root"
    if actual_user != want_user:
        drift.append({"field": "user", "code": "USER_DRIFT",
                      "expected": want_user, "actual": actual_user})
    return drift


def create_command(profile: dict, env: dict) -> list:
    """The exact argv --apply uses. By DIGEST, never by tag: a moved tag is how a
    rebuilt image silently replaces the one a Gate was verified against."""
    argv = ["docker", "run", "-d", "--name", env["CONTAINER_NAME"]]
    for device in profile["required_devices"]:
        argv += ["--device", device]
    for want in profile["required_binds"]:
        destination = want["destination"]
        source = (destination if profile.get("bind_source_equals_destination", True)
                  else want.get("source"))
        suffix = ":ro" if want.get("read_only") else ""
        argv += ["-v", "%s:%s%s" % (source, destination, suffix)]
    user = profile.get("required_container_user", "root")
    if user and user != "root":
        argv += ["--user", user]
    argv += [profile["required_image_digest"], "sleep", "infinity"]
    return argv


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true",
                        help="create the container when absent; never modify an existing one")
    parser.add_argument("--check", action="store_true", help="inspect only (default)")
    parser.add_argument("--container", default=None,
                        help="inspect THIS container instead of the profile's (read-only; "
                             "used by the drift negatives)")
    parser.add_argument("--profile", default=str(PROFILE_PATH))
    parser.add_argument("--env-file", default=str(ENV_FILE))
    parser.add_argument("--dry-run", action="store_true",
                        help="with --apply, print the docker argv and create nothing")
    args = parser.parse_args(argv)

    report = {"tool": "provision.py", "mode": "apply" if args.apply else "check"}
    try:
        profile = load_profile(Path(args.profile))
        env = load_env_file(Path(args.env_file))
        report["profile"] = str(Path(args.profile).resolve())
        report["profile_image_name"] = profile["required_image_name"]
        report["profile_image_digest"] = profile["required_image_digest"]
        local_digest = image_digest(profile)
        report["image_present_locally"] = True
        report["image_local_digest"] = local_digest
        if local_digest != profile["required_image_digest"]:
            report["verdict"] = "IMAGE_TAG_MOVED"
            report["detail"] = ("the tag %r now resolves to %s, but the profile pins %s; "
                                "the pinned image was rebuilt or replaced"
                                % (profile["required_image_name"], local_digest,
                                   profile["required_image_digest"]))
            report["mutated"] = False
            print(json.dumps(report, indent=1, sort_keys=True))
            return EXIT_DRIFT

        name = args.container or env["CONTAINER_NAME"]
        report["container"] = name
        inspect = docker_inspect(name)
        if inspect is None:
            report["verdict"] = "CONTAINER_ABSENT"
            report["mutated"] = False
            if args.apply:
                command = create_command(profile, dict(env, CONTAINER_NAME=name))
                report["create_argv"] = command
                if args.dry_run:
                    report["dry_run"] = True
                    report["detail"] = "would create the container; nothing was run"
                    print(json.dumps(report, indent=1, sort_keys=True))
                    return EXIT_OK
                rc, out, err = run(command, timeout=300)
                report["create_rc"] = rc
                report["create_stdout"] = out.strip()[:400]
                report["create_stderr"] = err.strip()[:400]
                report["mutated"] = rc == 0
                report["verdict"] = "CREATED" if rc == 0 else "CREATE_FAILED"
                print(json.dumps(report, indent=1, sort_keys=True))
                return EXIT_OK if rc == 0 else EXIT_CONFIG
            print(json.dumps(report, indent=1, sort_keys=True))
            return EXIT_ABSENT

        drift = compare(inspect, profile)
        report["observed"] = {
            "image": inspect.get("Image"),
            "devices": [(device or {}).get("PathOnHost")
                        for device in ((inspect.get("HostConfig") or {}).get("Devices") or [])],
            "binds": binds_of(inspect),
            "user": ((inspect.get("Config") or {}).get("User") or "root").strip() or "root",
            "container_id": inspect.get("Id"),
        }
        if drift:
            report["verdict"] = "DEPLOYMENT_DRIFT"
            report["drift"] = drift
            report["remediation"] = ("manual: this tool will NOT remove or recreate a "
                                     "running container")
            report["mutated"] = False
            print(json.dumps(report, indent=1, sort_keys=True))
            return EXIT_DRIFT

        report["verdict"] = "PROVISIONED_NOOP"
        report["mutated"] = False
        report["detail"] = "image digest, devices, binds and user all match the profile"
        print(json.dumps(report, indent=1, sort_keys=True))
        return EXIT_OK
    except ProvisionError as exc:
        report["verdict"] = exc.code
        report["detail"] = str(exc)
        report["mutated"] = False
        print(json.dumps(report, indent=1, sort_keys=True))
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
