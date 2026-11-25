#!/usr/bin/env python3
"""GLM5.2 single-card golden runner for the megakernel-npu adapter.

This is a local adapter extension. It does not modify the digest-pinned
manifest, and it does not replace the existing megakernel-npu gates. It reads
machine configuration from `glm52.env` (git-ignored) beside `env.sh` and runs
the documented BlockRT/TD golden pipeline for one explicitly authorized device.

Commands:
    prepare     Export the pinned reference source into GLM52_REFERENCE_HOME.
    build       Build the reference runtime with -p A5 -t Dump (jobs <= 16).
    preflight   Verify reference/TD artifacts, Dump flag, manager ABI symbols.
    run         Run one layer count on an explicitly authorized device id.
    all         Run layers 1, 2 and 4 sequentially; stop on first failure.
    config      Print the resolved configuration (paths only).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

PROG = "glm52-single-card"
ADAPTER_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ADAPTER_DIR / "glm52.env"
ADAPTER_ENV = ADAPTER_DIR / "env.sh"
ASSIGN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")

REQUIRED = (
    "GLM52_REFERENCE_SOURCE",
    "GLM52_REFERENCE_COMMIT",
    "GLM52_REFERENCE_HOME",
    "GLM52_TD_SOURCE",
    "GLM52_SHMEM_HOME",
    "GLM52_GOLDEN_ROOT",
    "GLM52_LOG_ROOT",
)

DEFAULTS = {
    "GLM52_BUILD_JOBS": "16",
    "GLM52_BUILD_PLATFORM": "A5",
    "GLM52_BUILD_TYPE": "Dump",
    "GLM52_TOKENS": "8",
    "GLM52_TP_SIZE": "1",
    "GLM52_SCOPE": "boundaries",
}

MAX_BUILD_JOBS = 16


class ConfigError(ValueError):
    """A malformed or incomplete machine configuration."""


def _now():
    return datetime.now(timezone.utc)


def _stamp():
    return _now().strftime("%Y%m%d_%H%M%S")


def _parse_env_file(path, prefixes):
    if not path.is_file():
        raise ConfigError(f"configuration file is missing: {path}")
    values = {}
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = ASSIGN.match(line)
        if not match:
            raise ConfigError(f"{path}:{lineno}: expected NAME=VALUE, got {raw!r}")
        name, value = match.group(1), match.group(2).strip()
        if not name.startswith(prefixes):
            raise ConfigError(f"{path}:{lineno}: unexpected key {name!r}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if name in values:
            raise ConfigError(f"{path}:{lineno}: duplicate key {name!r}")
        values[name] = value
    return values


def _config_path():
    selected = (os.environ.get("GLM52_ENV_FILE") or "").strip()
    if selected:
        path = Path(selected)
        if not path.is_absolute() or not path.is_file():
            raise ConfigError(f"GLM52_ENV_FILE must be an absolute regular file: {selected!r}")
        return path
    return DEFAULT_CONFIG


def load_config():
    config = _parse_env_file(_config_path(), ("GLM52_",))
    adapter = _parse_env_file(ADAPTER_ENV, ("MEGAKERNEL_NPU_",))

    can_setup = adapter.get("MEGAKERNEL_NPU_CANN_SETUP", "")
    config.setdefault("GLM52_CANN_SETUP", can_setup)
    config.setdefault("GLM52_CANN_DIR", str(Path(can_setup).parent) if can_setup else "")
    config.setdefault("GLM52_PYTHON", adapter.get("MEGAKERNEL_NPU_PYTHON", ""))
    for name, value in DEFAULTS.items():
        config.setdefault(name, value)

    missing = [name for name in REQUIRED if not (config.get(name) or "").strip()]
    if missing:
        raise ConfigError("missing required keys in glm52.env: " + ", ".join(missing))
    for name in ("GLM52_CANN_SETUP", "GLM52_CANN_DIR", "GLM52_PYTHON"):
        if not (config.get(name) or "").strip():
            raise ConfigError(f"missing {name}: set it in glm52.env or env.sh")

    for name in REQUIRED + ("GLM52_CANN_SETUP", "GLM52_PYTHON", "GLM52_SHMEM_HOME"):
        if name == "GLM52_REFERENCE_COMMIT":
            continue
        if not Path(config[name]).is_absolute():
            raise ConfigError(f"{name} must be an absolute path: {config[name]!r}")

    try:
        jobs = int(config["GLM52_BUILD_JOBS"])
    except ValueError as exc:
        raise ConfigError(f"GLM52_BUILD_JOBS must be an integer: {config['GLM52_BUILD_JOBS']!r}") from exc
    if not 1 <= jobs <= MAX_BUILD_JOBS:
        raise ConfigError(f"GLM52_BUILD_JOBS must be in 1..{MAX_BUILD_JOBS}, got {jobs}")
    config["GLM52_BUILD_JOBS"] = str(jobs)
    return config


def base_env(config):
    env = dict(os.environ)
    python = Path(config["GLM52_PYTHON"])
    reference = Path(config["GLM52_REFERENCE_HOME"])
    td = Path(config["GLM52_TD_SOURCE"])
    env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
    env["CANN_DIR"] = config["GLM52_CANN_DIR"]
    env["ASCEND_TOOLKIT_HOME"] = config["GLM52_CANN_DIR"]
    env["SHMEM_HOME_PATH"] = config["GLM52_SHMEM_HOME"]
    env["ENABLE_PLATFORM_A5"] = "ON"
    env["MEGA_RUNTIME_HOME"] = str(reference)
    env["MEGA_KERNEL_HOME"] = str(reference / "output")
    env["MEGA_KERNEL_DIR"] = str(reference)
    ld_parts = [str(reference / "output"), str(reference / "output" / "lib")]
    if env.get("LD_LIBRARY_PATH"):
        ld_parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(ld_parts)
    py_parts = [str(reference / "python"), str(td / "python")]
    if env.get("PYTHONPATH"):
        py_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(py_parts)
    return env


def _activation(config):
    lines = [
        "set -eo pipefail",
        "set +u",
        f"source {shlex.quote(config['GLM52_CANN_SETUP'])}",
        "set -u",
        f"export CANN_DIR={shlex.quote(config['GLM52_CANN_DIR'])}",
        f"export ASCEND_TOOLKIT_HOME={shlex.quote(config['GLM52_CANN_DIR'])}",
        f"export SHMEM_HOME_PATH={shlex.quote(config['GLM52_SHMEM_HOME'])}",
        "export ENABLE_PLATFORM_A5=ON",
    ]
    bishengir = (config.get("GLM52_BISHENGIR_BIN") or "").strip()
    if bishengir:
        lines.append(f"export PATH={shlex.quote(bishengir)}:$PATH")
        lines.append(f"export TRITON_NPU_COMPILER_PATH={shlex.quote(bishengir)}")
    return "\n".join(lines)


def _cann_paths(config):
    root = Path(config["GLM52_CANN_DIR"])
    for base in (root / "x86_64-linux", root / "aarch64-linux", root):
        if (base / "include" / "acl" / "acl.h").is_file():
            include = base / "include"
            asc = base / "asc" / "include"
            ge = include / "graph"
            candidates = [
                base / "lib64" / "libascendcl.so",
                base / "runtime" / "lib64" / "libascendcl.so",
                base / "runtime" / "lib" / "libascendcl.so",
            ]
            ascendcl = next((path for path in candidates if path.is_file()), None)
            if ascendcl is None:
                raise ConfigError(f"libascendcl.so is missing under {base}")
            if not (asc / "tiling" / "platform" / "platform_ascendc.h").is_file():
                raise ConfigError(f"platform_ascendc.h is missing under {asc}")
            if not ge.is_dir():
                raise ConfigError(f"graph includes are missing under {include}")
            return {"include": str(include), "asc": str(asc), "ge": str(ge), "ascendcl": str(ascendcl)}
    raise ConfigError(f"cannot locate the CANN arch include roots under {root}")


def _site_packages(config):
    done = subprocess.run(
        [config["GLM52_PYTHON"], "-c",
         "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        capture_output=True, text=True, check=True,
    )
    return Path(done.stdout.strip())


def _ensure_overlay(config):
    """Expose the live triton_dist source without shadowing the built triton.

    The pipeline prepends ROOT/python to sys.path. A checkout whose
    python/triton symlink points at the unbuilt triton-ascend source breaks
    ``import triton``. The overlay copies triton_dist and links the installed
    triton package, which is the same layout the AgentGates CPU/device gates
    use for their live-source overlay.
    """
    overlay = Path(config["GLM52_LOG_ROOT"]) / "overlay"
    python_root = overlay / "python"
    python_root.mkdir(parents=True, exist_ok=True)
    source = Path(config["GLM52_TD_SOURCE"]) / "python" / "triton_dist"
    destination = python_root / "triton_dist"
    if not source.is_dir():
        raise ConfigError(f"triton_dist source is missing: {source}")
    subprocess.run(
        ["rsync", "-a", "--delete", "--exclude", "_prebuilt/", str(source) + "/", str(destination) + "/"],
        check=True,
    )
    installed_triton = _site_packages(config) / "triton"
    if not installed_triton.is_dir():
        raise ConfigError(f"installed triton package is missing: {installed_triton}")
    link = python_root / "triton"
    if link.is_symlink() or link.exists():
        if link.is_dir() and not link.is_symlink():
            import shutil
            shutil.rmtree(link)
        else:
            link.unlink()
    link.symlink_to(installed_triton, target_is_directory=True)
    include_link = overlay / "include"
    td_include = Path(config["GLM52_TD_SOURCE"]) / "include"
    if include_link.is_symlink() or include_link.exists():
        if include_link.is_dir() and not include_link.is_symlink():
            import shutil
            shutil.rmtree(include_link)
        else:
            include_link.unlink()
    include_link.symlink_to(td_include, target_is_directory=True)
    ops_link = overlay / "3rdparty" / "ascendc-ops"
    reference_ops = Path(config["GLM52_REFERENCE_HOME"]) / "examples" / "operators"
    if not reference_ops.is_dir():
        raise ConfigError(f"reference operator bundle is missing: {reference_ops}")
    ops_link.parent.mkdir(parents=True, exist_ok=True)
    if ops_link.is_symlink() or ops_link.exists():
        if ops_link.is_dir() and not ops_link.is_symlink():
            import shutil
            shutil.rmtree(ops_link)
        else:
            ops_link.unlink()
    ops_link.symlink_to(reference_ops, target_is_directory=True)
    return overlay


def _overlay_env(config, overlay):
    env = base_env(config)
    reference = Path(config["GLM52_REFERENCE_HOME"])
    parts = [str(overlay / "python"), str(reference / "python")]
    if os.environ.get("PYTHONPATH"):
        parts.append(os.environ["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _tail(path, lines=80):
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(text[-lines:])


def _run(script, cwd, log_path, env):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# {PROG} {_now().isoformat()}\n# cwd={cwd}\n\n")
        log.flush()
        done = subprocess.run(
            ["/bin/bash", "-c", script], cwd=str(cwd), env=env,
            stdout=log, stderr=subprocess.STDOUT,
        )
    return done.returncode


def cmd_prepare(config, force):
    source = Path(config["GLM52_REFERENCE_SOURCE"])
    commit = config["GLM52_REFERENCE_COMMIT"]
    destination = Path(config["GLM52_REFERENCE_HOME"])
    if not source.is_dir():
        raise ConfigError(f"GLM52_REFERENCE_SOURCE does not exist: {source}")
    if destination.exists() and any(destination.iterdir()) and not force:
        raise ConfigError(
            f"GLM52_REFERENCE_HOME is not empty: {destination} (use --force to re-export)")
    check = subprocess.run(
        ["git", "-C", str(source), "cat-file", "-e", commit + "^{commit}"], capture_output=True,
    )
    if check.returncode != 0:
        raise ConfigError(f"commit {commit} is not present in {source}")
    archive = subprocess.run(
        ["git", "-C", str(source), "archive", "--format=tar", commit],
        capture_output=True, check=True,
    )
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-x", "-C", str(destination)], input=archive.stdout, check=True)
    for name in ("catlass", "catlass_a5"):
        src_sub = source / "3rdparty" / name
        dst_sub = destination / "3rdparty" / name
        if not src_sub.is_dir():
            raise ConfigError(f"required submodule content is missing: {src_sub}")
        dst_sub.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["rsync", "-a", "--exclude", ".git", str(src_sub) + "/", str(dst_sub) + "/"],
            check=True,
        )
    provenance = {
        "format": 1,
        "source": str(source),
        "commit": commit,
        "exportedAt": _now().isoformat(),
        "note": "local export for the GLM5.2 reference runtime; submodules copied from the working tree",
    }
    (destination / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"prepared {destination} from {source} @ {commit}")


def cmd_build(config):
    reference = Path(config["GLM52_REFERENCE_HOME"])
    if not (reference / "build.sh").is_file() or not (reference / "CMakeLists.txt").is_file():
        raise ConfigError(f"reference source is not prepared: {reference} (run prepare first)")
    jobs = int(config["GLM52_BUILD_JOBS"])
    log_path = Path(config["GLM52_LOG_ROOT"]) / f"build_{_stamp()}.log"
    script = "\n".join(
        [
            _activation(config),
            f"export MEGA_KERNEL_HOME={shlex.quote(str(reference / 'output'))}",
            f"export MEGA_KERNEL_DIR={shlex.quote(str(reference))}",
            f"cd {shlex.quote(str(reference))}",
            f"bash build.sh -c -p {shlex.quote(config['GLM52_BUILD_PLATFORM'])} "
            f"-t {shlex.quote(config['GLM52_BUILD_TYPE'])} -j {jobs}",
        ]
    )
    started = _now()
    rc = _run(script, reference, log_path, base_env(config))
    stamp = {
        "format": 1,
        "startedAt": started.isoformat(),
        "finishedAt": _now().isoformat(),
        "exitCode": rc,
        "jobs": jobs,
        "platform": config["GLM52_BUILD_PLATFORM"],
        "buildType": config["GLM52_BUILD_TYPE"],
        "cann": config["GLM52_CANN_DIR"],
        "python": config["GLM52_PYTHON"],
        "crossCxx": config.get("GLM52_CROSS_CXX", ""),
        "log": str(log_path),
    }
    (reference / "build-stamp.json").write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
    if rc != 0:
        print(f"build FAILED (exit {rc}); log: {log_path}", file=sys.stderr)
        sys.stderr.write(_tail(log_path) + "\n")
        return rc
    print(f"build PASS; log: {log_path}")
    return cmd_build_tiling(config)


def cmd_build_tiling(config):
    overlay = _ensure_overlay(config)
    source = (
        overlay / "python" / "triton_dist" / "mega_kernel_ascend"
        / "models_blade" / "tiling" / "ops" / "ascendc"
    )
    if not (source / "CMakeLists.txt").is_file():
        raise ConfigError(f"ascendc tiling source is missing: {source}")
    build_dir = Path(config["GLM52_LOG_ROOT"]) / "tiling-build"
    python = config["GLM52_PYTHON"]
    pybind11_dir = subprocess.run(
        [python, "-c", "import pybind11; print(pybind11.get_cmake_dir())"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    cann = _cann_paths(config)
    jobs = int(config["GLM52_BUILD_JOBS"])
    script = "\n".join(
        [
            _activation(config),
            f"cd {shlex.quote(str(source))}",
            f"cmake -S . -B {shlex.quote(str(build_dir))} -DCMAKE_BUILD_TYPE=Release "
            f"-DPython3_EXECUTABLE={shlex.quote(python)} "
            f"-Dpybind11_DIR={shlex.quote(pybind11_dir)} "
            f"-DASCENDC_TILING_CANN_INCLUDE={shlex.quote(cann['include'])} "
            f"-DASCENDC_TILING_GE_INCLUDE={shlex.quote(cann['ge'])} "
            f"-DASCENDC_TILING_ASC_INCLUDE={shlex.quote(cann['asc'])} "
            f"-DASCENDC_TILING_ASCENDCL={shlex.quote(cann['ascendcl'])}",
            f"cmake --build {shlex.quote(str(build_dir))} "
            f"--target megakernel_ascendc_tiling -j {jobs}",
        ]
    )
    log_path = Path(config["GLM52_LOG_ROOT"]) / f"tiling_{_stamp()}.log"
    rc = _run(script, source, log_path, _overlay_env(config, overlay))
    extensions = sorted((source / "_prebuilt").glob("megakernel_ascendc_tiling*.so"))
    stamp = {
        "format": 1,
        "exitCode": rc,
        "jobs": jobs,
        "source": str(source),
        "log": str(log_path),
        "extensions": [str(path) for path in extensions],
        "finishedAt": _now().isoformat(),
    }
    (Path(config["GLM52_LOG_ROOT"]) / "tiling-build-stamp.json").write_text(
        json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
    if rc != 0 or not extensions:
        print(f"tiling build FAILED (exit {rc}); log: {log_path}", file=sys.stderr)
        sys.stderr.write(_tail(log_path) + "\n")
        return 1 if rc == 0 else rc
    print(f"tiling build PASS: {extensions[-1]}; log: {log_path}")
    return 0


def _symbol_names(path):
    done = subprocess.run(["nm", "-D", "--defined-only", str(path)], capture_output=True, text=True)
    if done.returncode != 0:
        return None
    names = []
    for line in done.stdout.splitlines():
        parts = line.split()
        if parts:
            names.append(parts[-1])
    return sorted(names)


def _size_fingerprint(path):
    done = subprocess.run(["nm", "-S", "-D", "--defined-only", str(path)], capture_output=True, text=True)
    if done.returncode != 0:
        return None
    rows = []
    for line in done.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4:
            rows.append((parts[1], parts[3]))
    rows.sort()
    return hashlib.sha256("\n".join(f"{size} {name}" for size, name in rows).encode()).hexdigest()


def _installed_manager(package):
    with tempfile.TemporaryDirectory(prefix="glm52-manager-") as tmp:
        path = Path(tmp) / "libaicpu_manager.so"
        done = subprocess.run(
            ["tar", "-xOzf", str(package), "aicpu_kernels_device/libaicpu_manager.so"],
            capture_output=True,
        )
        if done.returncode != 0 or not done.stdout:
            return None
        path.write_bytes(done.stdout)
        return _symbol_names(path), _size_fingerprint(path)


def _python_check(config, env, code):
    script = "\n".join([
        _activation(config),
        "exec " + shlex.quote(config["GLM52_PYTHON"]) + " -",
    ])
    return subprocess.run(
        ["/bin/bash", "-c", script], input=code, text=True,
        capture_output=True, env=env,
    )


def cmd_preflight(config):
    overlay = _ensure_overlay(config)
    env = _overlay_env(config, overlay)
    reference = Path(config["GLM52_REFERENCE_HOME"])
    checks = []

    def record(name, fn):
        try:
            detail = fn()
            checks.append((name, True, detail or "ok"))
        except Exception as exc:  # noqa: BLE001 - preflight reports every failure
            checks.append((name, False, f"{type(exc).__name__}: {exc}"))

    def artifacts():
        expected = [
            reference / "output" / "libaicpu_manager.so",
            reference / "output" / "lib" / "blockrt_ops.so",
            reference / "output" / "libdefault_callback.so",
            reference / "output" / "libskip_inactive_callback.so",
            reference / "output" / "libprint_callback.so",
        ]
        expected += list((reference / "output" / "lib").glob("ascendmk_rt.cpython-311*.so"))
        expected += list((reference / "output" / "lib").glob("ascendmk_dp.cpython-311*.so"))
        missing = [str(p) for p in expected if not p.is_file()]
        if missing:
            raise FileNotFoundError("missing artifacts: " + ", ".join(missing))
        return f"{len(expected)} artifacts"

    def blockrt_api():
        code = "\n".join([
            "import blockrt",
            "import ascendmk_rt",
            "assert ascendmk_rt.TENSOR_DUMP_ENABLED is True, 'TENSOR_DUMP_ENABLED is false'",
            "from blockrt.models.glm_5_2.glm5_2 import Glm52MegaKernel, Glm52MTPMegaKernel",
            "from blockrt.models.glm_5_2.model_args import ForwardBatchInfo",
            "assert hasattr(Glm52MegaKernel, 'clear_graph_cache'), 'clear_graph_cache missing'",
            "assert 'index_rope_positions' in ForwardBatchInfo.__dataclass_fields__",
            "print('TENSOR_DUMP_ENABLED and model APIs ok')",
        ])
        done = _python_check(config, env, code)
        if done.returncode != 0:
            raise RuntimeError((done.stderr or done.stdout).strip()[-1500:])
        return done.stdout.strip().splitlines()[-1]

    def td_api():
        code = "\n".join([
            "import triton_dist, triton",
            "from triton_dist.tools import ascend_device_export",
            "print('triton_dist', triton_dist.__file__, 'triton', triton.__version__)",
        ])
        done = _python_check(config, env, code)
        if done.returncode != 0:
            raise RuntimeError((done.stderr or done.stdout).strip()[-1500:])
        return done.stdout.strip().splitlines()[-1]

    def bishengir_tools():
        bin_dir = (config.get("GLM52_BISHENGIR_BIN") or "").strip()
        if not bin_dir:
            raise RuntimeError("GLM52_BISHENGIR_BIN is empty; TD dispatch needs a matched opt/compile pair")
        for name in ("bishengir-opt", "bishengir-compile"):
            path = Path(bin_dir) / name
            if not path.is_file() or not os.access(path, os.X_OK):
                raise FileNotFoundError(f"missing executable: {path}")
        done = subprocess.run(
            [str(Path(bin_dir) / "bishengir-opt"), "--version"],
            capture_output=True, text=True,
        )
        if done.returncode != 0:
            raise RuntimeError((done.stderr or done.stdout).strip()[-500:])
        return done.stdout.strip().splitlines()[0]

    def ascendc_bitcode():
        lib = _site_packages(config) / "triton" / "backends" / "ascend" / "lib"
        files = sorted(lib.glob("megakernel_op.*.c310.bc"))
        if len(files) != 4:
            raise FileNotFoundError(f"expected four megakernel bitcode files under {lib}, got {len(files)}")
        for path in files:
            data = path.read_bytes()
            if b"dtile_ascendc" not in data:
                raise RuntimeError(f"{path.name} has no dtile_ascendc symbols; rebuild with TRITON_ENABLE_ASCENDC_OPS=ON")
        return ", ".join(f"{path.name}={path.stat().st_size}" for path in files)

    def tiling_ext():
        code = "\n".join([
            "from triton_dist.mega_kernel_ascend.models_blade.tiling.ops.ascendc import _ext_loader",
            "module = _ext_loader.load_op_tiling_ext('gather_v2')",
            "print('tiling ext ok:', module.__file__)",
        ])
        done = _python_check(config, env, code)
        if done.returncode != 0:
            raise RuntimeError((done.stderr or done.stdout).strip()[-1500:])
        return done.stdout.strip().splitlines()[-1]

    def manager_abi():
        new_manager = reference / "output" / "libaicpu_manager.so"
        names = _symbol_names(new_manager)
        if not names:
            raise RuntimeError(f"cannot read dynamic symbols: {new_manager}")
        installed = _installed_manager(Path(config["GLM52_CANN_DIR"]) / "compat" / "Ascend-aicpu_pk.tar.gz")
        if installed is None:
            raise FileNotFoundError("installed Ascend-aicpu_pk.tar.gz is missing or unreadable")
        installed_names, installed_sizes = installed

        def abi_symbols(values):
            c_symbols = [name for name in values if not name.startswith("_Z")]
            core = [
                name for name in values
                if name.startswith("_Z")
                and not name.startswith(("_ZNSt", "_ZZNSt", "_ZSt"))
            ]
            return c_symbols, core

        built_c, built_core = abi_symbols(names)
        installed_c, installed_core = abi_symbols(installed_names)
        if built_c != installed_c or built_core != installed_core:
            raise RuntimeError(
                "manager ABI symbols differ "
                f"(C {len(built_c)}/{len(installed_c)}, core {len(built_core)}/{len(installed_core)})")
        note = f"{len(built_c)} C + {len(built_core)} core symbols match"
        if _size_fingerprint(new_manager) != installed_sizes:
            note += "; std-template symbols differ (compiler version only)"
        return note

    def build_stamp():
        path = reference / "build-stamp.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("exitCode") != 0:
            raise RuntimeError(f"build stamp records exit {data.get('exitCode')}")
        if int(data.get("jobs", 0)) > MAX_BUILD_JOBS:
            raise RuntimeError(f"build jobs exceed {MAX_BUILD_JOBS}: {data.get('jobs')}")
        if data.get("buildType") != "Dump":
            raise RuntimeError(f"build type is not Dump: {data.get('buildType')}")
        return f"jobs={data['jobs']} type={data['buildType']}"

    record("reference artifacts", artifacts)
    record("Dump flag + BlockRT API", blockrt_api)
    record("TD wheel + export_device", td_api)
    record("AscendC device bitcode", ascendc_bitcode)
    record("bishengir tools", bishengir_tools)
    record("ascendc tiling ext", tiling_ext)
    record("aicpu manager ABI", manager_abi)
    record("build stamp", build_stamp)

    failed = [row for row in checks if not row[1]]
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    return 0 if not failed else 1


def _pipeline_report(log_text):
    report = None
    for line in log_text.splitlines():
        marker = "report="
        if "[ALLPASS]" in line and marker in line:
            report = line.split(marker, 1)[1].strip().split()[0]
    return report


def cmd_run(config, layers, device_id, no_precompile, force_golden):
    if layers < 1:
        raise ConfigError(f"--layers must be positive, got {layers}")
    if device_id < 0:
        raise ConfigError(f"--device-id must be nonnegative, got {device_id}")
    reference = Path(config["GLM52_REFERENCE_HOME"])
    td = Path(config["GLM52_TD_SOURCE"])
    overlay = _ensure_overlay(config)
    pipeline = overlay / "python" / "triton_dist" / "mega_kernel_ascend" / "test" / "models" / "test_glm5_2_golden_pipeline.py"
    if not pipeline.is_file():
        raise ConfigError(f"pipeline is missing: {pipeline}")
    command = [
        config["GLM52_PYTHON"], str(pipeline),
        "--runtime-home", str(reference),
        "--golden-root", config["GLM52_GOLDEN_ROOT"],
        "--layers", str(layers),
        "--tokens", config["GLM52_TOKENS"],
        "--tp-size", config["GLM52_TP_SIZE"],
        "--device-id", str(device_id),
        "--scope", config["GLM52_SCOPE"],
        "--verbose",
    ]
    if no_precompile:
        command.append("--no-precompile-dispatch")
    if force_golden:
        command.append("--force-golden")
    log_path = Path(config["GLM52_LOG_ROOT"]) / f"layers{layers}_dev{device_id}_{_stamp()}.log"
    script = "\n".join(
        [
            _activation(config),
            f"cd {shlex.quote(str(td))}",
            "exec " + " ".join(shlex.quote(part) for part in command),
        ]
    )
    started = _now()
    rc = _run(script, td, log_path, _overlay_env(config, overlay))
    text = log_path.read_text(encoding="utf-8", errors="replace")
    report = _pipeline_report(text)
    passed = False
    if report and Path(report).is_file():
        try:
            passed = bool(json.loads(Path(report).read_text(encoding="utf-8"))["passed"])
        except (OSError, ValueError, KeyError):
            passed = False
    allpass = "[ALLPASS]" in text
    ok = rc == 0 and allpass and passed
    print(f"layers={layers} device={device_id} exit={rc} allpass={allpass} passed={passed}")
    print(f"log: {log_path}")
    if report:
        print(f"report: {report}")
    if not ok:
        sys.stderr.write(_tail(log_path) + "\n")
    return 0 if ok else 1


def cmd_config(config):
    for name in sorted(config):
        print(f"{name}={config[name]}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog=PROG, description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="export the pinned reference source")
    prepare.add_argument("--force", action="store_true", help="re-export into a non-empty destination")
    sub.add_parser("build", help="build the reference runtime with -p A5 -t Dump")
    sub.add_parser("build-tiling", help="build the AscendC host tiling extension")
    sub.add_parser("preflight", help="verify artifacts, Dump flag and manager ABI")
    run = sub.add_parser("run", help="run one layer count on an authorized device")
    run.add_argument("--layers", type=int, required=True)
    run.add_argument("--device-id", type=int, required=True)
    run.add_argument("--no-precompile-dispatch", action="store_true")
    run.add_argument("--force-golden", action="store_true")
    all_cmd = sub.add_parser("all", help="run layers 1, 2 and 4 sequentially")
    all_cmd.add_argument("--device-id", type=int, required=True)
    all_cmd.add_argument("--no-precompile-dispatch", action="store_true")
    all_cmd.add_argument("--force-golden", action="store_true")
    sub.add_parser("config", help="print the resolved configuration")
    args = parser.parse_args(argv)

    try:
        config = load_config()
    except ConfigError as exc:
        print(f"{PROG}: {exc}", file=sys.stderr)
        return 2
    try:
        if args.command == "prepare":
            cmd_prepare(config, args.force)
            return 0
        if args.command == "build":
            return cmd_build(config)
        if args.command == "build-tiling":
            return cmd_build_tiling(config)
        if args.command == "preflight":
            return cmd_preflight(config)
        if args.command == "run":
            return cmd_run(config, args.layers, args.device_id,
                           args.no_precompile_dispatch, args.force_golden)
        if args.command == "all":
            for layers in (1, 2, 4):
                if cmd_run(config, layers, args.device_id,
                           args.no_precompile_dispatch, args.force_golden) != 0:
                    return 1
            return 0
        if args.command == "config":
            return cmd_config(config)
    except ConfigError as exc:
        print(f"{PROG}: {exc}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        print(f"{PROG}: command failed: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
