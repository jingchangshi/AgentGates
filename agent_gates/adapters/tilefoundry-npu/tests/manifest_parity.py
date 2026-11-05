#!/usr/bin/env python3
"""ENV-06 / C10: the manifest and the thing it describes must be the SAME closure.

A manifest is the adapter's identity: which files are pinned, which recipes run,
in what order, and which observations a Gate promises. Four kinds of drift are
invisible in a normal run, because all four look like success:

  * a helper that a canonical recipe executes is not pinned -- the recipe runs,
    but the identity a receipt records does not cover the code that ran;
  * a recipe that runs a script which imports a sibling that is not pinned --
    same hole, one level in;
  * a dependency edge missing from the declared order -- provisioning, configure
    and verify can then run in any order and still 'pass';
  * a configCheck that MUTATES the environment it is supposed to only parse.

So the checks here are static (no container, no device, no network) AND
mutation-tested: each check is re-run against a deliberately broken copy, and a
mutation that is NOT detected is itself a failure. A checker nobody has seen
fail is not evidence.

Exit codes: 0 the two agree, 1 they do not, 2 the inputs could not be read.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACK = HERE.parent
MANIFEST = PACK / "manifest.json"
SCRIPTS = PACK / "scripts"
GATE = SCRIPTS / "tilefoundry_gate.py"

GATE_FOR_SUFFIX = {"-cpu": "cpu", "-device": "device", "-build": "build"}

PLACEHOLDER = "{pack}/"

#: configCheck may ONLY inspect. A selector outside this set is either a mutation
#: or an unknown flag that argparse would refuse at run time; both belong here.
READ_ONLY_SELECTORS = {"--help", "--check-config", "--check", "--facts", "--findings"}
MUTATION_FLAGS = {"--apply", "--start", "--inject-error", "--cpu", "--device",
                  "--build", "--deployment-attest"}

#: Machine-local, git-ignored configuration. It is deliberately NOT part of the
#: adapter's tracked identity, so it must never be pinned.
MACHINE_LOCAL = {"env.sh"}

#: The EXECUTION CLOSURE this adapter must pin, spelled out rather than derived:
#: an omission has to be a visible edit to this list, not a silently shorter scan.
REQUIRED_CLOSURE = (
    "scripts/tilefoundry_gate.py",
    "scripts/configure.py",
    "scripts/env_context.py",
    "scripts/provision.py",
    "scripts/verify_environment.py",
    "deployment_profile.json",
    "tests/provision_negatives.sh",
    "tests/verify_negatives.sh",
    "tests/test_no_bare_device_subprocess.py",
    "tests/test_configure_negatives.py",
    "tests/test_verify_tf_cpu_chain.py",
    "tests/manifest_parity.py",
    "tests/fixtures/docker-inspect-foreign.json",
)

HELPER_RECIPES = ("tilefoundry-npu.provision",
                  "tilefoundry-npu.configure",
                  "tilefoundry-npu.verify")
GATE_RECIPES = ("tilefoundry-npu.build", "tilefoundry-npu.cpu",
                "tilefoundry-npu.device")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------- observation parity

def assigned_dict(tree, name: str, before_line: int):
    """The nearest dict assigned to `name` BEFORE the given line.

    Scoped by position on purpose. Each Gate assigns its own `observations`;
    returning the last assignment in the file makes every Gate report the final
    Gate's keys, which looks like agreement and is not.
    """
    found = None
    best = -1
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        if node.lineno >= before_line:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name and node.lineno > best:
                best = node.lineno
                found = node.value
    return found


def reported_keys(source: str) -> dict:
    """Return {gate_name: {observation keys}} by reading the write_evidence calls.

    Read from the AST rather than by regex, so a commented-out key or a key in a
    docstring cannot count as reported.
    """
    tree = ast.parse(source)
    found: dict = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "id", getattr(func, "attr", ""))
        if name != "write_evidence":
            continue
        if len(node.args) < 3:
            continue
        gate_node = node.args[1]
        if not isinstance(gate_node, ast.Constant) or not isinstance(gate_node.value, str):
            continue
        gate_name = gate_node.value
        record = node.args[2]
        if not isinstance(record, ast.Dict):
            continue
        keys = set()
        for key_node, value_node in zip(record.keys, record.values):
            if not isinstance(key_node, ast.Constant):
                continue
            if key_node.value != "observations":
                continue
            # "observations" is usually a NAME referring to a dict assigned just
            # above, not an inline literal. Resolving the name matters: reading only
            # inline dicts finds zero keys and the check silently passes on an empty
            # set, which is exactly the drift this test exists to catch.
            resolved = value_node
            if isinstance(resolved, ast.Name):
                resolved = assigned_dict(tree, resolved.id, node.lineno)
            if isinstance(resolved, ast.Dict):
                for obs_key in resolved.keys:
                    if isinstance(obs_key, ast.Constant) and isinstance(obs_key.value, str):
                        keys.add(obs_key.value)
        found[gate_name] = keys
    return found


def check_observations(manifest: dict, source: str) -> list:
    """Every declared check is reported, and every reported check is declared."""
    failures = []
    reported = reported_keys(source)
    for gate in manifest.get("gates", []):
        gate_id = gate.get("id", "")
        suffix = next((s for s in GATE_FOR_SUFFIX if gate_id.endswith(s)), None)
        if suffix is None:
            failures.append("%s: no gate script suffix matches this id" % gate_id)
            continue
        gate_name = GATE_FOR_SUFFIX[suffix]
        declared = set(gate.get("observationChecks", {}))
        actual = reported.get(gate_name)
        if actual is None:
            failures.append("%s: the script never writes observations for %r"
                            % (gate_id, gate_name))
            continue
        missing = sorted(declared - actual)
        unread = sorted(actual - declared)
        if missing or unread:
            failures.append("%s: declared-but-not-reported=%r reported-but-not-declared=%r"
                            % (gate_id, missing, unread))
    return failures


# ------------------------------------------------------------- recipe graph

def _recipe_deps(recipe: dict) -> list:
    deps = recipe.get("dependsOn") or []
    return deps if isinstance(deps, list) else []


def _reaches(manifest: dict, start: str, target: str, seen=None) -> bool:
    seen = set() if seen is None else seen
    if start == target:
        return True
    if start in seen:
        return False
    seen.add(start)
    recipe = _recipe_by_id(manifest, start)
    if recipe is None:
        return False
    return any(_reaches(manifest, dep, target, seen) for dep in _recipe_deps(recipe))


def _recipe_by_id(manifest: dict, recipe_id: str):
    for recipe in manifest.get("recipes", []):
        if recipe.get("id") == recipe_id:
            return recipe
    return None


def check_recipe_graph(manifest: dict) -> list:
    """Explicit, ordered dependencies: provision -> configure -> verify -> gates."""
    failures = []
    recipes = manifest.get("recipes", [])
    ids = [recipe.get("id") for recipe in recipes]
    if len(set(ids)) != len(ids):
        failures.append("duplicate recipe id")
    known = set(ids)
    for recipe in recipes:
        deps = _recipe_deps(recipe)
        if not isinstance(deps, list) or any(not isinstance(x, str) for x in deps):
            failures.append("%s: dependsOn must be a list of recipe ids"
                            % recipe.get("id"))
            continue
        for dep in deps:
            if dep not in known:
                failures.append("%s: dependsOn references unknown recipe %r"
                                % (recipe.get("id"), dep))
            if dep == recipe.get("id"):
                failures.append("%s: dependsOn itself" % recipe.get("id"))
    # A cycle makes "order" meaningless; detect it explicitly.
    visiting, done = set(), set()

    def visit(recipe_id):
        if recipe_id in done:
            return False
        if recipe_id in visiting:
            return True
        visiting.add(recipe_id)
        recipe = _recipe_by_id(manifest, recipe_id)
        for dep in _recipe_deps(recipe or {}):
            if visit(dep):
                return True
        visiting.discard(recipe_id)
        done.add(recipe_id)
        return False

    for recipe_id in ids:
        if visit(recipe_id):
            failures.append("recipe dependency cycle through %s" % recipe_id)
            break

    for recipe_id in HELPER_RECIPES:
        if recipe_id not in known:
            failures.append("missing canonical helper recipe %s" % recipe_id)
    if all(r in known for r in HELPER_RECIPES):
        if "tilefoundry-npu.provision" not in _recipe_deps(
                _recipe_by_id(manifest, "tilefoundry-npu.configure") or {}):
            failures.append("configure must declare dependsOn provision")
        if "tilefoundry-npu.configure" not in _recipe_deps(
                _recipe_by_id(manifest, "tilefoundry-npu.verify") or {}):
            failures.append("verify must declare dependsOn configure")
        for recipe_id in HELPER_RECIPES:
            recipe = _recipe_by_id(manifest, recipe_id) or {}
            if recipe.get("device") != "none":
                failures.append("%s must be host-only (device: none)" % recipe_id)
        for recipe_id in GATE_RECIPES:
            recipe = _recipe_by_id(manifest, recipe_id)
            if recipe is None:
                failures.append("missing canonical gate recipe %s" % recipe_id)
                continue
            if not _reaches(manifest, recipe_id, "tilefoundry-npu.verify"):
                failures.append("%s must declare a dependency path to verify"
                                % recipe_id)
    return failures


# ---------------------------------------------------------- files / closure

def referenced_pack_files(manifest: dict) -> list:
    """(owner, field, relative path) for every {pack}/ reference."""
    found = []
    for recipe in manifest.get("recipes", []):
        for field in ("argv", "configCheck"):
            for argument in recipe.get(field) or []:
                if isinstance(argument, str) and argument.startswith(PLACEHOLDER):
                    found.append((recipe.get("id"), field, argument[len(PLACEHOLDER):]))
    for capability in (manifest.get("permissions", {})
                       .get("hostReadOnlyCapabilities") or []):
        for argument in capability.get("argv") or []:
            if isinstance(argument, str) and argument.startswith(PLACEHOLDER):
                found.append((capability.get("id"), "argv",
                              argument[len(PLACEHOLDER):]))
    return found


def executed_scripts(manifest: dict) -> list:
    """Pinned .py files that a canonical recipe actually executes."""
    scripts = set()
    for _owner, _field, relative in referenced_pack_files(manifest):
        if relative.startswith("scripts/") and relative.endswith(".py"):
            scripts.add(relative)
    return sorted(scripts)


def sibling_imports(path: Path) -> set:
    """Top-level module names imported by *path*."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def literal_pack_files(path: Path) -> set:
    """Pack-root file names named as literals in *path* (e.g. a profile read)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
            if "/" in text or "\\" in text or not text.endswith(".json"):
                continue
            if (PACK / text).is_file():
                names.add(text)
    return names


def check_config_checks(manifest: dict) -> list:
    """A configCheck parses; it must not change the environment it validates."""
    failures = []
    for recipe in manifest.get("recipes", []):
        recipe_id = recipe.get("id")
        check = recipe.get("configCheck")
        if not isinstance(check, list) or not check:
            failures.append("%s: configCheck is missing" % recipe_id)
            continue
        if not any(isinstance(arg, str) and arg.startswith(PLACEHOLDER) for arg in check):
            failures.append("%s: configCheck must name a pinned {pack} file" % recipe_id)
        selectors = [arg for arg in check if arg in READ_ONLY_SELECTORS]
        if len(selectors) != 1:
            failures.append("%s: configCheck must select exactly one read-only "
                            "operation, got %r" % (recipe_id, selectors))
        mutating = [arg for arg in check if arg in MUTATION_FLAGS]
        if mutating:
            failures.append("%s: configCheck may not mutate the environment: %r"
                            % (recipe_id, mutating))
    return failures


def check_files(manifest: dict) -> list:
    failures = []
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        return ["manifest has no pinned files at all"]

    for name, expected in files.items():
        if name in MACHINE_LOCAL:
            failures.append("%s is machine-local and must not be pinned" % name)
            continue
        target = PACK / name
        if not target.is_file():
            failures.append("pinned file is missing: %s" % name)
        elif not isinstance(expected, str) or sha256(target) != expected:
            failures.append("pinned hash mismatch: %s" % name)

    for owner, field, relative in referenced_pack_files(manifest):
        if relative not in files:
            failures.append("%s.%s executes unpinned file %r"
                            % (owner, field, relative))

    for name in REQUIRED_CLOSURE:
        if name not in files:
            failures.append("execution closure does not pin %s" % name)

    # Imports of an EXECUTED script are part of what ran, so they must be pinned.
    for relative in executed_scripts(manifest):
        script = PACK / relative
        if not script.is_file():
            continue
        for imported in sibling_imports(script):
            sibling = "scripts/%s.py" % imported
            if (PACK / sibling).is_file() and sibling not in files:
                failures.append("%s imports unpinned sibling %s" % (relative, sibling))
        for data in literal_pack_files(script):
            if data not in files:
                failures.append("%s reads unpinned data file %s" % (relative, data))
    return failures


# ------------------------------------------------------------------- driver

def check_manifest(manifest: dict, source: str) -> list:
    return (check_files(manifest) + check_config_checks(manifest)
            + check_recipe_graph(manifest) + check_observations(manifest, source))


def mutation_checks(manifest: dict, source: str) -> list:
    """Every checker must be able to FAIL: mutate a copy and require detection."""
    failures = []

    def expect_detected(label, mutate, predicate=None):
        mutated = copy.deepcopy(manifest)
        mutate(mutated)
        found = check_manifest(mutated, source)
        if predicate is not None:
            found = [failure for failure in found if predicate(failure)]
        if found:
            print("  PASS  mutation detected: %s" % label)
        else:
            failures.append("mutation NOT detected: %s" % label)

    def drop_helper(mutated):
        mutated["files"].pop("scripts/env_context.py", None)

    def drop_recipe_target(mutated):
        mutated["files"].pop("scripts/provision.py", None)

    def drop_checker(mutated):
        gate = next(g for g in mutated["gates"]
                    if g["id"] == "tilefoundry-npu-device")
        gate["observationChecks"].pop(sorted(gate["observationChecks"])[0], None)

    def drop_dependency(mutated):
        recipe = next(r for r in mutated["recipes"]
                      if r["id"] == "tilefoundry-npu.configure")
        recipe["dependsOn"] = []

    def mutate_config_check(mutated):
        recipe = next(r for r in mutated["recipes"]
                      if r["id"] == "tilefoundry-npu.provision")
        recipe["configCheck"] = ["python3", "-B",
                                 PLACEHOLDER + "scripts/provision.py", "--apply"]

    expect_detected("a pinned helper deleted from files", drop_helper)
    expect_detected("a canonical recipe's target unpinned", drop_recipe_target)
    expect_detected("an observationCheck deleted", drop_checker,
                    lambda failure: "declared-but-not-reported" in failure
                    or "reported-but-not-declared" in failure)
    expect_detected("a required dependency edge deleted", drop_dependency,
                    lambda failure: "dependsOn" in failure)
    expect_detected("a configCheck turned into a mutation", mutate_config_check,
                    lambda failure: "configCheck may not mutate" in failure)
    return failures


def main() -> int:
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        source = GATE.read_text(encoding="utf-8")
    except (OSError, json.JSONDecodeError) as error:
        print("could not read the inputs: %s" % error, file=sys.stderr)
        return 2

    failures = check_manifest(manifest, source)

    # Readable per-section results, so a failure names its own contract.
    for gate in manifest.get("gates", []):
        gate_id = gate.get("id", "")
        declared = gate.get("observationChecks", {})
        if any(failure.startswith(gate_id) for failure in failures):
            print("  FAIL  %s" % gate_id)
            for failure in failures:
                if failure.startswith(gate_id):
                    print("        %s" % failure.split(": ", 1)[-1])
        else:
            print("  PASS  %s (%d checks agree)" % (gate_id, len(declared)))
    for label, checker in (("pinned files and execution closure", check_files),
                           ("configChecks are read-only", check_config_checks),
                           ("recipe dependency order", check_recipe_graph)):
        found = checker(manifest)
        if found:
            print("  FAIL  %s" % label)
            for failure in found:
                print("        %s" % failure)
        else:
            print("  PASS  %s" % label)

    # A recipe may use only the one supported placeholder. Anything else is
    # expanded by nobody, so the Gate would receive a literal token and fail at
    # run time instead of at load time.
    allowed = {"{pack}"}
    for recipe in manifest.get("recipes", []):
        for argument in recipe.get("argv", []) + recipe.get("configCheck", []):
            if not isinstance(argument, str):
                failures.append("%s: argv item is not a string" % recipe.get("id"))
                continue
            for token in re.findall(r"\{[^}]*\}", argument):
                if token not in allowed:
                    failures.append("%s: unsupported placeholder %r in %r"
                                    % (recipe.get("id"), token, argument))
    device_recipe = next((r for r in manifest.get("recipes", [])
                          if r.get("id") == "tilefoundry-npu.device"), None)
    if device_recipe is None or "--device" not in device_recipe.get("argv", []):
        failures.append("the device recipe does not select the device Gate")

    print("== the checkers can fail ==")
    failures.extend(mutation_checks(manifest, source))

    if failures:
        print("RESULT: FAIL (%d)" % len(failures))
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
