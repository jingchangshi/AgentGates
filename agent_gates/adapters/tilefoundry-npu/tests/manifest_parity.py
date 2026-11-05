#!/usr/bin/env python3
"""Check that manifest.json and the gate script agree about every check.

A gate declares observationChecks in the manifest and the script reports what it
actually observed. If the two drift apart, the failure is silent and one-directional:

  * a check the manifest names but the script never reports means the Gate can PASS
    while a field the manifest promised does not exist;
  * a check the script reports but the manifest does not name is measured and then
    thrown away, which is a measurement nobody will ever read.

Neither shows up in a normal run, because both look like success. So it is checked
statically: no container, no device, no network.

Exit codes: 0 the two agree, 1 they do not, 2 the inputs could not be read.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE.parent / "manifest.json"
GATE = HERE.parent / "scripts" / "tilefoundry_gate.py"

GATE_FOR_SUFFIX = {"-cpu": "cpu", "-device": "device", "-build": "build"}


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


def main() -> int:
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        source = GATE.read_text(encoding="utf-8")
    except (OSError, json.JSONDecodeError) as error:
        print("could not read the inputs: %s" % error, file=sys.stderr)
        return 2

    reported = reported_keys(source)
    failures = 0

    for gate in manifest.get("gates", []):
        gate_id = gate.get("id", "")
        suffix = next((s for s in GATE_FOR_SUFFIX if gate_id.endswith(s)), None)
        if suffix is None:
            print("  FAIL  %s: no gate script suffix matches this id" % gate_id)
            failures += 1
            continue
        gate_name = GATE_FOR_SUFFIX[suffix]
        declared = set(gate.get("observationChecks", {}))
        actual = reported.get(gate_name)

        if actual is None:
            print("  FAIL  %s: the script never writes observations for %r"
                  % (gate_id, gate_name))
            failures += 1
            continue

        missing = sorted(declared - actual)
        unread = sorted(actual - declared)
        if missing or unread:
            failures += 1
            print("  FAIL  %s" % gate_id)
            if missing:
                print("        declared but never reported: %s" % ", ".join(missing))
            if unread:
                print("        reported but not declared: %s" % ", ".join(unread))
        else:
            print("  PASS  %s (%d checks agree)" % (gate_id, len(declared)))

    # A gate in the manifest with no recipe, or a recipe with no gate, is the same
    # kind of drift one level up.
    recipe_ids = {r.get("id") for r in manifest.get("recipes", [])}
    for gate in manifest.get("gates", []):
        if gate.get("recipeId") not in recipe_ids:
            print("  FAIL  %s: recipeId %r is not among the recipes"
                  % (gate.get("id"), gate.get("recipeId")))
            failures += 1

    # The pinned hash and the live script are the same bargain as the checks: a
    # manifest that pins a digest the script no longer has will refuse to load,
    # but only when someone runs it. Check it here, statically.
    pinned = manifest.get("files", {}).get("scripts/tilefoundry_gate.py")
    live = hashlib.sha256(GATE.read_bytes()).hexdigest()
    if pinned != live:
        print("  FAIL  pinned script hash %s does not match the script %s"
              % (pinned, live))
        failures += 1
    else:
        print("  PASS  the pinned script hash matches the script on disk")

    # A recipe may use only the one supported placeholder. Anything else is
    # expanded by nobody, so the Gate would receive a literal token and fail at
    # run time instead of at load time.
    allowed = {"{pack}"}
    for recipe in manifest.get("recipes", []):
        for argument in recipe.get("argv", []):
            if not isinstance(argument, str):
                failures += 1
                print("  FAIL  %s: argv item is not a string" % recipe.get("id"))
                continue
            for token in re.findall(r"\{[^}]*\}", argument):
                if token not in allowed:
                    failures += 1
                    print("  FAIL  %s: unsupported placeholder %r in %r"
                          % (recipe.get("id"), token, argument))
    device_recipe = next((r for r in manifest.get("recipes", [])
                          if r.get("id") == "tilefoundry-npu.device"), None)
    if device_recipe is None or "--device" not in device_recipe.get("argv", []):
        failures += 1
        print("  FAIL  the device recipe does not select the device Gate")
    else:
        print("  PASS  the device recipe selects the device Gate")

    if failures:
        print("RESULT: FAIL (%d)" % failures)
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())