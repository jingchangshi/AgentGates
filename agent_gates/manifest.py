"""Digest-pinned adapter contracts and fixed recipe resolution."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

ID = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
ITEM_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    pass


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def strict_json(text: str):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value
    def constant(value):
        raise ValueError(f"invalid JSON constant: {value}")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


@dataclass(frozen=True)
class Recipe:
    id: str
    argv: tuple[str, ...]
    cwd: str
    timeout: float
    config_check: tuple[str, ...] = ()
    blocked_exit_codes: tuple[int, ...] = ()


class Manifest:
    def __init__(self, path: Path):
        self.path = Path(path).absolute()
        if self.path.name != "manifest.json" or self.path.is_symlink():
            raise ManifestError("adapter manifest must be a regular manifest.json")
        try:
            raw = strict_json(self.path.read_text())
        except (OSError, ValueError) as exc:
            raise ManifestError(f"invalid adapter manifest: {exc}") from exc
        if not isinstance(raw, dict) or type(raw.get("schemaVersion")) is not int or raw["schemaVersion"] != 1:
            raise ManifestError("unsupported manifest schemaVersion (expected 1)")
        if not isinstance(raw.get("id"), str) or not ID.fullmatch(raw["id"]):
            raise ManifestError("invalid adapter ID")
        # Host execution is the manifest's own consent, so it has to be granted
        # explicitly. Reading a missing block as "allowed" would make an adapter
        # that simply forgot the key run arbitrary commands on the host, and the
        # rest of this schema rejects a merely truthy value everywhere else.
        permissions = raw.get("permissions")
        if not isinstance(permissions, dict) or permissions.get("hostExecution") is not True:
            raise ManifestError("adapter does not declare permissions.hostExecution: true")
        self.root = self.path.parent.resolve()
        self.id = raw["id"]
        self.raw = raw
        self.manifest_hash = digest(self.path)
        self.files = raw.get("files", {})
        if not isinstance(self.files, dict):
            raise ManifestError("manifest files must be an object")
        self.validate_files()
        self.recipes = self._index("recipes")
        self.gates = self._index("gates")
        if not self.recipes or not self.gates:
            raise ManifestError("adapter must declare recipes and gates")
        for item in self.recipes.values():
            self.resolve_recipe(item["id"], self.root)
        for gate in self.gates.values():
            if gate.get("recipeId") not in self.recipes:
                raise ManifestError("gate references an unknown recipe")
            if gate.get("gateType") != "deterministic-test":
                raise ManifestError("only deterministic-test gates are supported")
            checks = gate.get("observationChecks")
            if not isinstance(checks, dict) or not checks or any(not isinstance(k, str) or not k for k in checks):
                raise ManifestError("gate needs nonempty observationChecks")
            dependencies = gate.get("dependsOn", [])
            if not isinstance(dependencies, list) or any(not isinstance(x, str) or x not in self.gates for x in dependencies):
                raise ManifestError("gate dependsOn references an unknown gate")
        self.order(list(self.gates))

    def _index(self, key):
        items = self.raw.get(key)
        if not isinstance(items, list):
            raise ManifestError(f"manifest {key} must be a list")
        indexed = {}
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not ITEM_ID.fullmatch(item["id"]):
                raise ManifestError(f"invalid {key} identifier")
            if item["id"] in indexed:
                raise ManifestError(f"duplicate {key} identifier")
            indexed[item["id"]] = item
        return indexed

    def validate_files(self):
        for name, expected in self.files.items():
            if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
                raise ManifestError("pinned file path must be relative inside adapter")
            target = self.root / name
            if any(part.is_symlink() for part in [target, *target.parents] if part != self.root and part.is_relative_to(self.root)):
                raise ManifestError(f"pinned file must not be a symlink: {name}")
            if not isinstance(expected, str) or not SHA256.fullmatch(expected) or not target.is_file() or digest(target) != expected:
                raise ManifestError(f"adapter file hash mismatch: {name}")
        if digest(self.path) != self.manifest_hash:
            raise ManifestError("manifest changed after loading")

    def _argv(self, argv, *, check=False):
        if not isinstance(argv, list) or not argv or len(argv) > 128 or any(not isinstance(x, str) or not x or "\x00" in x for x in argv):
            raise ManifestError("recipe argv must be a nonempty list of strings")
        if check and not any("{pack}" in x for x in argv):
            raise ManifestError("configCheck must name a pinned {pack} file")
        expanded = []
        for argument in argv:
            if "{pack}" in argument:
                if not argument.startswith("{pack}/") or argument.count("{pack}") != 1:
                    raise ManifestError("{pack} must name a pinned file")
                relative = argument[len("{pack}/"):]
                if relative not in self.files:
                    raise ManifestError("{pack} path must be a pinned manifest file")
                expanded.append(str(self.root / relative))
            else:
                if "{" in argument or "}" in argument:
                    raise ManifestError("unsupported recipe placeholder")
                expanded.append(argument)
        return tuple(expanded)

    def resolve_recipe(self, recipe_id: str, source: Path) -> Recipe:
        item = self.recipes[recipe_id]
        cwd = item.get("cwd", "pack")
        if cwd not in {"pack", "source"}:
            raise ManifestError("recipe cwd must select pack or source")
        timeout = item.get("timeout", 60)
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 86400:
            raise ManifestError("invalid recipe timeout")
        check = item.get("configCheck", [])
        if not isinstance(check, list):
            raise ManifestError("configCheck must be a list")
        blocked = item.get("blockedExitCodes", [])
        if not isinstance(blocked, list) or any(type(code) is not int or not 1 <= code <= 255 for code in blocked) \
                or len(set(blocked)) != len(blocked):
            raise ManifestError("blockedExitCodes must be distinct exit codes between 1 and 255")
        return Recipe(recipe_id, self._argv(item.get("argv")), str(Path(source).resolve() if cwd == "source" else self.root),
                      float(timeout), self._argv(check, check=True) if check else (), tuple(blocked))

    def order(self, selected: list[str]) -> list[str]:
        ordered, active, done = [], set(), set()
        def visit(gate_id):
            if gate_id not in self.gates:
                raise ManifestError(f"unknown gate: {gate_id}")
            if gate_id in active:
                raise ManifestError("gate dependencies contain a cycle")
            if gate_id in done:
                return
            active.add(gate_id)
            for dependency in self.gates[gate_id].get("dependsOn", []):
                visit(dependency)
            active.remove(gate_id)
            done.add(gate_id)
            ordered.append(gate_id)
        for gate_id in selected:
            visit(gate_id)
        return ordered
