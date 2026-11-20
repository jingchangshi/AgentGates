"""The narrow typed request that selects one source-relative target and candidate.

A promotion Gate runs a workload that is not fixed by the installed adapter: the
caller names a canonical target (`level4/attention_residual`) and the candidate
file inside the source checkout. Both travel as request data, never as a command
line, so the recipe argv stays pinned by the manifest. The receipt binds the
actual candidate bytes, and `verify` re-hashes them, so a receipt can only ever
describe the candidate that was executed.

The grammar is deliberately smaller than a path parser. A target has `/`
separated safe segments; a candidate is a canonical POSIX relative path with no
empty, `.` or `..` segment, and must resolve to an existing regular file under
the source root. The Adapter still performs its own workload validation; this
module only refuses values that are unsafe to pass to it.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import re

from .executor import file_reference
from .manifest import Manifest, Recipe

TARGET_ENV = "AGENT_GATES_TARGET"
CANDIDATE_ENV = "AGENT_GATES_CANDIDATE"
REQUEST_ENVIRONMENT = (TARGET_ENV, CANDIDATE_ENV)

TARGET = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)*\Z")
TARGET_LIMIT = 256
CANDIDATE_LIMIT = 1024


class TargetRequestError(ValueError):
    """A request that cannot be honoured; never an environment verdict."""


def validate_target(value) -> str:
    if not isinstance(value, str) or not value or len(value) > TARGET_LIMIT \
            or "\x00" in value or not TARGET.fullmatch(value):
        raise TargetRequestError(
            "invalid target: expected slash-separated safe segments such as "
            "level4/attention_residual, at most 256 characters")
    return value


def validate_candidate(value) -> str:
    if not isinstance(value, str) or not value or len(value) > CANDIDATE_LIMIT \
            or "\x00" in value or "\\" in value:
        raise TargetRequestError(
            "invalid candidate: expected a workspace-relative POSIX path")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or str(parsed) != value \
            or any(part in {"", ".", ".."} for part in parsed.parts):
        raise TargetRequestError(
            "invalid candidate: expected a canonical relative path with no '.' or '..' segment")
    return value


def candidate_reference(source: Path, relative: str) -> dict:
    """Bind the candidate file's actual bytes, not the path that named them."""
    relative = validate_candidate(relative)
    root = Path(source).resolve(strict=True)
    path = root / relative
    if path.is_symlink():
        raise TargetRequestError("candidate must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise TargetRequestError(f"candidate does not exist: {relative}") from exc
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise TargetRequestError("candidate must be a regular file under the source root")
    return {"relativePath": relative, **file_reference(resolved)}


def resolve_request(manifest: Manifest, ordered: list[str], source: Path, *,
                    target: str | None, candidate: str | None) -> dict | None:
    """Validate the pair against the selected gates and bind the candidate.

    A required route (`"dynamicTarget": true`) refuses to run without the pair.
    An optional route (`"dynamicTarget": "optional"`) accepts the pair when the
    caller supplies one and otherwise resolves its own managed fixed target.
    Any other route refuses the pair, so a request cannot silently become data
    that no recipe consumes. Making the pair both-or-neither keeps one request
    from naming a target without the bytes it refers to.
    """
    required_gates = []
    optional_gates = []
    for gate_id in ordered:
        recipe = manifest.resolve_recipe(manifest.gates[gate_id]["recipeId"], source)
        if recipe.dynamic_target:
            required_gates.append(gate_id)
        elif recipe.dynamic_target_optional:
            optional_gates.append(gate_id)
    if target is None and candidate is None:
        if required_gates:
            raise TargetRequestError(
                "gate " + ", ".join(required_gates)
                + " requires both target and candidate")
        return None
    if target is None or candidate is None:
        raise TargetRequestError("target and candidate must be supplied together")
    if not required_gates and not optional_gates:
        raise TargetRequestError(
            "the selected gates do not accept a target request")
    return {"target": validate_target(target),
            "candidate": candidate_reference(source, candidate)}


def accepts_request(recipe: Recipe) -> bool:
    """Whether this Recipe consumes the pair when the run has one."""
    return recipe.dynamic_target or recipe.dynamic_target_optional


def apply_environment(env: dict, recipe: Recipe, request: dict | None) -> dict:
    """The recipe environment with the request added only where it is declared."""
    if request is None or not accepts_request(recipe):
        return env
    result = dict(env)
    result[TARGET_ENV] = request["target"]
    result[CANDIDATE_ENV] = request["candidate"]["relativePath"]
    return result
