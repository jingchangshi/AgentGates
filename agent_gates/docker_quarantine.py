"""Centralized fail-closed quarantine for legacy Docker-based adapters and routes."""

QUARANTINE_DOC = "docs/docker-mode-quarantine.md"

QUARANTINED_ADAPTERS = frozenset({"tilelang-hivm", "tilelang-hivm-npu"})

QUARANTINED_GATES = {
    "ascendnpu-ir": frozenset({"ascend-build", "ascend-regression"}),
    "tilelang-hivm": frozenset({"tilelang-hivm-build", "tilelang-hivm-regression"}),
    "tilelang-hivm-npu": frozenset({"tilelang-hivm-npu-sanity"}),
}

QUARANTINED_RECIPES = {
    "ascendnpu-ir": frozenset({"ascendnpu-ir.incremental-build", "ascendnpu-ir.regression"}),
    "tilelang-hivm": frozenset({"tilelang-hivm.build", "tilelang-hivm.regression"}),
    "tilelang-hivm-npu": frozenset({"tilelang-hivm-npu.sanity"}),
}

QUARANTINED_SCRIPT_SUFFIXES = (
    "/scripts/docker_gate.py",
    "/tilelang-hivm-npu/scripts/npu_gate.py",
    "/scripts/tilefoundry_gate.py",
    "/scripts/provision.py",
)


def quarantine_reason(target: str) -> str:
    """Explain why a quarantined Docker target is blocked."""
    return (
        f"DOCKER_MODE_DISABLED: {target} uses quarantined Docker mode and is "
        f"disabled; use host-mode adapters/gates instead (see {QUARANTINE_DOC})"
    )


def is_quarantined_recipe(manifest_id: str, recipe_id: str, argv=()) -> bool:
    """Return True if a recipe or its entrypoint belongs to quarantined Docker mode."""
    if recipe_id in QUARANTINED_RECIPES.get(manifest_id, ()):
        return True
    for token in argv or ():
        norm = str(token).replace("\\", "/")
        if norm.endswith(QUARANTINED_SCRIPT_SUFFIXES):
            return True
    return False
