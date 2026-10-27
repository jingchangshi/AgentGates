"""Only installed adapters are executable; arbitrary IDs cannot select paths."""
from pathlib import Path
from .manifest import ID, Manifest, ManifestError


class Registry:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root is not None else Path(__file__).parent / "adapters"

    def get(self, adapter_id: str) -> Manifest:
        if not isinstance(adapter_id, str) or not ID.fullmatch(adapter_id):
            raise ManifestError("invalid adapter ID")
        directory = self.root / adapter_id
        if directory.is_symlink() or not directory.is_dir():
            raise ManifestError(f"unknown adapter: {adapter_id}")
        manifest = Manifest(directory / "manifest.json")
        if manifest.id != adapter_id:
            raise ManifestError("adapter directory and manifest ID disagree")
        return manifest

    def list(self) -> list[Manifest]:
        return [self.get(path.parent.name) for path in sorted(self.root.glob("*/manifest.json"))]
