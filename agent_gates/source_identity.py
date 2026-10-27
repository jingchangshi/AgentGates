"""Hash source bytes, including dirty and untracked files, without Git filters."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

GIT_SAFE = ("-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null", "-c", "core.pager=cat",
            "-c", "core.untrackedCache=false")


def git(repo: Path, *args):
    environment = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    return subprocess.run(["git", *GIT_SAFE, "-C", str(repo), *args], capture_output=True, env=environment, timeout=120)


def source_root(path: Path) -> Path:
    path = Path(path).resolve(strict=True)
    if not path.is_dir():
        raise ValueError("source must be a directory")
    result = git(path, "rev-parse", "--show-toplevel")
    if result.returncode == 0:
        root = Path(os.fsdecode(result.stdout.rstrip(b"\n"))).resolve(strict=True)
        if root != path:
            raise ValueError("--source must name the repository root")
    elif (path / ".git").exists():
        raise ValueError("cannot identify Git source root")
    return path


def snapshot(root: Path, *, _nested=False) -> dict:
    root = Path(root).resolve(strict=True) if _nested else source_root(root)
    top = git(root, "rev-parse", "--show-toplevel")
    own_git = top.returncode == 0 and Path(os.fsdecode(top.stdout.rstrip(b"\n"))).resolve() == root
    listing = git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    head = git(root, "rev-parse", "HEAD")
    if own_git and listing.returncode == 0:
        if head.returncode:
            raise ValueError("Git source has no readable committed HEAD")
        revision = head.stdout.decode().strip()
        names = sorted(set(os.fsdecode(name) for name in listing.stdout.split(b"\0") if name))
        mode = "git-files"
    else:
        revision = "NO_GIT"
        names = []
        for directory, dirs, files in os.walk(root, followlinks=False):
            for name in dirs[:]:
                path = Path(directory) / name
                if path.is_symlink():
                    files.append(name)
                    dirs.remove(name)
            names.extend(str((Path(directory) / name).relative_to(root)) for name in files)
        names.sort()
        mode = "filesystem"
    digest = hashlib.sha256(revision.encode() + b"\0")
    if own_git:
        # Bind initialized or empty gitlink directories to their index object IDs.
        staged = git(root, "ls-files", "--stage", "-z")
        if staged.returncode:
            raise ValueError("cannot capture Git index identity")
        digest.update(staged.stdout + b"\0")
    external = []
    for name in names:
        path = root / name
        if not path.is_relative_to(root) or ".." in Path(name).parts:
            raise ValueError("source file escapes root")
        try:
            info = path.lstat()
        except FileNotFoundError:
            digest.update(json.dumps([name, "deleted"]).encode() + b"\0")
            continue
        metadata = [name, stat.S_IMODE(info.st_mode)]
        if stat.S_ISLNK(info.st_mode):
            link = os.readlink(path)
            metadata.extend(["symlink", link])
            # Bind the link text without traversing it. Compiler test fixtures
            # contain dangling links and editor skill directories may be external.
            # Relevant executable products are separately resolved and hashed.
            external.append(name)
        elif stat.S_ISDIR(info.st_mode):
            # Gitlinks/submodules need their own complete byte identity.
            nested = snapshot(path, _nested=True)
            metadata.extend(["submodule", nested["hash"]])
        elif stat.S_ISREG(info.st_mode):
            if not path.resolve(strict=True).is_relative_to(root):
                raise ValueError(f"source file traverses an external symlink: {name}")
            metadata.extend(["file", info.st_size])
        else:
            raise ValueError(f"unsupported source file type: {name}")
        digest.update(json.dumps(metadata, ensure_ascii=True, separators=(",", ":")).encode() + b"\0")
        if stat.S_ISREG(info.st_mode):
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
    return {"root": str(root), "head": revision, "hash": digest.hexdigest(), "mode": mode,
            "fileCount": len(names), "symlinks": external}
