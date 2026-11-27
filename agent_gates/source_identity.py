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


def resolve_git_common_dir(path: Path) -> Path | None:
    """Return the resolved Git common directory for `path`, or None."""
    result = git(path, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if result.returncode != 0:
        return None
    raw = os.fsdecode(result.stdout.rstrip(b"\n")).strip()
    if not raw:
        return None
    try:
        return Path(raw).resolve(strict=True)
    except OSError:
        return None


def list_linked_worktrees(repo: Path) -> frozenset[Path]:
    """Return all resolved worktree root paths registered with `repo`."""
    result = git(repo, "worktree", "list", "--porcelain", "-z")
    if result.returncode != 0:
        return frozenset()
    prefix = b"worktree "
    paths: set[Path] = set()
    for token in result.stdout.split(b"\0"):
        if not token.startswith(prefix):
            continue
        raw = os.fsdecode(token[len(prefix):])
        if not raw:
            continue
        try:
            paths.add(Path(raw).resolve(strict=True))
        except OSError:
            continue
    return frozenset(paths)


def _canonical_source_directory(value: str | Path) -> Path:
    """Validate and resolve a user-supplied source directory without following symlinks."""
    if isinstance(value, Path):
        raw_str = str(value)
    elif isinstance(value, str):
        raw_str = value
    else:
        raise ValueError("source must be a nonempty path")
    if not raw_str.strip():
        raise ValueError("source must be a nonempty path")
    raw = Path(os.path.expanduser(raw_str))
    if not raw.is_absolute():
        raise ValueError("source must be an absolute path")
    if ".." in raw.parts:
        raise ValueError("source must not contain '..' components")
    if raw.is_symlink():
        raise ValueError("source must not be a symlink")
    if not raw.exists() or not raw.is_dir():
        raise ValueError("source must name an existing directory")
    for parent in raw.parents:
        if parent.is_symlink():
            raise ValueError(f"source parent component is a symlink: {parent}")
    return raw.resolve(strict=True)


def resolve_workspace_source(
    registered_root: Path,
    requested_source: str | Path | None,
    *,
    allow_worktrees: bool = True,
) -> Path:
    """Resolve `requested_source` to either `registered_root` or a linked Git worktree."""
    reg = Path(registered_root).resolve(strict=True)
    if requested_source is None:
        return reg
    candidate = _canonical_source_directory(requested_source)
    if candidate == reg:
        return reg
    if not allow_worktrees:
        raise ValueError("linked git worktrees are disabled for this workspace")

    top = git(candidate, "rev-parse", "--show-toplevel")
    if top.returncode != 0:
        raise ValueError("--source must name a Git worktree top-level directory")
    try:
        top_path = Path(os.fsdecode(top.stdout.rstrip(b"\n"))).resolve(strict=True)
    except OSError as exc:
        raise ValueError("--source must name a Git worktree top-level directory") from exc
    if top_path != candidate:
        raise ValueError("--source must name a Git worktree top-level directory")

    reg_common = resolve_git_common_dir(reg)
    cand_common = resolve_git_common_dir(candidate)
    if reg_common is None or cand_common is None or cand_common != reg_common:
        raise ValueError("source is not a linked git worktree of workspace source_root")
    if candidate not in list_linked_worktrees(reg):
        raise ValueError("source is not registered in git worktree list of workspace")

    git_dir_proc = git(candidate, "rev-parse", "--path-format=absolute", "--git-dir")
    if git_dir_proc.returncode != 0:
        raise ValueError("source does not have a valid linked worktree gitdir")
    git_dir_raw = os.fsdecode(git_dir_proc.stdout.rstrip(b"\n")).strip()
    if not git_dir_raw:
        raise ValueError("source does not have a valid linked worktree gitdir")
    try:
        cand_git_dir = Path(git_dir_raw).resolve(strict=True)
    except OSError as exc:
        raise ValueError("source does not have a valid linked worktree gitdir") from exc
    worktrees_dir = (reg_common / "worktrees").resolve(strict=False)
    if not (cand_git_dir.is_relative_to(worktrees_dir) and cand_git_dir != worktrees_dir):
        raise ValueError("source does not have a valid linked worktree gitdir")

    return candidate


def exclusion_rules(root: Path) -> list[str]:
    """Every rule that could hide a source file from `ls-files --others`.

    `--exclude-standard` honours three ignore sources: the checkout's own
    `.gitignore` files (tracked, so they are already hashed as source), the
    global `core.excludesFile`, and `.git/info/exclude`. The last two are not
    part of the snapshot: a file added to either makes arbitrary bytes vanish
    from the hash while a recorded PASS keeps verifying. They are read here and
    bound into the digest, so changing them changes the source identity.
    """
    common_dir = resolve_git_common_dir(root)
    exclude_path = (common_dir / "info" / "exclude") if common_dir is not None else (root / ".git" / "info" / "exclude")
    paths = [exclude_path]
    configured = git(root, "config", "--get", "core.excludesFile")
    if configured.returncode == 0:
        name = os.fsdecode(configured.stdout.rstrip(b"\n")).strip()
        if name:
            paths.append(Path(os.path.expanduser(name)))
    rules = []
    for path in paths:
        try:
            text = path.read_text(errors="surrogateescape")
        except OSError:
            rules.append(f"{path}\0<absent>")
            continue
        rules.append(f"{path}\0{text}")
    return rules


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
    use_git = own_git and listing.returncode == 0 and (not _nested or head.returncode == 0)
    if use_git:
        if head.returncode:
            raise ValueError("Git source has no readable committed HEAD")
        revision = head.stdout.decode().strip()
        names = sorted(set(os.fsdecode(name) for name in listing.stdout.split(b"\0") if name))
        mode = "git-files"
    else:
        revision = "NO_GIT"
        names = []
        unreadable = []

        def _record(error):
            # os.walk swallows PermissionError by default, so an unreadable
            # directory contributes nothing to the digest or to fileCount. That is
            # fail-open in the worst way: a root-owned 0700 dump tree inside the
            # source root would leave every later receipt verifying as current
            # while its contents went unhashed. Observed for real - a CANN
            # extra-info/ tree landed in a source directory and was invisible to
            # this function. Recording the path forces the digest to move.
            unreadable.append(error.filename or '?')

        for directory, dirs, files in os.walk(root, followlinks=False,
                                            onerror=_record):
            for name in dirs[:]:
                path = Path(directory) / name
                if path.is_symlink():
                    files.append(name)
                    dirs.remove(name)
            names.extend(str((Path(directory) / name).relative_to(root)) for name in files)
        names.sort()
        mode = "filesystem"
    digest = hashlib.sha256(revision.encode() + b"\0")
    # Bind the identity of anything the walk could not list. Without this, an
    # unreadable directory is simply absent from the enumeration and the digest
    # is unchanged - so a source tree that gained a root-owned 0700 tree would
    # keep verifying as current. Recording the path forces the hash to move.
    if mode == "filesystem":
        for name in sorted(unreadable):
            digest.update(json.dumps(["unreadable", name]).encode() + b"\0")
    if use_git:
        # Bind initialized or empty gitlink directories to their index object IDs.
        staged = git(root, "ls-files", "--stage", "-z")
        if staged.returncode:
            raise ValueError("cannot capture Git index identity")
        digest.update(staged.stdout + b"\0")
        # The enumeration below is a function of these rules as much as of the
        # tree, so the rules are part of the identity. Without this, editing
        # `.git/info/exclude` or `core.excludesFile` hides arbitrary bytes from
        # the hash while an already-issued PASS keeps verifying as current.
        for rule in exclusion_rules(root):
            digest.update(rule.encode("utf-8", "surrogateescape") + b"\0")
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
