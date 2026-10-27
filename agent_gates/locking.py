"""Process-wide leases retained by the supervisor until descendants are reaped."""
from contextlib import ExitStack, contextmanager
import fcntl
import hashlib
import os
from pathlib import Path


class LockBusy(RuntimeError):
    pass


def normalise(resource: str) -> str:
    """One lease per on-disk resource, however its path is spelled.

    Keys are hashed strings, so `source:/x`, `source:/x/` and `source:/x/.`
    used to be three different leases for one directory and two runs could
    share a build tree. Any absolute path after the `prefix:` is resolved;
    non-path keys such as `npu:3` are left exactly as written.
    """
    prefix, separator, value = resource.partition(":")
    if separator and value.startswith("/"):
        return f"{prefix}:{os.path.realpath(value)}"
    return resource


class ResourceLock:
    def __init__(self, resource: str, root: Path | None = None):
        # One namespace per host user, independent of evidence-directory selection.
        self.root = Path(root) if root else Path.home() / ".local/state/agent-gates/locks"
        resource = normalise(resource)
        self.path = self.root / (hashlib.sha256(resource.encode()).hexdigest() + ".lock")
        self.resource = resource
        self.fd = None

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW
        self.fd = os.open(self.path, flags, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(self.fd)
            self.fd = None
            raise LockBusy(f"resource busy: {self.resource}") from exc
        return self

    def fileno(self):
        if self.fd is None:
            raise RuntimeError("resource lock not acquired")
        return self.fd

    def __exit__(self, *_):
        if self.fd is not None:
            # No LOCK_UN: an inherited supervisor descriptor must retain the lease.
            os.close(self.fd)
            self.fd = None


@contextmanager
def acquire(resources: list[str], *, root: Path | None = None):
    with ExitStack() as stack:
        locks = [stack.enter_context(ResourceLock(resource, root)) for resource in sorted(set(resources))]
        yield tuple(lock.fileno() for lock in locks)
