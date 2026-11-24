"""Process-wide leases retained by the supervisor until descendants are reaped.

A lease is either **immediate** (the default, and what every non-device resource
uses) or **bounded-wait**. The wait mode exists for the device window: on a
shared machine two of our own runs may legitimately want the same card a second
apart, and a window that is refused instead of awaited turns ordinary
coordination into a spurious BLOCKED_ENV. Waiting is bounded by the recipe's own
declaration, so a lease that never opens still ends as a refusal rather than a
hang.

A window is cooperative: it coordinates AgentGates runs with each other. It does
not, and cannot, exclude other tenants of the machine - another user's process on
the same card is normal and is reported as occupancy, never as a refusal.
"""
from contextlib import ExitStack, contextmanager
import fcntl
import hashlib
import os
from pathlib import Path
import time


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


def lock_root(root: Path | None = None) -> Path:
    """The lease namespace shared by the engine and by Adapter scripts."""
    return Path(root) if root else Path.home() / ".local/state/agent-gates/locks"


DEVICE_LOCK_ROOT_ENV = "AGENT_GATES_DEVICE_LOCK_ROOT"


def device_lock_root(environ=None) -> Path:
    """The one lease namespace for a physical card, shared by V1 and V2.

    Source and build leases follow the selected state directory, but a card does
    not move with it: a V1 process using the default evidence root and a V2
    worker using `<state_root>/locks` must still block each other on `npu:<id>`.
    This root is therefore independent of configuration, with one explicit
    operator override for tests and unusual deployments.
    """
    source = os.environ if environ is None else environ
    explicit = (source.get(DEVICE_LOCK_ROOT_ENV) or "").strip()
    if not explicit:
        return lock_root(None)
    path = Path(explicit).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{DEVICE_LOCK_ROOT_ENV} must be an absolute path")
    return path


def is_shared_device_lock_root(root: Path | None = None, environ=None) -> bool:
    """Return True when root matches the explicit shared AGENT_GATES_DEVICE_LOCK_ROOT."""
    source = os.environ if environ is None else environ
    explicit = (source.get(DEVICE_LOCK_ROOT_ENV) or "").strip()
    if not explicit:
        return False
    if root is None:
        return True
    return Path(root).resolve() == Path(explicit).expanduser().resolve()


def ensure_lock_dir(directory: Path, *, shared: bool = False) -> None:
    """Create a lock directory with 0o1777 (shared) or 0o700 (private) permissions."""
    mode = 0o1777 if shared else 0o700
    directory.mkdir(parents=True, exist_ok=True, mode=mode)
    if shared:
        try:
            if directory.stat().st_uid == os.geteuid():
                os.chmod(directory, 0o1777)
        except OSError:
            pass


def open_lock_file(path: Path, *, shared: bool = False) -> int:
    """Open or create a lock file with 0o666 (shared) or 0o600 (private) permissions."""
    ensure_lock_dir(path.parent, shared=shared)
    file_mode = 0o666 if shared else 0o600
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, file_mode)
    if shared:
        try:
            if os.fstat(fd).st_uid == os.geteuid():
                os.fchmod(fd, 0o666)
        except OSError:
            pass
    return fd


def lock_path(resource: str, root: Path | None = None) -> Path:
    """The file one resource's lease lives in.

    Exposed because the engine publishes it to a Recipe that opens its own
    window: the Adapter script must flock the *same* file the engine would,
    and re-deriving the hash from a second implementation is how two leases
    for one card appear.
    """
    return lock_root(root) / (hashlib.sha256(normalise(resource).encode()).hexdigest() + ".lock")


class ResourceLock:
    def __init__(self, resource: str, root: Path | None = None, timeout: float = 0.0):
        # One namespace per host user, independent of evidence-directory selection.
        self.root = lock_root(root)
        resource = normalise(resource)
        self.path = lock_path(resource, self.root)
        self.resource = resource
        # 0 means "take it now or never"; a positive value is a bounded wait.
        self.timeout = max(0.0, float(timeout))
        self.waited = 0.0
        self.fd = None

    def __enter__(self):
        shared = is_shared_device_lock_root(self.root)
        self.fd = open_lock_file(self.path, shared=shared)
        started = time.monotonic()
        deadline = started + self.timeout
        while True:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.waited = time.monotonic() - started
                return self
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    os.close(self.fd)
                    self.fd = None
                    waited = round(time.monotonic() - started, 3)
                    raise LockBusy(f"resource busy: {self.resource} "
                                   f"(waited {waited}s of {self.timeout}s)") from None
                # Poll rather than block: a signal-interrupted flock would raise
                # into the runner's cleanup path, and the deadline must be ours.
                time.sleep(min(0.05, remaining))

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
def acquire(resources: list[str], *, root: Path | None = None, timeout: float = 0.0):
    with ExitStack() as stack:
        locks = [stack.enter_context(ResourceLock(resource, root, timeout=timeout))
                 for resource in sorted(set(resources))]
        yield tuple(lock.fileno() for lock in locks)
