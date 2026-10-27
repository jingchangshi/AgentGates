"""Linux process supervisor for a fixed argv Recipe.

The Runner starts this file by absolute path in isolated Python mode. The
supervisor retains inherited lease descriptors, observes Runner death through
pidfd where the interpreter provides one and through a start-time comparison
otherwise, and reaps descendants before it exits. No project module is imported.
"""
import ctypes
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        os.chmod(temporary, 0o600)
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def terminate_tree(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()
    # PR_SET_CHILD_SUBREAPER adopts grandchildren, including those that start
    # a separate session. Kill and reap each adopted child before closing leases.
    children_path = Path(f'/proc/self/task/{os.getpid()}/children')
    while True:
        children = [int(value) for value in children_path.read_text().split()]
        for child in children:
            try:
                os.kill(child, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            os.waitpid(-1, 0)
        except ChildProcessError:
            break


# Prefer `os.pidfd_open` for the parent-liveness check, but do not require it.
# Some CPython builds (for example the 3.11.14 in a conda installation on this host)
# ship without it even though the API exists since 3.9, because it is gated on a
# build-time feature. Calling it unconditionally raised AttributeError on the first
# Recipe there, which every Gate-driven test reported as `supervisor failed with
# exit 1` with no indication of the cause.
PIDFD_SUPPORTED = hasattr(os, 'pidfd_open')


def watch_parent(parent_pid, parent_fd):
    """Report whether the Runner is still alive.

    A pidfd is the authoritative signal and is used when available. Without it,
    fall back to a signal-0 probe. The probe cannot distinguish a dead parent
    from one whose pid was reused, but it keeps the supervisor functional on
    builds that omit `pidfd_open`, and a reused pid only delays the observation.
    """
    if parent_fd is not None:
        return bool(select.select([parent_fd], [], [], 0.05)[0])
    try:
        os.kill(parent_pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


class DependencyError(RuntimeError):
    """A prerequisite of the supervisor is missing. The message names the fix."""


def _start_time(pid):
    """Return this pid's start time, or None if the process is gone.

    Field 22 of `/proc/<pid>/stat` is measured in clock ticks since boot, so it is
    constant for the life of a process. (pid, startTime) therefore identifies one
    process instance: a reused pid always carries a different start time. That is
    the same guarantee a pidfd gives, which matters because the fallback below must
    not mistake a reused pid for a live Runner.
    """
    try:
        stat = Path(f'/proc/{int(pid)}/stat').read_text()
    except (OSError, ValueError, TypeError):
        return None
    # The comm field is parenthesised and may contain spaces or parentheses, so
    # parse from the last ')' rather than splitting the whole line.
    tail = stat.rsplit(')', 1)[-1].split()
    # After comm, field 3 (state) is index 0, so field 22 (starttime) is index 19.
    try:
        return tail[19]
    except IndexError:
        return None


def parent_watcher(parent_pid, pidfd_open):
    """Return a callable reporting whether the Runner is still alive.

    A pidfd is used when the interpreter provides `os.pidfd_open`. Some CPython
    builds omit it because they were compiled against kernel headers without
    `__NR_pidfd_open` (every 3.11 build in a conda installation on this host does),
    so the fallback compares the parent's start time instead. Both answer the same
    question with the same precision; neither reports a reused pid as alive.
    """
    if pidfd_open is not None:
        fd = pidfd_open(parent_pid)
        return fd, lambda: bool(select.select([fd], [], [], 0)[0])
    expected = _start_time(parent_pid)
    if expected is None:
        # No start time and no pidfd: the only safe answer is "treat as gone".
        return None, lambda: True

    def changed():
        return _start_time(parent_pid) != expected

    return None, changed


def require_pidfd():
    """Return `os.pidfd_open` if this build has it, else None.

    Absence is not an error: `parent_watcher` falls back to a start-time comparison
    that is equally exact. This returns the call only so the caller can prefer it.
    """
    return getattr(os, 'pidfd_open', None)


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    result_path = config['resultPath']
    live_path = config['livePath']
    # Linux is an explicit requirement. Do not silently run without cleanup.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'cannot enable descendant reaping')
    parent_pid = config['parentPid']
    if os.getppid() != parent_pid:
        save(result_path, {'exitCode': None, 'timedOut': False, 'interrupted': True,
                           'reason': 'RUNNER_EXITED_BEFORE_LAUNCH'})
        return 0
    parent_fd, parent_gone = parent_watcher(parent_pid, require_pidfd())
    interrupted = False

    def cancel(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    if parent_gone():
        save(result_path, {'exitCode': None, 'timedOut': False, 'interrupted': True,
                           'reason': 'RUNNER_EXITED_BEFORE_LAUNCH'})
        if parent_fd is not None:
            os.close(parent_fd)
        return 0
    try:
        proc = subprocess.Popen(config['argv'], cwd=config['cwd'], shell=False,
                                start_new_session=True)
    except OSError as exc:
        save(result_path, {'exitCode': 127, 'timedOut': False, 'interrupted': False,
                           'reason': 'EXECUTABLE_LAUNCH_FAILED', 'error': str(exc)})
        if parent_fd is not None:
            os.close(parent_fd)
        return 0
    timed_out = False
    reason = None
    deadline = time.monotonic() + config['timeout']
    try:
        save(live_path, {'supervisorPid': os.getpid(), 'childPid': proc.pid,
                         'parentPid': parent_pid, 'state': 'RUNNING'})
        while proc.poll() is None:
            if interrupted or parent_gone():
                interrupted = True
                reason = 'RUNNER_EXITED_OR_CANCELLED'
                break
            if time.monotonic() >= deadline:
                timed_out = True
                reason = 'RECIPE_TIMEOUT'
                break
            time.sleep(0.02)
    finally:
        terminate_tree(proc)
        if parent_fd is not None:
            os.close(parent_fd)
    result = {'exitCode': proc.returncode, 'timedOut': timed_out,
              'interrupted': interrupted, 'reason': reason}
    save(result_path, result)
    save(live_path, {'supervisorPid': os.getpid(), 'childPid': proc.pid,
                     'parentPid': parent_pid, 'state': 'TERMINAL', **result})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
