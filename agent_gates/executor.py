"""Execute fixed argv through an independent Linux descendant supervisor."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .manifest import digest


def now():
    return datetime.now(timezone.utc).isoformat()


def file_reference(path: Path):
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest(path)}


def executable_identity(argv, cwd, env):
    name = argv[0]
    path = shutil.which(name, path=env.get("PATH")) if "/" not in name else str(Path(cwd) / name)
    if not path or not Path(path).is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"recipe executable unavailable: {name}")
    return file_reference(Path(path))


def execute(argv, *, cwd, env, timeout, log_dir: Path, lease_fds=()):
    log_dir = Path(log_dir).resolve()
    log_dir.mkdir(parents=True, mode=0o700)
    config_path = log_dir / "process-config.json"
    result_path = log_dir / "process-result.json"
    live_path = log_dir / "process-live.json"
    config = {"argv": list(argv), "cwd": str(cwd), "timeout": timeout, "parentPid": os.getpid(),
              "resultPath": str(result_path), "livePath": str(live_path)}
    with config_path.open("x") as stream:
        os.chmod(config_path, 0o600)
        json.dump(config, stream)
    started_at, started = now(), time.monotonic()
    out_path, err_path = log_dir / "stdout.log", log_dir / "stderr.log"
    with out_path.open("xb") as stdout, err_path.open("xb") as stderr:
        os.chmod(out_path, 0o600)
        os.chmod(err_path, 0o600)
        proc = subprocess.Popen([sys.executable, "-I", str(Path(__file__).with_name("process_supervisor.py")), str(config_path)],
                                env=env, stdout=stdout, stderr=stderr, start_new_session=True,
                                pass_fds=tuple(lease_fds))
        try:
            proc.wait(timeout=timeout + 10)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(f"supervisor cleanup incomplete; lease retained; inspect {live_path}") from exc
            stdout.flush()
            stderr.flush()
            os.fsync(stdout.fileno())
            os.fsync(stderr.fileno())
    if proc.returncode or not result_path.is_file():
        raise RuntimeError(f"supervisor failed (exit {proc.returncode}); inspect {err_path}")
    result = json.loads(result_path.read_text())
    return {**result, "argv": list(argv), "cwd": str(cwd), "timeout": timeout,
            "signal": -result["exitCode"] if isinstance(result.get("exitCode"), int) and result["exitCode"] < 0 else None,
            "elapsed": time.monotonic() - started, "startedAt": started_at, "finishedAt": now(),
            "stdout": file_reference(out_path), "stderr": file_reference(err_path),
            "processResult": file_reference(result_path)}
