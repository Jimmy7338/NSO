#!/usr/bin/env python3
"""Preserve a failed native attempt and retry its unchanged sealed actor once."""

import argparse
import os
import shutil
import subprocess
import time

from run import ARMS, FREEZE, HERE, read, sha, write


def retry(layout, arm):
    name = f"{layout}_{arm}"
    execution = HERE / "execution"
    record = read(execution / f"{name}.exit.json")
    assert record["exit_code"] != 0 and record["freeze_sha256"] == sha(FREEZE)
    assert {"layout": layout, "arm": arm} in read(FREEZE)["schedule"]
    native = HERE / "runs" / layout / arm
    assert not (native / "manifest.json").exists()
    archive = HERE / "execution_attempt_002" / name
    archive.mkdir(parents=True, exist_ok=False)
    hbeat = native / "heartbeat.json"
    last_paid = read(hbeat).get("actual_paid") if hbeat.is_file() else None
    for extension in ("launch.json", "exit.json", "log"):
        shutil.copy2(execution / f"{name}.{extension}", archive / extension)
    native.rename(archive / "partial_native")
    write(
        archive / "RECOVERY.json",
        {
            "first_failed_exit_code": record["exit_code"],
            "first_failed_last_heartbeat_paid": last_paid,
            "unchanged_actor_freeze_sha256": sha(FREEZE),
            "decision": "Repeat the sealed case once after technical execution failure. Enable Python faulthandler.",
        },
    )
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", PYTHONFAULTHANDLER="1")
    log = archive / "retry.log"
    started = time.time()
    with log.open("x") as output:
        process = subprocess.Popen(record["command"], cwd=HERE, stdout=output, stderr=subprocess.STDOUT, env=env)
        launch = record | {"pid": process.pid, "started_unix_s": started, "attempt": 2}
        for stale in ("exit_code", "finished_unix_s", "log_sha256", "receipt_path", "receipt_sha256"):
            launch.pop(stale)
        write(archive / "retry_launch.json", launch)
        code = process.wait()
    receipt = native / ("manifest.json" if code == 0 else "failure.json")
    if code and not receipt.exists():
        write(receipt, {"status": "technical_failure", "error": f"Native process exited with {code}",
                        "native_log_sha256": sha(log)})
    retry_record = launch | {
        "exit_code": code,
        "finished_unix_s": time.time(),
        "log_sha256": sha(log),
        "receipt_path": str(receipt),
        "receipt_sha256": sha(receipt),
    }
    write(archive / "retry_exit.json", retry_record)
    print(f"{name} attempt 2 exit {code}", flush=True)
    return code != 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layout")
    parser.add_argument("arm", choices=ARMS)
    args = parser.parse_args()
    raise SystemExit(retry(args.layout, args.arm))


if __name__ == "__main__":
    main()
