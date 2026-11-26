"""Launch every frozen TARE floor with isolated worker processes."""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "execution/formal")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((ROOT / "TARE_PROTOCOL.json").read_text())
    layouts = protocol["layouts"]
    env = os.environ | {var: "1" for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                            "MKL_NUM_THREADS", "PYTHONDONTWRITEBYTECODE")}
    env.update(ROS_MASTER_URI="http://127.0.0.1:9", ROS_IP="127.0.0.1")

    def run(layout):
        command = [sys.executable, str(ROOT / "run_tare.py"), layout, "--out", str(output / layout)]
        started = time.time()
        with (output / f"{layout}.log").open("x") as log:
            proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
        receipt = {"layout": layout, "exit_code": proc.returncode, "start_unix": started, "end_unix": time.time()}
        with (output / f"{layout}.exit.json").open("x") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
        print(receipt, flush=True)
        return receipt

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run, layouts))
    return int(any(row["exit_code"] for row in results))


if __name__ == "__main__":
    sys.exit(main())
