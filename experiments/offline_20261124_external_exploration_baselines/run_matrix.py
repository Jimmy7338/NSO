"""Run the frozen external selection on every registered floor."""

import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "execution/explore"
layouts = json.loads((ROOT / "EXPLORE_PROTOCOL.json").read_text())["layouts"]
OUTPUT.mkdir(parents=True, exist_ok=False)

environment = os.environ | {key: "1" for key in
                            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONDONTWRITEBYTECODE")}


def run(layout):
    launched = time.time()
    with (OUTPUT / f"{layout}.log").open("x") as log:
        result = subprocess.run([sys.executable, str(ROOT / "run_explore.py"), layout,
                                 "--out", str(OUTPUT / layout)], stdout=log,
                                stderr=subprocess.STDOUT, env=environment, check=False)
    receipt = {"layout": layout, "exit_code": result.returncode, "start_unix": launched, "end_unix": time.time()}
    with (OUTPUT / f"{layout}.exit.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(receipt, flush=True)
    return receipt


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run, layouts))
sys.exit(int(any(row["exit_code"] for row in results)))
