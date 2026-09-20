import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

root = Path('/root/NSO')
task = root / 'tmp/v38-tex'
env = dict(os.environ)
env['TECTONIC_CACHE_DIR'] = str(task / 'cache')
env['XDG_CACHE_HOME'] = str(task / 'cache')
env['XDG_CONFIG_HOME'] = str(task / 'config')
cmd = [str(task / 'bin/tectonic'), '--keep-logs', '--keep-intermediates', '--print', '--outdir', str(task / 'smoke'), str(task / 'smoke/minimal.tex')]
free_start = shutil.disk_usage(root).free
minimum = free_start
started = time.monotonic()
reason = None
with (task / 'smoke/build.stdout.txt').open('w') as log:
    process = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    while process.poll() is None:
        free = shutil.disk_usage(root).free
        minimum = min(minimum, free)
        if free < 80 * 1024**2:
            reason = '80_MiB_guard_preserves_64_MiB_reserve'
        if time.monotonic() - started > 240:
            reason = '240_second_attempt_limit'
        if reason:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            break
        time.sleep(0.05)
receipt = {'command': cmd, 'environment_overrides': {k:env[k] for k in ['TECTONIC_CACHE_DIR','XDG_CACHE_HOME','XDG_CONFIG_HOME']}, 'exit_code':process.returncode, 'stop_reason':reason,'free_start_bytes':free_start,'minimum_free_observed_bytes':minimum,'free_end_bytes':shutil.disk_usage(root).free,'elapsed_s':time.monotonic()-started}
(task / 'smoke/build_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
print((task / 'smoke/build.stdout.txt').read_text()[-8000:])
