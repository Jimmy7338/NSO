"""Prepare owned RTAB-Map sessions; source DBs are never opened for writing."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time

DEFAULT_ROOT = Path('/home/ouc/nso_robot/experiments/rtabmap_loop_shadow')


def prepare_session(root, mode, session, source=None, boot_id=None):
    root = Path(root).resolve()
    if mode not in ('mapping', 'localization'):
        raise ValueError('mode must be mapping or localization')
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', session):
        raise ValueError('invalid session name')
    if mode == 'mapping' and source is not None:
        raise ValueError('mapping must start from its own fresh database')
    database_source = None
    if mode == 'localization':
        if source is None:
            raise ValueError('localization requires an explicitly selected owned database')
        database_source = Path(source).resolve()
        if root not in database_source.parents or not database_source.is_file():
            raise ValueError('localization source must be an existing DB under owned module sessions')
    boot = boot_id or Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    output = root / boot / session
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()  # Never overwrite or silently resume a session.
    database = output / 'rtabmap.db'
    source_receipt = None
    if database_source is not None:
        # SQLite online backup gives a consistent isolated snapshot, including a WAL if present.
        with sqlite3.connect(database_source.as_uri() + '?mode=ro', uri=True) as src:
            with sqlite3.connect(str(database)) as dst:
                src.backup(dst)
        source_receipt = dict(path=str(database_source), bytes=database_source.stat().st_size,
                              operation='SQLite read-only source backup into independent session DB')
    receipt = dict(schema='nso.rtabmap_loop_shadow_start.v2', unix_s=time.time(), mode=mode,
                   boot_id=boot, database_path=str(database), source_database=source_receipt,
                   fresh_unknown_mapping=mode == 'mapping', publish_tf=False,
                   navigation_authority=False, actuator=False,
                   storage_profile='linked_keyframes_prepost2_depth16_RVL_v2',
                   feature_image_resolution_changed=True)
    work = Path(__file__).resolve().parents[1]
    paths = [work/'config/rtabmap_loop_shadow.launch', work/'runtime/rtabmap_loop_monitor.py',
             work/'runtime/run_rtabmap_loop_shadow.sh', Path(__file__).resolve()]
    receipt['sources'] = [dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
    (output/'startup.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', nargs='?', default=datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ'))
    parser.add_argument('--mode', choices=['mapping', 'localization'], default='mapping')
    parser.add_argument('--database', type=Path)
    args = parser.parse_args()
    output = prepare_session(DEFAULT_ROOT, args.mode, args.session, args.database)
    print(str(output))
