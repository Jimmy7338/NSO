"""JSONL transport for the unchanged paper controller; no ROS or motion APIs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

VENDOR = Path(__file__).resolve().parents[1] / 'vendor_nso'
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ReferenceSession:
    def __init__(self, task_path, mode='G', resume_log=None):
        from .task_bundle import load_task
        self.task = load_task(task_path)
        self.core = self.task.controller(mode=mode)
        self.resume_source = None
        if resume_log is not None:
            self._resume(resume_log, mode)

    @staticmethod
    def _same(actual, expected, label):
        canonical = lambda value: json.dumps(value, sort_keys=True, allow_nan=False)
        if canonical(actual) != canonical(expected):
            raise ValueError('resume replay differs: ' + label)

    def _resume(self, path, mode, ancestry=(), expected_sha256=None):
        path = Path(path).resolve()
        if path in ancestry:
            raise ValueError('resume log ancestry cycle')
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError('nested resume source SHA256 mismatch')
        rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        if not rows or rows[0].get('request', {}).get('method') != 'init':
            raise ValueError('resume log must begin with successful init')
        initialized = False
        for index, row in enumerate(rows):
            request, response = row['request'], row['response']
            if response.get('ok') is not True or response.get('id') != request.get('id'):
                raise ValueError('resume refuses failed or mismatched RPC')
            method, params = request['method'], request.get('params', {})
            result = response['result']
            if method == 'init':
                if initialized or index != 0:
                    raise ValueError('resume log has repeated init')
                self._same(params.get('mode', 'G'), mode, 'init mode')
                self._same(result['summary']['mode'], mode, 'recorded mode')
                self._same(result['task']['manifest']['sha256'],
                           self.task.receipt['manifest']['sha256'], 'task manifest SHA256')
                if Path(params['task_path']).resolve() != Path(self.task.receipt['manifest']['path']):
                    raise ValueError('resume task path differs')
                nested = params.get('resume_log')
                if nested is not None:
                    source = result.get('resume_source')
                    if not isinstance(source, dict) or not isinstance(source.get('sha256'), str):
                        raise ValueError('nested resume requires recorded source identity')
                    if Path(nested).resolve() != Path(source['path']).resolve():
                        raise ValueError('nested resume source path differs')
                    self._resume(nested, mode, ancestry + (path,), source['sha256'])
                identity = self.identity()
                for key in ('task', 'poses', 'anchor', 'binding', 'calibration', 'summary'):
                    self._same(identity[key], result[key], 'init ' + key)
                for key in ('current_state', 'pending', 'selected', 'resume_source'):
                    if nested is not None or key in result:
                        self._same(identity[key], result[key], 'init ' + key)
                initialized = True
            elif method == 'shutdown':
                break
            elif method == 'observe':
                if file_digest(params['path']) != result.get('source_file_sha256'):
                    raise ValueError('resume observation source_file_sha256 mismatch')
                self._same(self.observe(**params), result, 'observe response')
            elif method == 'next_action':
                if params:
                    raise ValueError('unexpected next_action parameters')
                self._same(self.next_action(), result, 'next_action response')
            elif method == 'summary':
                self._same(self.core.summary(), result, 'summary response')
            else:
                raise ValueError('resume refuses unknown RPC: ' + method)
        self.resume_source = dict(path=str(path), sha256=digest)

    def identity(self):
        return dict(task=self.task.receipt, poses=[list(p) for p in self.core.poses],
                    anchor=self.core.anchor, binding=self.task.binding,
                    calibration=self.task.calibration, summary=self.core.summary(),
                    current_state=self.core.state, pending=self.core.pending,
                    selected=self.core.selected, resume_source=self.resume_source)

    def observe(self, path, step, pose, action=None):
        import numpy as np
        from nso.cpu_four_modules_v35 import ObservationV35
        from nso.pixel_information_v34 import arrays_sha256
        with np.load(path, allow_pickle=False) as data:
            if self.task.receipt['scope'] == 'physical_experiment':
                if str(data['capture_scope'].item()) != 'paid_stopped_observation':
                    raise ValueError('diagnostic capture cannot be credited as a paid physical observation')
                if str(data['calibration_sha256'].item()) != self.task.receipt['calibration_sha256']:
                    raise ValueError('paid observation calibration differs from task predictions')
                receipt = json.loads(str(data['receipt_json'].item()))
                if not all(receipt.get(name, {}).get('stopped') is True
                           for name in ('stopped_before', 'stopped_after')):
                    raise ValueError('paid observation lacks actual stopped-feedback receipts')
            # The sensor adapter preserves raw measurements separately; these
            # arrays use the task's declared sampling and zero-missing contract.
            depth, rgb, ranges = (data[name].copy() for name in ('depth', 'rgb', 'ranges'))
            observation = ObservationV35(
                frame_id=str(data['frame_id'].item()), step=step, pose=tuple(pose),
                action=action, depth=depth, rgb=rgb, ranges=ranges)
        state = self.core.accept(observation)
        return dict(state=state, sensor_sha256=arrays_sha256(depth=depth, rgb=rgb, ranges=ranges),
                    source_file_sha256=file_digest(path), summary=self.core.summary())

    def next_action(self):
        action = self.core.next_action()
        return dict(action=action, selected=self.core.selected, pending=self.core.pending,
                    summary=self.core.summary())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    session = None
    with (output / 'decisions.jsonl').open('a', buffering=1) as log:
        for line in sys.stdin:
            request = None
            started = time.monotonic()
            try:
                request = json.loads(line)
                method, params = request['method'], request.get('params', {})
                if method == 'init':
                    if session is not None:
                        raise ValueError('reference worker already initialized')
                    session = ReferenceSession(**params)
                    result = session.identity()
                elif method == 'shutdown':
                    result = dict(shutdown=True)
                elif session is None:
                    raise ValueError('initialize reference task before observations or actions')
                elif method == 'observe':
                    result = session.observe(**params)
                elif method == 'next_action':
                    result = session.next_action()
                elif method == 'summary':
                    result = session.core.summary()
                else:
                    raise ValueError('unsupported reference RPC method')
                response = dict(id=request['id'], ok=True, result=result)
            except Exception as exc:
                response = dict(id=request.get('id') if isinstance(request, dict) else None,
                                ok=False, error=dict(type=type(exc).__name__, message=str(exc)))
            response['elapsed_s'] = time.monotonic() - started
            log.write(json.dumps(dict(request=request, response=response), allow_nan=False) + '\n')
            print(json.dumps(response, allow_nan=False), flush=True)
            if isinstance(request, dict) and request.get('method') == 'shutdown':
                break
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
