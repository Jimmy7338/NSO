#!/usr/bin/env python3
"""Use the sealed supervisor with the formal frontier telemetry entry point."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import study


def seal():
    study.seal('formal')
    path=study.freeze('formal')
    receipt=study.read(path)
    files=[Path(__file__),study.ROOT/'formal_runner.py']
    scene_root=study.ROOT/'scenes'
    actor=study.read(scene_root/'actor_freeze_holdout.json')
    evaluator=study.read(scene_root/'evaluator_freeze_holdout.json')
    assert actor['actors_run']==evaluator['actors_run']==0 and actor['budget']==evaluator['budget']==receipt['budget']
    receipt['actor_files'] |= actor['files_sha256'] | study.hashes(files)
    receipt['evaluator_files'] |= evaluator['files_sha256']
    study.verify(receipt['actor_files'])
    study.verify(receipt['evaluator_files'])
    assert not any('evaluator_references' in path or 'evaluator_truth' in path for path in receipt['actor_files'])
    receipt['launch_entry_point']=str(study.ROOT/'formal_runner.py')
    next_version=path.with_suffix('.expanded.tmp')
    study.write(next_version,receipt)
    next_version.replace(path)


def run(workers,resume):
    receipt=study.read(study.freeze('formal'))
    inherited=study.subprocess
    assert receipt['launch_entry_point']==str(study.ROOT/'formal_runner.py')

    def launch(command,*args,**kwargs):
        assert command[1]==str(study.ROOT/'runner.py')
        command[1]=receipt['launch_entry_point']
        return inherited.Popen(command,*args,**kwargs)

    study.subprocess=SimpleNamespace(Popen=launch,STDOUT=inherited.STDOUT)
    study.dispatch('formal',workers,resume)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('command',choices=('seal','run'))
    cli.add_argument('--workers',type=int,default=5)
    cli.add_argument('--resume',action='store_true')
    args=cli.parse_args()
    if args.command=='seal':
        seal()
    else:
        run(args.workers,args.resume)
