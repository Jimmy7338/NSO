#!/usr/bin/env python3
"""Bind renderer imports listed by the new scene seal before any actor runs."""

from pathlib import Path
import shutil
import sys

from runner import PREVIOUS, ROOT


sys.path.append(str(PREVIOUS))
from regression import read, sha, write
from study import verify


def main():
    path = ROOT / "FORMAL_FREEZE.json"
    frozen = read(path)
    scenes = Path(frozen["scene_root"])
    actor = read(scenes / "actor_freeze_holdout.json")
    evaluator = read(scenes / "evaluator_freeze_holdout.json")
    bindings = {"actor_files": actor["files_sha256"] | actor["simulator_only_collision_cache_sha256"],
                "evaluator_files": evaluator["files_sha256"]}
    additions = {}
    for key, ledger in bindings.items():
        verify(ledger)
        assert all(frozen[key].get(source, digest) == digest for source, digest in ledger.items())
        additions[key] = {source: digest for source, digest in ledger.items() if source not in frozen[key]}
    if not any(additions.values()):
        return
    assert not (ROOT / "execution/formal").exists()
    previous = ROOT / "FORMAL_FREEZE.preflight_001.json"
    assert not previous.exists()
    shutil.copyfile(path, previous)
    updated = frozen | {key: frozen[key] | additions[key] for key in bindings}
    assert not any("evaluator_references" in source or "evaluator_truth" in source
                   for source in updated["actor_files"])
    updated["actor_files"][str(Path(__file__).resolve())] = sha(__file__)
    temp = path.with_suffix(".tmp")
    write(temp, updated)
    temp.replace(path)
    write(ROOT / "formal_preactor_hash_repair.json", {
        "before_sha256": sha(previous), "after_sha256": sha(path), "added_source_bindings": additions,
        "prior_formal_actors": 0, "scope": "Imported source hashes from existing scene seals.",
    })


if __name__ == "__main__":
    main()
