#!/usr/bin/env python3
"""Verify existing saved observations; creates no World or physical trajectory."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nso.saved_replay_v44 import (load_saved_episode_v44, verify_saved_observations_v44,
                                 FailedSavedEpisodeV44, SavedEpisodeIntegrityErrorV44)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('--expected-manifest-sha256')
    parser.add_argument('--integrity-only', action='store_true', help='validate saved files without creating a mapper')
    args = parser.parse_args()
    try:
        if args.integrity_only:
            episode = load_saved_episode_v44(args.episode, expected_manifest_sha256=args.expected_manifest_sha256)
            result = dict(status='integrity_verified', source_episode_status=episode.status,
                frames=len(episode.frames), task_success=episode.task_success, eligible_study_episode=episode.eligible_study_episode, policy_determinism_verified=False,
                externally_pinned_manifest=episode.externally_pinned_manifest, new_worlds=0, physical_actions=0)
        else:
            result = verify_saved_observations_v44(args.episode, expected_manifest_sha256=args.expected_manifest_sha256)
    except (SavedEpisodeIntegrityErrorV44, OSError, ValueError, KeyError) as exc:
        result = dict(status='failed_source_episode' if isinstance(exc, FailedSavedEpisodeV44) else 'rejected',
            error=dict(type=type(exc).__name__, message=str(exc)), new_worlds=0, physical_actions=0)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0 if result['status'] in ('verified', 'integrity_verified') else 1


if __name__ == '__main__':
    sys.exit(main())
