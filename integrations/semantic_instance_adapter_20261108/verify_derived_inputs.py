"""Offline verification from compact arrays; needs numpy/scipy, not ROS or raw images."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.ndimage import binary_closing, label
from association import MeasuredAssociation


def main():
    root = Path(__file__).resolve().parent
    meta = json.loads((root/'derived_inputs.json').read_text())
    assert hashlib.sha256((root/'derived_inputs.npz').read_bytes()).hexdigest() == meta['data_sha256']
    assert hashlib.sha256((root/'association.py').read_bytes()).hexdigest() == meta['frozen_association_sha256']
    review = json.loads((root/'review.json').read_text())
    assert hashlib.sha256((root/'review.json').read_bytes()).hexdigest() == meta['original_review_sha256']
    association = MeasuredAssociation()
    verified = []
    with np.load(root/'derived_inputs.npz', allow_pickle=False) as data:
        for record in meta['frames']:
            action = record['action']; prefix = f'a{action:02d}_'
            uv, depth, K, T = (data[prefix+k] for k in ('uv', 'depth_m', 'K', 'planning_from_camera'))
            local = np.column_stack([(uv[:, 0]-K[0, 2])*depth/K[0, 0],
                                     (uv[:, 1]-K[1, 2])*depth/K[1, 1], depth])
            projected = np.floor((local @ T[:3, :3].T+T[:3, 3])/.15).astype(np.int64)
            keys, owners, categories = (data[prefix+k] for k in ('keys', 'owner', 'label'))
            assert np.array_equal(projected, keys)
            association.observe(record['frame_id'], keys, owners, categories)
            if action not in meta['checkpoints']:
                continue
            qkeys, points = data[prefix+'quality_keys'], data[prefix+'quality_point']
            memberships = association.membership(qkeys)
            derived = association.labels_for(qkeys, memberships)
            expected = next(c for c in review['checkpoints'] if c['action'] == action)
            counts = {str(t): int(np.count_nonzero(memberships == t)) for t in np.unique(memberships)}
            assert counts == expected['membership_counts'] and sum(counts.values()) == len(points)
            qualified = []
            shape = tuple(meta['config']['shape']); stride = int(np.prod(shape))+1
            for track in np.unique(memberships):
                indices = np.flatnonzero(memberships == track)
                cells = np.column_stack([shape[0]-1-np.floor(points[indices, 1]/.2),
                                          np.floor(points[indices, 0]/.2)]).astype(int)
                inside = ((cells >= 0) & (cells < np.array(shape))).all(axis=1)
                occupied = np.zeros(shape, bool); occupied[tuple(cells[inside].T)] = True
                groups, _ = label(binary_closing(occupied, structure=np.ones((3, 3))))
                ids = np.zeros(len(indices), int); ids[inside] = groups[tuple(cells[inside].T)]
                for group in sorted(set(ids)-{0}):
                    selected = indices[ids == group]
                    low, high = np.quantile(points[selected], [.05, .95], axis=0)
                    span = high-low
                    if len(selected) < 8 or max(span[:2]) > 3 or span[2] < .25:
                        continue
                    qualified.append((int(track)*stride+int(group), len(selected),
                                      int(np.count_nonzero(np.isin(derived[selected], [2, 3])))))
            expected_patches = [(p['group'], p['support_points'], p['marked_points'])
                                for p in expected['associated_patches']]
            assert qualified == expected_patches
            verified.append(action)
    result = dict(status='passed_compact_derived_input_checks', frames=len(meta['frames']),
                  projection_voxel_keys_exact=True, checkpoint_memberships_and_native_eligibility_exact=verified,
                  full_RGBD_or_detector_or_online_reconstruction_replay=False,
                  boundary=meta['boundary'])
    (root/'derived_verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
