"""体素/语义补全结果可视化。"""

import os
from typing import Optional

import numpy as np


def save_voxel_completion_comparison(
    scene_idx: int,
    step_global: int,
    original_semantic_map: np.ndarray,
    completed_semantic_map: np.ndarray,
    original_occupancy: np.ndarray,
    completed_occupancy: np.ndarray,
    explored_map: np.ndarray,
    dump_dir: str,
    num_classes: Optional[int] = None,
) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Voxel Visualization] matplotlib 未安装，跳过保存")
        return None

    def _density(sem_map: np.ndarray) -> np.ndarray:
        sem = np.asarray(sem_map, dtype=np.float32)
        if sem.ndim == 3:
            return np.sum(sem, axis=0)
        return sem

    orig_d = _density(original_semantic_map)
    comp_d = _density(completed_semantic_map)
    vmax = max(float(orig_d.max()), float(comp_d.max()), 1e-6)

    images_dir = os.path.join(dump_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    out_path = os.path.join(
        images_dir,
        f"voxel_completion_{scene_idx}_{step_global:08d}.png",
    )

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    axes[0, 0].imshow(orig_d, cmap="inferno", vmin=0, vmax=vmax)
    axes[0, 0].set_title("Semantic (before)")
    axes[0, 1].imshow(comp_d, cmap="inferno", vmin=0, vmax=vmax)
    axes[0, 1].set_title("Semantic (after)")
    diff = np.clip(comp_d - orig_d, 0, None)
    axes[0, 2].imshow(diff, cmap="hot")
    axes[0, 2].set_title("Semantic added")

    axes[1, 0].imshow(original_occupancy, cmap="gray")
    axes[1, 0].set_title("Occupancy (before)")
    axes[1, 1].imshow(completed_occupancy, cmap="gray")
    axes[1, 1].set_title("Occupancy (after)")
    axes[1, 2].imshow(explored_map, cmap="Blues")
    axes[1, 2].set_title("Explored map")

    for ax in axes.flat:
        ax.axis("off")

    plt.suptitle(f"Voxel completion | scene {scene_idx} | step {step_global}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path
