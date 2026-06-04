"""探索地图补全前后对比可视化。"""

from typing import Optional

import numpy as np


def save_exploration_completion_comparison(
    original_explored_map: np.ndarray,
    completed_explored_map: np.ndarray,
    confidence_map: np.ndarray,
    obstacle_map: np.ndarray,
    output_path: str,
    semantic_map: Optional[np.ndarray] = None,
    scene_idx: int = 0,
    step: int = 0,
) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Exploration Completion Viz] matplotlib 未安装，跳过保存")
        return None

    orig = np.asarray(original_explored_map, dtype=np.float32)
    comp = np.asarray(completed_explored_map, dtype=np.float32)
    conf = np.asarray(confidence_map, dtype=np.float32)
    obs = np.asarray(obstacle_map, dtype=np.float32)

    ncols = 3 if semantic_map is None else 4
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    axes[0].imshow(orig, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_title("Explored (before)")
    axes[1].imshow(comp, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Explored (after)")
    added = np.clip(comp - orig, 0, None)
    axes[2].imshow(added, cmap="hot")
    axes[2].set_title("Added exploration")

    if semantic_map is not None:
        sem = np.asarray(semantic_map, dtype=np.float32)
        if sem.ndim == 3:
            sem = np.sum(sem, axis=0)
        axes[3].imshow(sem, cmap="inferno")
        axes[3].set_title("Semantic density")

    for ax in axes:
        ax.axis("off")

    # 障碍物轮廓叠加在最后一张子图
    try:
        import cv2
        free = (obs < 0.5).astype(np.uint8)
        edge = free - cv2.erode(free, np.ones((3, 3), np.uint8))
        ys, xs = np.where(edge > 0)
        if len(xs) > 0:
            axes[-1].plot(xs, ys, "c.", markersize=0.3, alpha=0.4)
    except Exception:
        pass

    plt.suptitle(
        f"Exploration completion | scene {scene_idx} | step {step}\n"
        f"mean confidence={float(conf.mean()):.3f}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path
