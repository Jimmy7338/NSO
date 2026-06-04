"""确保导入的是 third_party habitat-lab 0.2.x，而非 env/habitat/habitat_api。"""
from __future__ import annotations

import os
import sys

_SETUP_DONE = False


def setup_habitat2_lab() -> None:
    global _SETUP_DONE
    if _SETUP_DONE:
        return

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    h2_lab = os.path.join(root, "third_party", "habitat-lab", "habitat-lab")
    vendored_api = os.path.join(root, "env", "habitat", "habitat_api")

    if not os.path.isdir(h2_lab):
        raise ImportError(
            f"未找到 habitat-lab 源码: {h2_lab}\n"
            "请运行: bash scripts/unpack_local_archives.sh && bash scripts/install_habitat2.sh"
        )

    h2_lab = os.path.abspath(h2_lab)
    vendored_api = os.path.abspath(vendored_api)

    # 去掉会把旧版 habitat 顶到前面的路径
    cleaned = []
    for p in sys.path:
        if not p:
            continue
        ap = os.path.abspath(p)
        if ap == vendored_api or ap.startswith(vendored_api + os.sep):
            continue
        cleaned.append(p)
    sys.path[:] = cleaned

    if h2_lab not in sys.path:
        sys.path.insert(0, h2_lab)

    # 若已误加载旧 habitat，清除缓存
    for name in list(sys.modules):
        if name == "habitat" or name.startswith("habitat."):
            del sys.modules[name]

    _SETUP_DONE = True
