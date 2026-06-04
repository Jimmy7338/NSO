import os
import sys

import matplotlib
import numpy as np

if os.environ.get("NSO_VIEWER_PROCESS"):
    matplotlib.use("TkAgg")
elif os.environ.get("NSO_VIS_NATIVE") == "1":
    matplotlib.use("TkAgg")
elif os.environ.get("NSO_USE_XVFB_GPU") or os.environ.get("NSO_X11_DISPLAY"):
    matplotlib.use("Agg")
else:
    matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import seaborn as sns
import skimage

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

_live_export_counter = 0


def _live_vis_every():
    return max(1, int(os.environ.get("NSO_VIS_EVERY", "1")))


def _should_export_live():
    global _live_export_counter
    every = _live_vis_every()
    _live_export_counter += 1
    return (_live_export_counter % every) == 0


def _live_frame_path(live_dir):
    if os.environ.get("NSO_VIS_JPEG", "1") == "1":
        return os.path.join(live_dir, "frame.jpg")
    return os.path.join(live_dir, "frame.png")


def _cv_resize(img, out_w, out_h):
    """放大用 CUBIC、缩小用 AREA，避免 INTER_AREA 放大导致发糊。"""
    h, w = img.shape[:2]
    if w == out_w and h == out_h:
        return img
    if out_w > w or out_h > h:
        interp = cv2.INTER_CUBIC
    else:
        interp = cv2.INTER_AREA
    return cv2.resize(img, (out_w, out_h), interpolation=interp)


def _map_px_from_pose(pos, map_h):
    """与 _draw_pose_cv2 一致的地图像素坐标。"""
    x, y, _ = pos
    px = int(x * 100.0 / 5.0)
    py = int(map_h - y * 100.0 / 5.0)
    return px, py


def _shift_pose_after_crop(pos, x0, y0, orig_h, crop_h):
    px, py = _map_px_from_pose(pos, orig_h)
    return (
        (px - x0) * 5.0 / 100.0,
        (crop_h - (py - y0)) * 5.0 / 100.0,
        pos[2],
    )


def crop_map_and_poses_from_mask(grid, pos, gt_pos, mask, pad=None, min_side=None):
    """按 visited/explored 等布尔掩码裁切地图，并平移位姿（比颜色阈值可靠）。"""
    if os.environ.get("NSO_VIS_MAP_ZOOM", "1") != "1":
        return grid, pos, gt_pos
    mask = np.asarray(mask).astype(bool)
    h, w = mask.shape[:2]
    activity = mask.copy()
    for pose in (pos, gt_pos):
        px, py = _map_px_from_pose(pose, h)
        r = 48
        activity[max(0, py - r):min(h, py + r),
                 max(0, px - r):min(w, px + r)] = True
    ys, xs = np.where(activity)
    if len(xs) < 1:
        return grid, pos, gt_pos
    if pad is None:
        pad = int(os.environ.get("NSO_VIS_MAP_ZOOM_PAD", "48"))
    if min_side is None:
        min_side = int(os.environ.get("NSO_VIS_MAP_ZOOM_MIN", "200"))
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad + 1)
    if (x1 - x0) < min_side:
        cx = (x0 + x1) // 2
        x0 = max(0, cx - min_side // 2)
        x1 = min(w, x0 + min_side)
        x0 = max(0, x1 - min_side)
    if (y1 - y0) < min_side:
        cy = (y0 + y1) // 2
        y0 = max(0, cy - min_side // 2)
        y1 = min(h, y0 + min_side)
        y0 = max(0, y1 - min_side)
    crop_h = y1 - y0
    cropped = grid[y0:y1, x0:x1]
    pos = _shift_pose_after_crop(pos, x0, y0, h, crop_h)
    gt_pos = _shift_pose_after_crop(gt_pos, x0, y0, h, crop_h)
    return cropped, pos, gt_pos


def _lock_map_axes(ax, grid):
    """防止箭头坐标把坐标轴撑大，导致地图缩在角落。"""
    mh, mw = grid.shape[0], grid.shape[1]
    ax.set_xlim(0, mw)
    ax.set_ylim(mh, 0)
    ax.set_aspect("equal", adjustable="box")


def _pose_to_imshow_xy(pose, grid):
    mh, mw = grid.shape[0], grid.shape[1]
    x, y, o = pose
    px = float(x) * 100.0 / 5.0
    py = mh - float(y) * 100.0 / 5.0
    px = float(np.clip(px, 0, max(mw - 1, 0)))
    py = float(np.clip(py, 0, max(mh - 1, 0)))
    return px, py, o


def _crop_map_for_live(grid_u8, pos, gt_pos):
    """全局地图显示前裁到已探索区域，并同步平移位姿坐标。"""
    if os.environ.get("NSO_VIS_MAP_ZOOM", "1") != "1":
        return grid_u8, pos, gt_pos
    h, w = grid_u8.shape[:2]
    if grid_u8.ndim == 2:
        activity = grid_u8 < 250
    else:
        activity = np.any(grid_u8 < 248, axis=2)
    for pose in (pos, gt_pos):
        px, py = _map_px_from_pose(pose, h)
        r = 48
        activity[max(0, py - r):min(h, py + r),
                 max(0, px - r):min(w, px + r)] = True
    ys, xs = np.where(activity)
    if len(xs) < 8:
        return grid_u8, pos, gt_pos
    pad = int(os.environ.get("NSO_VIS_MAP_ZOOM_PAD", "48"))
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad + 1)
    min_side = int(os.environ.get("NSO_VIS_MAP_ZOOM_MIN", "200"))
    if (x1 - x0) < min_side:
        cx = (x0 + x1) // 2
        x0 = max(0, cx - min_side // 2)
        x1 = min(w, x0 + min_side)
        x0 = max(0, x1 - min_side)
    if (y1 - y0) < min_side:
        cy = (y0 + y1) // 2
        y0 = max(0, cy - min_side // 2)
        y1 = min(h, y0 + min_side)
        y0 = max(0, y1 - min_side)
    crop_h = y1 - y0
    cropped = grid_u8[y0:y1, x0:x1]
    pos = _shift_pose_after_crop(pos, x0, y0, h, crop_h)
    gt_pos = _shift_pose_after_crop(gt_pos, x0, y0, h, crop_h)
    return cropped, pos, gt_pos


def _live_canvas_size():
    obs_w = int(os.environ.get("NSO_LIVE_OBS_W", "640"))
    obs_h = int(os.environ.get("NSO_LIVE_OBS_H", "480"))
    map_sz = int(os.environ.get("NSO_LIVE_MAP_SIZE", "480"))
    return obs_w, obs_h, map_sz


def _draw_pose_cv2(canvas, pos, color_bgr, agent_size=8):
    h, w = canvas.shape[:2]
    x, y, o = pos
    x = int(x * 100.0 / 5.0)
    y = int(h - y * 100.0 / 5.0)
    dx = int(np.cos(np.deg2rad(o)) * agent_size)
    dy = int(-np.sin(np.deg2rad(o)) * agent_size * 1.25)
    cv2.arrowedLine(
        canvas, (x - dx, y - dy), (x + dx, y + dy),
        color_bgr, 2, tipLength=0.35)


def _save_live_frame_cv2(obs, grid, live_dir, pos, gt_pos, timestep=None):
    """OpenCV 拼接帧，比 matplotlib savefig 快一个数量级。"""
    import tempfile
    obs_w, obs_h, map_sz = _live_canvas_size()
    map_w = map_h = map_sz
    obs_u8 = np.clip(obs, 0, 255).astype(np.uint8)
    if obs_u8.ndim == 2:
        obs_u8 = cv2.cvtColor(obs_u8, cv2.COLOR_GRAY2RGB)
    grid_u8 = np.clip(grid, 0, 255).astype(np.uint8)
    if grid_u8.ndim == 2:
        grid_u8 = cv2.cvtColor(grid_u8, cv2.COLOR_GRAY2RGB)
    grid_u8, pos, gt_pos = _crop_map_for_live(grid_u8, pos, gt_pos)
    obs_r = _cv_resize(obs_u8, obs_w, obs_h)
    map_r = _cv_resize(grid_u8, map_w, map_h)
    map_bgr = cv2.cvtColor(map_r, cv2.COLOR_RGB2BGR)
    _draw_pose_cv2(map_bgr, gt_pos, (160, 160, 160))
    _draw_pose_cv2(map_bgr, pos, (0, 0, 255))
    obs_bgr = cv2.cvtColor(obs_r, cv2.COLOR_RGB2BGR)
    canvas = np.hstack([obs_bgr, map_bgr])
    if timestep is not None:
        cv2.putText(
            canvas, "step {}".format(int(timestep)),
            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    frame_path = _live_frame_path(live_dir)
    os.makedirs(live_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=live_dir, suffix=".tmp")
    os.close(fd)
    try:
        quality = int(os.environ.get("NSO_VIS_JPEG_QUALITY", "95"))
        ok = False
        if frame_path.endswith(".jpg"):
            try:
                from PIL import Image as PILImage
                rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                PILImage.fromarray(rgb).save(
                    tmp_path, format="JPEG", quality=quality, optimize=True)
                ok = True
            except Exception:
                ok = cv2.imwrite(
                    tmp_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            ok = cv2.imwrite(tmp_path, canvas)
        if ok:
            os.replace(tmp_path, frame_path)
        elif os.path.exists(tmp_path):
            os.unlink(tmp_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def live_vis_fast_enabled():
    if os.environ.get("NSO_VIS_NATIVE") == "1":
        return False
    return (os.environ.get("NSO_LIVE_FAST", "1") == "1"
            and os.environ.get("NSO_LIVE_VIS_DIR")
            and HAS_CV2)


def visualize(fig, ax, img, grid, pos, gt_pos, dump_dir, rank, ep_no, t,
              visualize, print_images, vis_style,
              detected_classes=None, class_counts=None, class_avg_scores=None):
    live_dir = os.environ.get("NSO_LIVE_VIS_DIR")
    if (visualize and live_dir and live_vis_fast_enabled()
            and not print_images and _should_export_live()):
        _save_live_frame_cv2(img, grid, live_dir, pos, gt_pos, timestep=t)
        return

    for i in range(2):
        ax[i].clear()
        ax[i].set_yticks([])
        ax[i].set_xticks([])
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])

    ax[0].imshow(img)
    ax[0].set_title("Observation", family='sans-serif',
                    fontname='DejaVu Sans',
                    fontsize=20)

    if vis_style == 1:
        title = "Predicted Map and Pose"
    else:
        title = "Ground-Truth Map and Pose"

    grid = np.clip(grid, 0, 255).astype(np.uint8)
    if not os.environ.get("NSO_VIS_MASK_CROP_DONE"):
        grid, pos, gt_pos = _crop_map_for_live(grid, pos, gt_pos)

    ax[1].imshow(grid, origin="upper")
    ax[1].set_title(title, family='sans-serif',
                    fontname='DejaVu Sans',
                    fontsize=20)

    agent_size = 8

    def _draw_arrow(pose, fc, alpha):
        x, y, o = _pose_to_imshow_xy(pose, grid)
        dx = np.cos(np.deg2rad(o))
        dy = -np.sin(np.deg2rad(o))
        ax[1].arrow(
            x - dx, y - dy, dx * agent_size, dy * agent_size * 1.25,
            head_width=agent_size, head_length=agent_size * 1.25,
            length_includes_head=True, fc=fc, ec=fc, alpha=alpha,
            clip_on=True)

    _draw_arrow(gt_pos, "Grey", 0.9)
    _draw_arrow(pos, "Red", 0.6)

    for _ in range(5):
        plt.tight_layout()
    _lock_map_axes(ax[1], grid)

    # 在第二个子图（地图）的右上角添加语义标签列表显示
    # 注意：必须在tight_layout()之后添加，否则会被移除
    # 获取图像尺寸（注意：imshow的坐标系统，y轴从上到下）
    img_height, img_width = grid.shape[:2]
    
    # 显示语义标签信息
    # 添加调试：打印参数状态
    if detected_classes is not None:
        # 准备显示的文本
        if len(detected_classes) > 0:
            text_lines = ["Detected Objects:", ""]
            # 按出现次数排序
            if class_counts and len(class_counts) > 0:
                sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
                for cls_name, count in sorted_classes[:10]:  # 最多显示10个
                    avg_score = class_avg_scores.get(cls_name, 0.0) if class_avg_scores else 0.0
                    text_lines.append(f"{cls_name}: {count} ({avg_score:.2f})")
            elif len(detected_classes) > 0:
                for cls_name in detected_classes[:10]:
                    text_lines.append(f"{cls_name}")
            
            if len(text_lines) > 2:  # 至少有实际内容（除了标题和空行）
                # 构建完整文本
                full_text = "\n".join(text_lines)
                
                # 计算文本位置（使用图像坐标，右上角）
                # 使用相对位置，确保在不同图像尺寸下都能显示
                text_x = img_width * 0.98  # 距离右边缘2%
                text_y = img_height * 0.02  # 距离上边缘2%
                
                # 在ax[1]上添加文本，使用数据坐标系统
                # 使用zorder确保文本在最上层
                try:
                    text_obj = ax[1].text(text_x, text_y, full_text,
                          fontsize=8, family='monospace',
                          verticalalignment='top',
                          horizontalalignment='right',
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95, 
                                   edgecolor='black', linewidth=1.5),
                          color='black',
                          zorder=1000,  # 确保在最上层
                          transform=ax[1].transData)  # 使用数据坐标系统
                    # 强制刷新，确保文本显示
                    ax[1].figure.canvas.draw_idle()
                except Exception as e:
                    # 如果出错，尝试使用axes坐标系统
                    try:
                        ax[1].text(0.98, 0.98, full_text,
                              fontsize=8, family='monospace',
                              verticalalignment='top',
                              horizontalalignment='right',
                              bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95, 
                                       edgecolor='black', linewidth=1.5),
                              color='black',
                              zorder=1000,
                              transform=ax[1].transAxes)  # 使用axes坐标系统
                    except:
                        pass  # 如果还是失败，静默忽略

    if visualize:
        fig.canvas.draw_idle()
        if live_dir and not live_vis_fast_enabled():
            frame_path = _live_frame_path(live_dir)
            import tempfile
            os.makedirs(live_dir, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=live_dir, suffix=".png")
            os.close(fd)
            dpi = int(os.environ.get("NSO_VIS_DPI", "72"))
            try:
                fig.savefig(
                    tmp_path,
                    dpi=dpi,
                    bbox_inches="tight",
                    facecolor=fig.get_facecolor(),
                )
                os.replace(tmp_path, frame_path)
            except Exception:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
        elif not live_dir:
            fig.canvas.flush_events()
            plt.pause(0.05)

    if print_images:
        fn = '{}/episodes/{}/{}/{}-{}-Vis-{}.png'.format(
            dump_dir, (rank + 1), ep_no, rank, ep_no, t)
        plt.savefig(fn)


def insert_circle(mat, x, y, value):
    mat[x - 2: x + 3, y - 2:y + 3] = value
    mat[x - 3:x + 4, y - 1:y + 2] = value
    mat[x - 1:x + 2, y - 3:y + 4] = value
    return mat


def fill_color(colored, mat, color):
    for i in range(3):
        colored[:, :, 2 - i] *= (1 - mat)
        colored[:, :, 2 - i] += (1 - color[i]) * mat
    return colored


def get_colored_map(mat, collision_map, visited, visited_gt, goal,
                    explored, gt_map, gt_map_explored,
                    semantic_density=None, semantic_freshness=None,
                    structural_map=None):
    """
    生成彩色地图，可选叠加语义密度图
    Args:
        semantic_density: (H, W) numpy array，语义密度图，值越大表示语义密度越高
        semantic_freshness: (H, W) numpy array，表示“已观测但未到达”的语义价值
    """
    m, n = mat.shape
    colored = np.zeros((m, n, 3))
    pal = sns.color_palette("Paired")

    current_palette = [(0.9, 0.9, 0.9)]
    colored = fill_color(colored, gt_map, current_palette[0])

    current_palette = [(235. / 255., 243. / 255., 1.)]
    colored = fill_color(colored, explored, current_palette[0])

    green_palette = sns.light_palette("green")
    colored = fill_color(colored, mat, pal[2])

    current_palette = [(0.6, 0.6, 0.6)]
    colored = fill_color(colored, gt_map_explored, current_palette[0])

    colored = fill_color(colored, mat * gt_map_explored, pal[3])

    red_palette = sns.light_palette("red")

    colored = fill_color(colored, visited_gt, current_palette[0])
    colored = fill_color(colored, visited, pal[4])
    colored = fill_color(colored, visited * visited_gt, pal[5])

    colored = fill_color(colored, collision_map, pal[2])

    current_palette = sns.color_palette()

    selem = skimage.morphology.disk(4)
    goal_mat = np.zeros((m, n))
    goal_mat[goal[0], goal[1]] = 1
    goal_mat = 1 - skimage.morphology.binary_dilation(
        goal_mat, selem) != True

    colored = fill_color(colored, goal_mat, current_palette[0])

    current_palette = sns.color_palette("Paired")

    colored = 1 - colored
    colored *= 255
    colored = colored.astype(np.uint8)
    
    # 叠加语义密度图（如果提供）- 改进的可视化效果
    if semantic_density is not None and semantic_density.shape == (m, n) and HAS_CV2:
        # 归一化语义密度到 [0, 1]，使用更平滑的归一化
        sem_norm = semantic_density.astype(np.float32)
        sem_max = np.max(sem_norm)
        if sem_max > 0:
            # 使用更平滑的归一化，避免过度饱和
            sem_norm = np.clip(sem_norm / (sem_max * 0.8 + 1e-6), 0, 1)
        
        # 使用更精细的热力图颜色映射
        sem_uint8 = (sem_norm * 255).astype(np.uint8)
        # 使用JET colormap，提供更平滑的渐变效果
        sem_colored = cv2.applyColorMap(sem_uint8, cv2.COLORMAP_JET)
        sem_colored = cv2.cvtColor(sem_colored, cv2.COLOR_BGR2RGB)
        
        # 使用自适应阈值和更精细的叠加
        # 只显示有意义的语义密度区域（阈值可调，现在更低以显示更多细节）
        sem_mask = sem_norm > 0.05  # 降低阈值以显示更多细节
        
        # 使用自适应透明度：密度越高，叠加越明显
        alpha_base = 0.3  # 基础透明度
        alpha_map = alpha_base + sem_norm * 0.3  # 密度越高，透明度越高（0.3-0.6）
        alpha_map = np.clip(alpha_map, 0, 0.6)  # 限制最大透明度
        
        # 精细叠加：逐像素混合
        for c in range(3):
            colored[:, :, c] = np.where(
                sem_mask,
                colored[:, :, c] * (1 - alpha_map) + sem_colored[:, :, c] * alpha_map,
                colored[:, :, c]
            )

    # 叠加语义新鲜度（如果提供）：突出已观测但未到达的高价值语义区域
    if semantic_freshness is not None and semantic_freshness.shape == (m, n) and HAS_CV2:
        fresh_norm = np.clip(semantic_freshness.astype(np.float32), 0.0, 1.0)
        fresh_uint8 = (fresh_norm * 255).astype(np.uint8)
        fresh_colored = cv2.applyColorMap(fresh_uint8, cv2.COLORMAP_WINTER)
        fresh_colored = cv2.cvtColor(fresh_colored, cv2.COLOR_BGR2RGB)

        fresh_mask = fresh_norm > 0.05
        alpha_fresh = np.clip(0.25 + fresh_norm * 0.35, 0.25, 0.6)
        for c in range(3):
            colored[:, :, c] = np.where(
                fresh_mask,
                colored[:, :, c] * (1 - alpha_fresh) + fresh_colored[:, :, c] * alpha_fresh,
                colored[:, :, c]
            )
    
    # 叠加结构内容图（门框/狭窄/开阔）的综合值
    if structural_map is not None and structural_map.shape == (m, n) and HAS_CV2:
        struct_norm = structural_map.astype(np.float32)
        smax = np.max(struct_norm)
        if smax > 0:
            struct_norm = np.clip(struct_norm / (smax * 0.8 + 1e-6), 0, 1)
        struct_uint8 = (struct_norm * 255).astype(np.uint8)
        struct_colored = cv2.applyColorMap(struct_uint8, cv2.COLORMAP_AUTUMN)
        struct_colored = cv2.cvtColor(struct_colored, cv2.COLOR_BGR2RGB)
        struct_mask = struct_norm > 0.05
        alpha_struct = np.clip(0.25 + struct_norm * 0.3, 0.25, 0.55)
        for c in range(3):
            colored[:, :, c] = np.where(
                struct_mask,
                colored[:, :, c] * (1 - alpha_struct) + struct_colored[:, :, c] * alpha_struct,
                colored[:, :, c]
            )

    return colored
