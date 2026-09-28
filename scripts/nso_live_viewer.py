#!/usr/bin/env python3
"""在 MobaXterm X11 上显示 NSO 实时画面（与 xvfb+GPU 主进程分离）。"""
import os
import sys
import time
from io import BytesIO

FRAME_DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/nso_vis_live"
LOG_PATH = os.path.join(FRAME_DIR, "viewer.log")
POLL_MS = int(os.environ.get("NSO_VIEWER_POLL_MS", "50"))
MIN_FRAME_BYTES = 2048


def log(msg):
    line = "[{}] {}\n".format(time.strftime("%H:%M:%S"), msg)
    sys.stderr.write(line)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass


def _frame_candidates():
    yield os.path.join(FRAME_DIR, "frame.jpg")
    yield os.path.join(FRAME_DIR, "frame.png")


def _pick_frame_path():
    best = None
    best_mtime = 0.0
    for path in _frame_candidates():
        if os.path.isfile(path):
            mtime = os.path.getmtime(path)
            if mtime > best_mtime:
                best_mtime = mtime
                best = path
    return best, best_mtime


def load_frame_bytes():
    path, _ = _pick_frame_path()
    if not path or os.path.getsize(path) < MIN_FRAME_BYTES:
        return None, None
    with open(path, "rb") as f:
        return f.read(), path


def main():
    display = os.environ.get("DISPLAY", "")
    if not display:
        log("错误: DISPLAY 未设置")
        sys.exit(1)

    try:
        import tkinter as tk
        from PIL import Image, ImageTk
    except ImportError as e:
        log("错误: 缺少 tkinter 或 Pillow: {}".format(e))
        sys.exit(1)

    root = tk.Tk()
    root.title("NSO Live")
    geom = os.environ.get("NSO_VIEWER_GEOM", "1280x720")
    root.geometry(geom)
    root.configure(bg="#2b2b2b")
    try:
        root.attributes("-topmost", True)
        root.lift()
        root.focus_force()
    except tk.TclError:
        pass

    label = tk.Label(
        root,
        bg="#2b2b2b",
        text="Waiting for NSO frames...",
        fg="#ffffff",
        font=("DejaVu Sans", 14),
    )
    label.pack(fill=tk.BOTH, expand=True)

    state = {"photo": None, "mtime": 0.0, "shown": False, "path": ""}

    def on_first_frame():
        try:
            root.attributes("-topmost", False)
        except tk.TclError:
            pass

    def refresh():
        try:
            raw, path = load_frame_bytes()
            if raw and path:
                mtime = os.path.getmtime(path)
                if mtime > state["mtime"]:
                    img = Image.open(BytesIO(raw)).convert("RGB")
                    img.load()
                    w, h = img.size
                    hq = os.environ.get("NSO_VIS_HIGH_QUALITY", "1") == "1"
                    max_w = int(os.environ.get("NSO_VIEWER_MAX_W", "1280" if hq else "880"))
                    max_h = int(os.environ.get("NSO_VIEWER_MAX_H", "720" if hq else "380"))
                    scale = min(max_w / w, max_h / h, 1.0)
                    if scale < 1.0:
                        resample = getattr(Image, "Resampling", Image).LANCZOS
                        img = img.resize(
                            (int(w * scale), int(h * scale)),
                            resample,
                        )
                    state["photo"] = ImageTk.PhotoImage(img)
                    label.configure(image=state["photo"], text="")
                    state["mtime"] = mtime
                    state["path"] = path
                    if not state["shown"]:
                        state["shown"] = True
                        log("首帧已显示 {}x{} ({})".format(w, h, os.path.basename(path)))
                        root.after(300, on_first_frame)
                        try:
                            root.lift()
                            root.deiconify()
                        except tk.TclError:
                            pass
        except Exception as e:
            log("刷新失败: {}".format(e))
        root.after(POLL_MS, refresh)

    log("查看器已启动 DISPLAY={}".format(display))
    root.update_idletasks()
    root.after(0, refresh)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        root.destroy()


if __name__ == "__main__":
    main()
