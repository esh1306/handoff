"""Video and image helpers."""
import cv2
import numpy as np


def to_jpeg(rgb, max_side=768, quality=85):
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1:
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def extract_frames(path, n=8):
    """Return (info, [(label, rgb)]) with n frames spread evenly over the video."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Can't open video {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    for idx in np.linspace(0, max(total - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if ok:
            frames.append((f"video frame at t={idx / fps:.1f}s", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames could be read from {path}")
    h, w = frames[0][1].shape[:2]
    return {"duration_s": round(total / fps, 1), "fps": round(fps, 1), "width": w, "height": h}, frames


def pick(items, n):
    """At most n items, spread evenly, always keeping the first and last."""
    if len(items) <= n:
        return list(items)
    return [items[i] for i in np.linspace(0, len(items) - 1, n).astype(int)]
