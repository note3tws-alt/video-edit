"""Face detection (OpenCV YuNet, bundled ONNX) used for reframing and the B-roll halo."""
import os
import subprocess

import cv2
import numpy as np

from common import ASSETS

_MODEL = os.path.join(ASSETS, "models", "face_detection_yunet_2023mar.onnx")
_det = {}


def detect(frame_bgr, score=0.6):
    """Return faces as array of [x, y, w, h, score] sorted by area (largest first)."""
    h, w = frame_bgr.shape[:2]
    scale = 640.0 / max(h, w) if max(h, w) > 640 else 1.0
    small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale))) if scale != 1.0 else frame_bgr
    key = (small.shape[1], small.shape[0])
    if key not in _det:
        _det[key] = cv2.FaceDetectorYN.create(_MODEL, "", key, score)
    _, faces = _det[key].detect(small)
    if faces is None:
        return np.zeros((0, 5))
    out = np.array([[f[0] / scale, f[1] / scale, f[2] / scale, f[3] / scale, f[14]] for f in faces])
    return out[np.argsort(-(out[:, 2] * out[:, 3]))]


def grab_frames(path, times, width=None, height=None):
    """Decode single frames at given times (seconds) with ffmpeg; returns list of BGR arrays (or None)."""
    out = []
    for t in times:
        p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", path, "-frames:v", "1",
                            "-f", "image2pipe", "-vcodec", "png", "-"], stdout=subprocess.PIPE)
        if not p.stdout:
            out.append(None)
            continue
        img = cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR)
        out.append(img)
    return out


def face_center(path, duration, samples=9):
    """Median face centre (fractions of width/height) over sampled frames; None if no face found."""
    ts = [duration * (i + 0.5) / samples for i in range(samples)]
    pts = []
    for img in grab_frames(path, ts):
        if img is None:
            continue
        f = detect(img)
        if len(f):
            x, y, w, h, _ = f[0]
            pts.append(((x + w / 2) / img.shape[1], (y + h / 2) / img.shape[0], w / img.shape[1]))
    if not pts:
        return None
    a = np.array(pts)
    return {"cx": float(np.median(a[:, 0])), "cy": float(np.median(a[:, 1])), "w": float(np.median(a[:, 2]))}
