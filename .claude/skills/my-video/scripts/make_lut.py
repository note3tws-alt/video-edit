"""Bake the measured grade (per-channel curves + RGB saturation around luma, fitted on raw->graded pixel pairs)
into a 33^3 .cube LUT so ffmpeg reproduces the model exactly (curves+eq differs by ~2.4/255 MAE).

Usage: python make_lut.py   -> assets/luts/my-video.cube
"""
import os
import re

import numpy as np

from common import ASSETS, load_style


def model(rgb, filt):
    """rgb: (..., 3) floats 0..1 -> graded floats 0..1 (same math as the Step-1 fit)."""
    cur = {c: np.array([[float(a) for a in p.split("/")] for p in re.search(c + r"='([^']+)'", filt).group(1).split()]) for c in "rgb"}
    sat = float(re.search(r"saturation=([\d.]+)", filt).group(1)) if "saturation" in filt else 1.0
    o = np.stack([np.interp(rgb[..., i], cur[c][:, 0], cur[c][:, 1]) for i, c in enumerate("rgb")], -1)
    y = (0.299 * o[..., 0] + 0.587 * o[..., 1] + 0.114 * o[..., 2])[..., None]
    return np.clip(y + (o - y) * sat, 0, 1)


def write_cube(path, filt, n=33):
    g = np.linspace(0, 1, n)
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")  # .cube order: R fastest, then G, then B
    rgb = np.stack([r, gg, b], -1).reshape(-1, 3)
    out = model(rgb, filt)
    with open(path, "w") as f:
        f.write('TITLE "my-video grade (fitted on reference raw->graded pairs)"\n')
        f.write(f"LUT_3D_SIZE {n}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n")
        for v in out:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    return path


if __name__ == "__main__":
    st = load_style()
    d = os.path.join(ASSETS, "luts")
    os.makedirs(d, exist_ok=True)
    print(write_cube(os.path.join(d, "my-video.cube"), st["color"]["talking_head_curves"]))
    print(write_cube(os.path.join(d, "my-video-broll-warm.cube"), st["color"]["broll_warm_curves"]))
