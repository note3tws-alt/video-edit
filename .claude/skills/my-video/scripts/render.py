"""Steps d-g (render): graded footage per plan segments, B-roll, punch-in/animated zooms, zoom-blur transitions,
intro blur, captions, cards, halo; encodes with the mixed audio to the measured output format.

Usage: python render.py --work WORK --out OUT.mp4 [--max-seconds N]
"""
import argparse
import math
import os
import subprocess

import cv2
import numpy as np

from common import ASSETS, ease, has_upper, hex_rgb, lerp, load_json, load_style, log, save_json
from cards import CardRenderer
from faces import detect
from textfx import CaptionRenderer


class Reader:
    """Pull exactly n BGR frames from a media range through an ffmpeg filter chain."""

    def __init__(self, path, start, n, vf, W, H, fps):
        self.n, self.W, self.H, self.got, self.last = n, W, H, 0, None
        cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, start):.4f}", "-i", path, "-an",
               "-vf", vf, "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=W * H * 3 * 2)

    def read(self):
        size = self.W * self.H * 3
        buf = self.p.stdout.read(size) if self.got < self.n else b""
        if len(buf) == size:
            self.last = np.frombuffer(buf, np.uint8).reshape(self.H, self.W, 3).copy()
        elif self.last is None:
            self.last = np.zeros((self.H, self.W, 3), np.uint8)
        self.got += 1
        return self.last.copy()

    def close(self):
        try:
            self.p.stdout.close()
            self.p.kill()
            self.p.wait(timeout=5)
        except Exception:
            pass


def warp(frame, s, ax, ay):
    if abs(s - 1.0) < 1e-4:
        return frame
    M = np.float32([[s, 0, (1 - s) * ax], [0, s, (1 - s) * ay]])
    return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def zoom_blur(frame, s0, s1, ax, ay, samples):
    if abs(s1 - s0) < 0.01:
        return warp(frame, s1, ax, ay)
    acc = np.zeros(frame.shape, np.float32)
    for k in range(samples):
        acc += warp(frame, lerp(s0, s1, k / (samples - 1)), ax, ay)
    return (acc / samples).astype(np.uint8)


def shot_scale(sh, t):
    a = sh.get("anim")
    if not a:
        return sh["zoom"]
    if t < a["t"]:
        return a["from"]
    p = (t - a["t"]) / max(a["dur_s"], 1e-3)
    return lerp(a["from"], a["to"], ease(p, "inout" if a["name"] == "slow_push" else "out"))


def render(work, out, max_seconds=None):
    style = load_style()
    plan = load_json(os.path.join(work, "plan.json"))
    W, H, fps = plan["width"], plan["height"], plan["fps"]
    N = int(round(plan["duration"] * fps))
    if max_seconds:
        N = min(N, int(max_seconds * fps))
    ax, ay = style["motion"]["zoom_anchor"][0] * W, style["motion"]["zoom_anchor"][1] * H
    luts = os.path.join(ASSETS, "luts").replace("\\", "/").replace(":", "\\:")
    grade = style["color"]["talking_head_filter"].replace("{LUTS}", luts)
    warm_f = style["color"]["broll_warm_filter"].replace("{LUTS}", luts)
    zb = style["transitions"]["zoom_blur"]
    intro = style["structure"]["hook"]
    lang_upper = has_upper(plan.get("language", "en"))
    caps = CaptionRenderer(style, plan)
    cards = CardRenderer(style, plan)
    broll = [b for b in plan.get("broll", []) if b.get("path")]
    shots = plan["shots"]
    zb_cuts = [sh["t0"] for sh in shots if sh.get("enter") == "zoom_blur"]

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    mix = os.path.join(work, "mix.wav")
    enc = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps),
           "-i", "-"]
    if os.path.exists(mix):
        enc += ["-i", mix, "-map", "0:v", "-map", "1:a", "-c:a", style["format"]["acodec"],
                "-b:a", style["format"]["audio_bitrate"], "-ar", str(style["format"]["sample_rate"]), "-ac", "2"]
    enc += ["-vf", "scale=out_color_matrix=bt709:out_range=tv",  # RGB->YUV with the same matrix we tag
            "-c:v", style["format"]["vcodec"], "-preset", style["format"]["preset"], "-crf", str(style["format"]["crf"]),
            "-pix_fmt", style["format"]["pix_fmt"], "-colorspace", "bt709", "-color_primaries", "bt709",
            "-color_trc", "bt709", "-movflags", "+faststart", "-shortest", out]
    encp = subprocess.Popen(enc, stdin=subprocess.PIPE)

    # frame -> segment lookup
    seg_frames = []
    for i, s in enumerate(plan["segments"]):
        seg_frames += [i] * int(round((s["t1"] - s["t0"]) * fps))
    seg_frames += [len(plan["segments"]) - 1] * max(0, N - len(seg_frames))
    cur_seg, reader = -1, None
    b_state = {}
    prev_scale = 1.0
    log_frames = []
    for k in range(N):
        t = k / fps
        si = seg_frames[k]
        if si != cur_seg:
            if reader:
                reader.close()
            s = plan["segments"][si]
            nfr = seg_frames.count(si)
            reader = Reader(plan["sources"][s["src"]]["norm"], s["in"], nfr, grade, W, H, fps)
            cur_seg = si
        frame = reader.read()
        sh = next((x for x in shots if x["t0"] - 1e-6 <= t < x["t1"]), shots[-1])
        b = next((x for x in broll if x["t0"] <= t < x["t1"]), None)
        scale = shot_scale(sh, t)
        if b is not None:
            bi = broll.index(b)
            if bi not in b_state:
                nfr = int(round((b["t1"] - b["t0"]) * fps)) + 2
                vf = (f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={fps},setsar=1,"
                      + (style["color"]["broll_mono_filter"] if b["treatment"].startswith("mono") else warm_f))
                b_state[bi] = {"r": Reader(b["path"], b.get("src_in", 0.0), nfr, vf, W, H, fps), "face": None, "fk": -99}
            st = b_state[bi]
            frame = st["r"].read()
            if b["treatment"].endswith("kenburns"):
                kb = style["motion"]["broll_kenburns"]
                scale = lerp(kb["from"], kb["to"], (t - b["t0"]) / max(0.1, b["t1"] - b["t0"]))
            else:
                scale = 1.0
            frame = warp(frame, scale, W / 2, H / 2)
            if b["treatment"].endswith("halo"):
                if k - st["fk"] >= 2:  # detect every 2nd frame, then glide (no visible jumps)
                    f = detect(frame)
                    if len(f):
                        x, y, w, h, _ = max(f, key=lambda r: r[4])  # most confident face, as in the reference
                        st["target"] = np.array([x + w / 2, y + h / 2, h])
                    st["fk"] = k
                if st.get("target") is not None:
                    st["face"] = st["target"] if st["face"] is None else 0.85 * st["face"] + 0.15 * st["target"]
                if st["face"] is not None:
                    hc = style["graphics"]["halo"]
                    a = ease((t - b["t0"] - 0.1) / 0.3)
                    if a > 0:
                        cx, cy, fh = st["face"]
                        r = int(fh * hc["radius_face_h"])
                        c = (int(cx), int(cy + fh * hc["center_dy_face_h"]))
                        layer = frame.copy()
                        col = hex_rgb(hc["color"])[::-1]
                        glow = np.zeros_like(frame)
                        cv2.circle(glow, c, r, col, hc["width_px"] + hc["glow_px"], cv2.LINE_AA)
                        glow = cv2.GaussianBlur(glow, (0, 0), hc["glow_px"])
                        layer = cv2.add(layer, (glow * 0.8).astype(np.uint8))
                        cv2.circle(layer, c, r, col, hc["width_px"], cv2.LINE_AA)
                        cv2.addWeighted(layer, a, frame, 1 - a, 0, frame)
        else:
            # zoom-blur transition around cut T: 3 frames out, 5 frames in
            T = next((c for c in zb_cuts if c - zb["out_frames"] / fps - 1e-6 <= t < c + zb["in_frames"] / fps), None)
            if T is not None and t < T:
                j = int(round((t - (T - zb["out_frames"] / fps)) * fps)) + 1
                s_prev = scale * zb["out_scale_per_frame"] ** (j - 1)
                s_cur = scale * zb["out_scale_per_frame"] ** j
                frame = zoom_blur(frame, s_prev, s_cur, ax, ay, zb["blur_samples"])
                scale = s_cur
            elif T is not None:
                j = int(round((t - T) * fps))
                s_cur = scale * lerp(zb["in_start_scale"], 1.0, ease(j / zb["in_frames"]))
                s_prev = prev_scale if j == 0 else scale * lerp(zb["in_start_scale"], 1.0, ease((j - 1) / zb["in_frames"]))
                frame = zoom_blur(frame, s_prev, s_cur, ax, ay, zb["blur_samples"])
                scale = s_cur
            else:
                if sh.get("enter") == "intro" and t < intro["intro_zoom"]["dur_s"]:
                    scale *= lerp(intro["intro_zoom"]["from"], intro["intro_zoom"]["to"], ease(t / intro["intro_zoom"]["dur_s"]))
                frame = warp(frame, scale, ax, ay)
            if t < intro["intro_blur"]["frames"] / fps:
                sig = intro["intro_blur"]["sigma_px_start"] * (1 - k / intro["intro_blur"]["frames"])
                if sig > 0.5:
                    small = cv2.resize(frame, (W // 4, H // 4))
                    small = cv2.GaussianBlur(small, (0, 0), sig / 4)
                    frame = cv2.resize(small, (W, H))
        prev_scale = scale
        cards.draw(frame, t)
        caps.draw(frame, t, plan["captions"], lang_upper,
                  suppressed=range(len(plan["captions"])) if cards.acronym_active(t) else ())
        encp.stdin.write(frame.tobytes())
        log_frames.append(round(scale, 4))
        if k % (fps * 10) == 0:
            log(f"render {t:5.1f}s / {N / fps:.1f}s")
    if reader:
        reader.close()
    for st in b_state.values():
        st["r"].close()
    encp.stdin.close()
    encp.wait()
    save_json({"scale_per_frame": log_frames, "caption_layout": caps.layout_log,
               "card_bottoms": [x[2] for x in cards.log[-1:]] if cards.log else [],
               "card_max_bottom_px": max([x[2] for x in cards.log], default=0)}, os.path.join(work, "render_log.json"))
    log("wrote", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seconds", type=float, default=None)
    a = ap.parse_args()
    render(a.work, a.out, a.max_seconds)
