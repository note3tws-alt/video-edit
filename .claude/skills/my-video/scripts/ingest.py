"""Step a: ingest raw clips -> 1080x1920 / 30 fps CFR / 48 kHz intermediates + 16 kHz mono WAV for ASR.

Usage: python ingest.py --work WORK clip1.mp4 [clip2.mov ... | folder]
Writes WORK/sources.json.
"""
import argparse
import os
import re

from common import load_style, log, media_info, run, save_json
from faces import face_center

VIDEO_EXT = (".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".mts")


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def collect(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += [os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(VIDEO_EXT)]
        elif os.path.isfile(p):
            files.append(p)
    return sorted(dict.fromkeys(files), key=lambda f: natural_key(os.path.basename(f)))


def reframe_filter(info, W, H, face):
    """Cover-crop to W:H, keeping the face horizontally inside the frame for non-vertical footage."""
    sw, sh = info["width"], info["height"]
    target = W / H
    if abs(sw / sh - target) < 0.01:
        return f"scale={W}:{H}:flags=lanczos"
    if sw / sh > target:  # too wide: crop width
        cw = int(round(sh * target)) // 2 * 2
        cx = (face["cx"] if face else 0.5) * sw
        x = int(min(max(cx - cw / 2, 0), sw - cw))
        return f"crop={cw}:{sh}:{x}:0,scale={W}:{H}:flags=lanczos"
    ch = int(round(sw / target)) // 2 * 2
    cy = (face["cy"] if face else 0.45) * sh
    y = int(min(max(cy - ch * 0.40, 0), sh - ch))
    return f"crop={sw}:{ch}:0:{y},scale={W}:{H}:flags=lanczos"


def ingest(paths, work, style):
    fmt = style["format"]
    W, H, fps, sr = fmt["width"], fmt["height"], fmt["fps"], fmt["sample_rate"]
    os.makedirs(os.path.join(work, "norm"), exist_ok=True)
    out = []
    for i, src in enumerate(collect(paths)):
        info = media_info(src)
        if not info.get("has_video"):
            log("skip (no video):", src)
            continue
        face = face_center(src, info["duration"])
        hdr = info.get("color_transfer") in ("arib-std-b67", "smpte2084")  # iPhone HLG / HDR10 -> SDR bt709
        tonemap = ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,"
                   "zscale=t=bt709:m=bt709:r=tv,format=yuv420p,") if hdr else ""
        vf = tonemap + reframe_filter(info, W, H, face) + f":out_color_matrix=bt709:out_range=tv,fps={fps},format=yuv420p,setsar=1"
        sid = f"S{i + 1:02d}"
        norm = os.path.join(work, "norm", f"{sid}.mov")
        wav = os.path.join(work, "norm", f"{sid}_16k.wav")
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", src]
        if not info["has_audio"]:
            cmd += ["-f", "lavfi", "-i", f"anullsrc=r={sr}:cl=mono"]
        cmd += ["-map", "0:v:0", "-map", "0:a:0" if info["has_audio"] else "1:a:0", "-vf", vf,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-g", str(fps), "-bf", "0",
                "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
                "-c:a", "pcm_s16le", "-ar", str(sr), "-ac", "1", "-shortest"]
        run(cmd + [norm])
        run(["ffmpeg", "-y", "-v", "error", "-i", norm, "-vn", "-ac", "1", "-ar", "16000", wav])
        ninfo = media_info(norm)
        out.append({"id": sid, "src": os.path.abspath(src), "norm": os.path.abspath(norm), "wav16": os.path.abspath(wav),
                    "duration": ninfo["duration"], "orig": {k: info.get(k) for k in ("width", "height", "fps", "duration")},
                    "face": face})
        log(f"ingested {src} -> {sid} ({info.get('width')}x{info.get('height')} @ {info.get('fps', 0):.2f} fps, "
            f"{info['duration']:.1f}s)")
    if not out:
        raise SystemExit("no usable clips")
    save_json({"sources": out}, os.path.join(work, "sources.json"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("clips", nargs="+")
    a = ap.parse_args()
    ingest(a.clips, a.work, load_style())
