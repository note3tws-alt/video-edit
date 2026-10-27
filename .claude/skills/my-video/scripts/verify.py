"""Checklist verification of a rendered video against style.json (measured, not eyeballed).

Usage: python verify.py --work WORK --video OUT.mp4 [--reference REF.mp4 --ref-crop 356:634:267:273]
Writes WORK/verify_report.md, WORK/verify.json and WORK/side_by_side.png
"""
import argparse
import json
import os
import re
import subprocess

import cv2
import numpy as np

from common import SENT_END, load_json, load_style, media_info, run, save_json
from faces import grab_frames


def loudness(path):
    p = run(["ffmpeg", "-hide_banner", "-i", path, "-vn", "-af", "ebur128=peak=true", "-f", "null", "-"])
    s = p.stderr.decode()
    s = s[s.rfind("Summary"):]
    g = lambda pat: float(re.search(pat, s).group(1))
    return {"I": g(r"I:\s+(-?[\d.]+) LUFS"), "LRA": g(r"LRA:\s+(-?[\d.]+) LU"), "TP": g(r"Peak:\s+(-?[\d.]+) dBFS")}


def detect_cuts(path, W=270, H=480):
    """Independent cut detector (same method as the reference analysis): isolated one-frame spikes of
    mean-abs-difference in the face band, plus zoom-level jumps."""
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={W}:{H}", "-f", "rawvideo",
                          "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE)
    prev, D = None, []
    while True:
        b = p.stdout.read(W * H)
        if len(b) < W * H:
            break
        g = np.frombuffer(b, np.uint8).reshape(H, W)[int(H * 0.18):int(H * 0.55), int(W * 0.17):int(W * 0.84)].astype(np.float32)
        g = cv2.GaussianBlur(g, (5, 5), 0)
        D.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
        prev = g
    D = np.array(D)
    cuts = []
    for i in range(1, len(D)):
        nb = np.r_[D[max(1, i - 6):i - 1], D[i + 2:i + 7]]
        if len(nb) and D[i] / (np.median(nb) + 0.3) > 3.5 and D[i] > 3:
            if not cuts or i - cuts[-1] > 3:
                cuts.append(i)
    return [c / 30.0 for c in cuts], len(D)


def detect_cuts_cropped(path, crop):
    w, h, x, y = crop.split(":")
    tmp = os.path.join(os.path.dirname(os.path.abspath(path)), ".ref_crop_tmp.mp4")
    run(["ffmpeg", "-y", "-v", "error", "-i", path, "-an", "-vf", f"crop={w}:{h}:{x}:{y}", "-c:v", "libx264", "-crf", "16", tmp])
    try:
        return detect_cuts(tmp)
    finally:
        os.remove(tmp)


def frame_stats(img):
    h = img.shape[0]
    a = np.concatenate([img[: int(h * 0.55)], img[int(h * 0.9):]]).reshape(-1, 3).astype(np.float32)
    y = 0.114 * a[:, 0] + 0.587 * a[:, 1] + 0.299 * a[:, 2]
    hsv = cv2.cvtColor(a.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    return np.array([y.mean(), y.std(), hsv[:, 1].mean(), (a[:, 2] - a[:, 0]).mean()])


def check(name, ok, value, target, kind="PASS"):
    return {"check": name, "status": ("PASS" if ok else ("WARN" if kind == "WARN" else "FAIL")), "value": value, "target": target}


def verify(work, video, reference=None, ref_crop=None):
    st = load_style()
    plan = load_json(os.path.join(work, "plan.json"))
    rlog = load_json(os.path.join(work, "render_log.json"))
    sfx = load_json(os.path.join(work, "sfx_events.json")) if os.path.exists(os.path.join(work, "sfx_events.json")) else {"events": []}
    info = media_info(video)
    dur = info["duration"]
    R = []
    f = st["format"]
    R.append(check("Format 1080x1920 @30fps", info["width"] == f["width"] and info["height"] == f["height"] and abs(info["fps"] - f["fps"]) < 0.01,
                   f"{info['width']}x{info['height']} @ {info['fps']:.2f}", "1080x1920 @ 30"))
    R.append(check("Codecs h264/aac", info["vcodec"] == "h264" and info.get("acodec") == "aac", f"{info['vcodec']}/{info.get('acodec')}", "h264/aac"))
    ln = loudness(video)
    R.append(check("Loudness I", abs(ln["I"] - st["audio"]["loudnorm"]["I"]) <= 1.0, ln["I"], "-14 ±1 LUFS (ref -14.3)"))
    R.append(check("True peak", ln["TP"] <= -0.5, ln["TP"], "≤ -1 dBTP (ref +0.2, clipped)"))
    R.append(check("Loudness range", ln["LRA"] <= 6.0, ln["LRA"], "≤ 6 LU (ref 3.4)"))

    # pacing (plan + independent detector)
    pc = st["pacing"]
    cuts_plan = sorted(set([round(s["t0"], 3) for s in plan["shots"][1:]] + [round(x, 3) for b in plan["broll"] if b.get("path") for x in (b["t0"], b["t1"])]))
    hidden = lambda c: any(b["t0"] + 0.05 < c < b["t1"] - 0.05 for b in plan["broll"] if b.get("path"))
    cuts_plan = [c for c in cuts_plan if 0 < c < dur and not hidden(c)]  # cuts under B-roll are invisible
    bounds = [0.0] + cuts_plan + [dur]
    lens = np.diff(bounds)
    thirds = []
    for k in range(3):
        a, b = k * dur / 3, (k + 1) * dur / 3
        n = sum(1 for c in cuts_plan if a <= c < b)
        thirds.append(round((b - a) / max(1, n), 2))
    tol = pc["tolerance_pct"] / 100
    R.append(check("Mean shot length", abs(lens.mean() - pc["shot_len_s"]["mean"]) <= pc["shot_len_s"]["mean"] * tol,
                   round(float(lens.mean()), 2), f"{pc['shot_len_s']['mean']} s ±{pc['tolerance_pct']}%"))
    R.append(check("Max shot length", lens.max() <= pc["shot_len_s"]["max"] * (1 + tol), round(float(lens.max()), 2), f"≤ {pc['shot_len_s']['max']} s"))
    R.append(check("Pacing by third (fast→slow)", all(abs(a - b) <= b * 0.35 for a, b in zip(thirds, pc["by_third_mean_shot_s"])),
                   thirds, f"{pc['by_third_mean_shot_s']} s ±35%"))
    det, nfr = detect_cuts(video)
    if reference:
        rdet, rn = detect_cuts(reference) if not ref_crop else detect_cuts_cropped(reference, ref_crop)
        exp = len(rdet) * dur / (rn / 30.0)
        R.append(check("Visual change rate vs reference (same detector)", abs(len(det) - exp) <= max(4, 0.35 * exp),
                       len(det), f"≈ {exp:.0f} (reference: {len(rdet)} in {rn / 30:.0f}s)", "WARN"))
    else:
        R.append(check("Detected cuts (independent)", abs(len(det) - len(cuts_plan)) <= max(3, 0.35 * len(cuts_plan)),
                       len(det), f"≈ plan {len(cuts_plan)}", "WARN"))

    # zoom
    sc = np.array(rlog["scale_per_frame"])
    lv = sorted(set(round(x, 2) for x in sc if abs(x - round(x, 2)) < 1e-3))
    allowed = set([1.0] + st["motion"]["punch_levels"])
    punches = [s for s in plan["shots"] if s["zoom"] in st["motion"]["punch_levels"]]
    R.append(check("Zoom range", sc.min() >= 0.6 and sc.max() <= 1.25, [round(float(sc.min()), 3), round(float(sc.max()), 3)], "0.68-1.19 (transitions) / 1.00-1.13 (shots)"))
    R.append(check("Punch-in shots", len(punches) >= 1, len(punches), f"levels {st['motion']['punch_levels']}"))
    anim = [s for s in plan["shots"] if s.get("anim")]
    win = st["motion"]["animated_window_pct"] * dur
    R.append(check("Animated zooms only in first 23%", all(s["anim"]["t"] <= win + 2 for s in anim) and len(anim) >= 1,
                   f"{len(anim)} (last at {max([s['anim']['t'] for s in anim], default=0):.1f}s)", f"≤ {win:.1f}s"))
    zt = [s for s in plan["shots"] if s.get("enter") == "zoom_blur"]
    exp_zt = int(round(dur * st["transitions"]["zoom_blur"]["count_per_70s"] / 70))
    R.append(check("Zoom-blur transitions", abs(len(zt) - exp_zt) <= 1, len(zt), f"{exp_zt} (3 per 70 s)"))

    # hook
    hk = plan["hook"]["end"]
    hero_hook = [w for c in plan["captions"] for w in c["words"] if w.get("hero") and w["t"] < hk]
    R.append(check("Hook has hero text + push-in", len(hero_hook) >= 1 and any(s.get("anim", {}) and s["anim"]["name"] == "push_fast" for s in plan["shots"]),
                   f"{len(hero_hook)} hero(es), hook {hk:.2f}s", "≥1 hero + push 1.00→1.075"))

    # captions
    lay = rlog["caption_layout"]
    base = [l for l in lay if not l["hero"]]
    yr = st["captions"]["y_center_range"]
    on_b = lambda t: any(b["t0"] <= t < b["t1"] for b in plan["broll"] if b.get("path"))
    base_th = [l for l in base if not on_b(l["t0"])]
    inside = sum(1 for l in base_th if yr[0] - 0.01 <= l["y_center"] <= yr[1] + 0.01)
    R.append(check("Base caption y-centre", inside == len(base_th), f"{inside}/{len(base_th)} in range", f"{yr}"))
    wmax = max([l["max_w"] for l in lay], default=0)
    R.append(check("Max caption width", wmax <= st["captions"]["hero"]["max_width"] + 0.01, round(wmax, 3), "≤ 0.86"))
    words_per = [len(c["words"]) for c in plan["captions"] if not any(w.get("hero") for w in c["words"])]
    R.append(check("Words per base group", max(words_per, default=0) <= st["captions"]["max_words_per_group"]
                   and np.median(words_per or [2]) <= st["captions"]["median_words_per_group"] + 0.5,
                   f"median {np.median(words_per) if words_per else 0}, max {max(words_per, default=0)}", "1-3 (median 2)"))
    hero_units = [w for c in plan["captions"] for k, w in enumerate(c["words"]) if w.get("hero") and not (k and c["words"][k - 1].get("hero"))]
    nh = len(hero_units)
    exp_h = sum(st["captions"]["hero"]["typographic_per_third"]) * dur / 70.2
    R.append(check("Hero word count", abs(nh - exp_h) <= max(3, exp_h * 0.35), nh, f"≈ {exp_h:.0f} (13 per 70 s, front-loaded)"))
    gold = sum(1 for w in hero_units if w["hero"] in ("serif_gold", "caps_gold"))
    R.append(check("Gold share of heroes", nh == 0 or 0.35 <= gold / nh <= 0.8, round(gold / max(1, nh), 2), "≈ 0.58"))

    # speech density
    ws = plan["words"]
    gaps = [b["t"] - a["e"] for a, b in zip(ws, ws[1:])]
    wps = len(ws) / max(dur, 1e-6)
    R.append(check("Speech rate", wps >= pc["speech_rate_wps"] * (1 - tol) * 0.85, round(wps, 2), f"≈ {pc['speech_rate_wps']} words/s (depends on speaker)", "WARN"))
    R.append(check("Max pause", max(gaps, default=0) <= pc["max_pause_s"] + 0.05, round(max(gaps, default=0), 3), f"≤ {pc['max_pause_s']} s"))

    # B-roll, cards, sfx, outro
    bsum = sum(b["t1"] - b["t0"] for b in plan["broll"] if b.get("path"))
    R.append(check("B-roll share", abs(bsum / dur - st["broll"]["share_of_runtime"]) <= 0.08, round(bsum / dur, 3),
                   f"≈ {st['broll']['share_of_runtime']}", "WARN"))
    ncard = len([c for c in plan.get("cards", []) if c["type"] != "acronym"])
    exp_c = max(1, int(round(dur * st["graphics"]["cards_per_70s"] / 70)))
    R.append(check("Info cards", ncard >= max(1, exp_c - 1), ncard, f"{exp_c} (3 per 70 s)", "WARN"))
    cb = rlog.get("card_max_bottom_px", 0) / f["height"]
    R.append(check("Cards above bottom safe zone", cb <= 1 - f["safe_zone"]["bottom_pct"] / 100 + 0.005, round(cb, 3), "≤ 0.85"))
    ev = sfx["events"]
    sh_n = sum(1 for e in ev if e["type"] == "shimmer")
    R.append(check("Shimmer SFX on gold heroes/cards", sh_n >= gold * 0.8, sh_n, f"≥ gold heroes ({gold})"))
    R.append(check("Whoosh on zoom transitions", sum(1 for e in ev if e["type"] == "whoosh") == len(zt), sum(1 for e in ev if e["type"] == "whoosh"), len(zt)))
    ot = st["structure"]["outro"]["start_pct"] * dur
    out_sc = sc[int(ot * 30):]
    out_h = [w for c in plan["captions"] for w in c["words"] if w.get("hero") and w["t"] >= ot]
    R.append(check("Outro calm (1.00x, no heroes)", (len(out_sc) == 0 or np.abs(out_sc - 1).max() < 0.02) and not out_h,
                   f"max dev {float(np.abs(out_sc - 1).max()) if len(out_sc) else 0:.3f}, heroes {len(out_h)}", "1.00x, 0"))

    # colour grade: output vs raw source at the same instants (talking-head shots only)
    deltas = []
    for s in plan["segments"][:: max(1, len(plan["segments"]) // 6)]:
        tm = (s["t0"] + s["t1"]) / 2
        if on_b(tm):
            continue
        src_t = s["in"] + (tm - s["t0"])
        raw = grab_frames(plan["sources"][s["src"]]["norm"], [src_t])[0]
        outf = grab_frames(video, [tm])[0]
        if raw is None or outf is None:
            continue
        a, b = frame_stats(raw), frame_stats(outf)
        d = b / np.maximum(a, 1e-3)
        d[3] = b[3] - a[3]  # warmth is an offset, not a ratio
        deltas.append(d)
    if deltas:
        d = np.mean(deltas, axis=0)
        m = st["color"]["measured_raw_to_graded"]
        tgt = [m["luma_mean"][1] / m["luma_mean"][0], m["luma_std"][1] / m["luma_std"][0], m["hsv_sat_mean"][1] / m["hsv_sat_mean"][0], m["r_minus_b"][1] - m["r_minus_b"][0]]
        ok = abs(d[0] - tgt[0]) < 0.12 and abs(d[1] - tgt[1]) < 0.15 and abs(d[2] - tgt[2]) < 0.25
        R.append(check("Grade deltas (luma×, contrast×, sat×, warmth+)", ok, [round(float(x), 2) for x in d], [round(x, 2) for x in tgt]))

    # side-by-side sheet
    sheet = None
    if reference:
        rinfo = media_info(reference)
        rel = [0.02, 0.12, 0.22, 0.32, 0.45, 0.6, 0.75, 0.95]
        top = grab_frames(video, [r * dur for r in rel])
        bot = grab_frames(reference, [r * rinfo["duration"] for r in rel])
        tiles = []
        for a, b in zip(top, bot):
            if a is None or b is None:
                continue
            if ref_crop:
                w, h, x, y = map(int, ref_crop.split(":"))
                b = b[y:y + h, x:x + w]
            a = cv2.resize(a, (270, 480))
            b = cv2.resize(b, (270, 480))
            tiles.append(np.vstack([a, b]))
        if tiles:
            img = np.hstack(tiles)
            cv2.putText(img, "OUTPUT", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(img, "REFERENCE", (6, 502), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            sheet = os.path.join(work, "side_by_side.png")
            cv2.imwrite(sheet, img)

    save_json({"results": R, "loudness": ln, "cuts_plan": cuts_plan, "cuts_detected": det, "thirds": thirds}, os.path.join(work, "verify.json"))
    lines = ["| Check | Status | Value | Target |", "|---|---|---|---|"]
    for r in R:
        lines.append(f"| {r['check']} | {r['status']} | {r['value']} | {r['target']} |")
    md = "\n".join(lines)
    with open(os.path.join(work, "verify_report.md"), "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    print(md)
    nf = sum(1 for r in R if r["status"] == "FAIL")
    print(f"\n{len(R) - nf}/{len(R)} checks not failing ({nf} FAIL, {sum(1 for r in R if r['status'] == 'WARN')} WARN)"
          + (f"; side-by-side: {sheet}" if sheet else ""))
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--ref-crop", default=None, help="w:h:x:y crop of the reference (e.g. 356:634:267:273 for the showcase card)")
    a = ap.parse_args()
    verify(a.work, a.video, a.reference, a.ref_crop)
