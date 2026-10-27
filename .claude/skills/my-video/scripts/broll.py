"""Fill plan.json B-roll slots: user-supplied clips first (--broll-dir), then Pexels (PEXELS_API_KEY).

Usage: python broll.py --work WORK [--broll-dir DIR] [--no-pexels]
Slots that cannot be filled are dropped (the talking head continues), and the reason is logged.
"""
import argparse
import json
import os
import re
import urllib.parse
import urllib.request

from common import load_json, load_style, log, media_info, norm_word, save_json
from ingest import VIDEO_EXT, natural_key


def user_clips(d):
    if not d or not os.path.isdir(d):
        return []
    fs = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(VIDEO_EXT)]
    return sorted(fs, key=lambda f: natural_key(os.path.basename(f)))


def best_user_clip(slot, clips, used):
    q = set(norm_word(w) for w in re.split(r"\s+", slot.get("query", "")) if w)
    free = [c for c in clips if c not in used] or clips
    scored = sorted(free, key=lambda c: -len(q & set(norm_word(x) for x in re.split(r"[\W_]+", os.path.basename(c)))))
    return scored[0] if scored else None


def pexels_search(query, cfg, key):
    url = "https://api.pexels.com/videos/search?" + urllib.parse.urlencode(
        {"query": query, "orientation": cfg["orientation"], "per_page": cfg["per_page"], "size": "medium"})
    req = urllib.request.Request(url, headers={"Authorization": key, "User-Agent": "my-video-skill"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    best = None
    for v in data.get("videos", []):
        for f in v.get("video_files", []):
            w, h = f.get("width") or 0, f.get("height") or 0
            if h < cfg["min_height"] or h <= w or f.get("file_type") != "video/mp4":
                continue
            score = abs(h - 1920) + (0 if v.get("duration", 0) >= 5 else 5000)
            if best is None or score < best[0]:
                best = (score, f["link"], v.get("id"), v.get("url"), v.get("user", {}).get("name"))
    return best


def fill(work, broll_dir=None, use_pexels=True, refill=False):
    style = load_style()
    plan_p = os.path.join(work, "plan.json")
    plan = load_json(plan_p)
    clips = user_clips(broll_dir)
    used = set()
    key = os.environ.get(style["broll"]["pexels"]["api_env"])
    os.makedirs(os.path.join(work, "broll"), exist_ok=True)
    kept = []
    for i, slot in enumerate(plan.get("broll", [])):
        if slot.get("path") and os.path.exists(slot["path"]) and not refill:
            kept.append(slot)
            continue
        path, credit = None, None
        slot["t1"] = slot.get("planned_t1", slot["t1"])  # a re-fill starts from the planned length
        if clips:
            path = best_user_clip(slot, clips, used)
            used.add(path)
        elif use_pexels and key and slot.get("query"):
            try:
                hit = pexels_search(slot["query"], style["broll"]["pexels"], key)
                if hit:
                    path = os.path.join(work, "broll", f"pexels_{hit[2]}.mp4")
                    if not os.path.exists(path):
                        urllib.request.urlretrieve(hit[1], path)
                    credit = f"Pexels video {hit[2]} by {hit[4]} ({hit[3]})"
                else:
                    log(f"B-roll slot {i}: no portrait Pexels result for '{slot['query']}'")
            except Exception as e:
                log(f"B-roll slot {i}: Pexels unavailable ({str(e)[:120]}) - is api.pexels.com / videos.pexels.com allowed?")
        elif use_pexels and not key:
            log(f"B-roll slot {i}: no user clips and {style['broll']['pexels']['api_env']} not set -> slot dropped")
        if not path:
            continue
        dur = media_info(path)["duration"]
        need = slot["t1"] - slot["t0"]
        if dur < need:
            slot["t1"] = round(slot["t0"] + dur - 0.05, 3)
        slot["src_in"] = round(max(0.0, min(dur - need, dur * 0.1)), 3) if dur > need else 0.0
        slot["path"] = os.path.abspath(path)
        slot["credit"] = credit
        kept.append(slot)
        log(f"B-roll slot {i}: {os.path.basename(path)} ({slot['t0']:.2f}-{slot['t1']:.2f}s, {slot['treatment']})")
    plan["broll"] = kept
    save_json(plan, plan_p)
    return kept


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--broll-dir", default=None)
    ap.add_argument("--no-pexels", action="store_true")
    ap.add_argument("--refill", action="store_true", help="re-pick clips for every slot (after editing queries)")
    a = ap.parse_args()
    fill(a.work, a.broll_dir, not a.no_pexels, a.refill)
