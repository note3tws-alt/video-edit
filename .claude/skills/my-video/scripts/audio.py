"""Step f (audio): splice voice per plan segments, voice chain, SFX at measured relative levels,
optional music (user track / library / generated bed) with ducking, two-pass loudnorm.

Usage: python audio.py --work WORK [--music none|auto|/path/track.mp3]
Writes WORK/mix.wav and WORK/sfx_events.json
"""
import argparse
import glob
import json
import os
import subprocess

import numpy as np
import soundfile as sf

from common import ASSETS, load_json, load_style, log, run, save_json
from sfx import SR, build_all, generate_music_bed

GOLD = ("serif_gold", "caps_gold")


def read_audio(path, sr=SR):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                       stdout=subprocess.PIPE, check=True)
    return np.frombuffer(p.stdout, np.float32).copy()


def db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-12)


def active_rms_db(x, sr=SR, floor_db=-45):
    hop = sr // 50
    fr = x[: len(x) // hop * hop].reshape(-1, hop)
    r = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-12)
    act = r[r > floor_db]
    return float(10 * np.log10(np.mean(10 ** (act / 10)))) if len(act) else -60.0


def splice_voice(plan, style):
    n = int(round(plan["duration"] * SR))
    out = np.zeros(n, np.float32)
    srcs = {}
    xf = int(style["silence_removal"]["audio_crossfade_ms"] * SR / 1000)
    for s in plan["segments"]:
        if s["src"] not in srcs:
            srcs[s["src"]] = read_audio(plan["sources"][s["src"]]["norm"])
        a = srcs[s["src"]]
        i0, i1 = int(round(s["in"] * SR)), int(round(s["out"] * SR))
        o0 = int(round(s["t0"] * SR))
        chunk = a[i0:i1].copy()
        L = min(len(chunk), n - o0)
        chunk = chunk[:L]
        if L > 2 * xf:
            chunk[:xf] *= np.linspace(0, 1, xf)
            chunk[-xf:] *= np.linspace(1, 0, xf)
        out[o0:o0 + L] += chunk
    return out


def sfx_events(plan, style):
    cfg = style["audio"]["sfx"]
    ev = []
    hook_end = plan["hook"]["end"]
    for c in plan["captions"]:
        for w in c["words"]:
            if w.get("hero") in GOLD or (w.get("hero") and c.get("pos") == "top" and w["t"] < hook_end):
                ev.append({"type": "shimmer", "t": round(w["t"] + cfg["shimmer"].get("offset_s", 0.10), 3)})
    for card in plan.get("cards", []):
        if card["type"] in ("compare", "timeline", "checklist"):
            ev.append({"type": "shimmer", "t": round(card["t0"] + 0.05, 3)})
        for it in card.get("items", []) + card.get("points", []):
            if it.get("t") is not None:
                ev.append({"type": "tap", "t": round(it["t"], 3)})
    for sh in plan["shots"]:
        if sh.get("enter") == "zoom_blur":
            ev.append({"type": "whoosh", "t": round(sh["t0"] - 3 / plan["fps"] - cfg["whoosh"]["lead_s"], 3)})
    # one shimmer at most every 0.6 s
    ev.sort(key=lambda e: e["t"])
    out, last = [], {}
    for e in ev:
        if e["t"] - last.get(e["type"], -9) >= 0.6:
            out.append(e)
            last[e["type"]] = e["t"]
    return out


def loudnorm_2pass(inp, outp, ln):
    f1 = f"loudnorm=I={ln['I']}:TP={ln['TP']}:LRA={ln['LRA']}:print_format=json"
    p = run(["ffmpeg", "-hide_banner", "-i", inp, "-af", f1, "-f", "null", "-"])
    txt = p.stderr.decode()
    m = json.loads(txt[txt.rfind("{"):txt.rfind("}") + 1])
    f2 = (f"loudnorm=I={ln['I']}:TP={ln['TP']}:LRA={ln['LRA']}:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
          f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    run(["ffmpeg", "-y", "-v", "error", "-i", inp, "-af", f2 + f",aresample={SR}", "-ac", "2", "-c:a", "pcm_s24le", outp])


def pick_music(choice, duration, work):
    if not choice or choice == "none":
        return None
    if choice == "auto":
        lib = sorted(glob.glob(os.path.join(ASSETS, "music", "*.*")))
        if lib:
            return lib[0]
        return generate_music_bed(duration, os.path.join(work, "music_generated.wav"))
    return choice


def mix(work, music="none"):
    style = load_style()
    plan = load_json(os.path.join(work, "plan.json"))
    acfg = style["audio"]
    voice = splice_voice(plan, style)
    raw = os.path.join(work, "voice_raw.wav")
    proc = os.path.join(work, "voice_proc.wav")
    sf.write(raw, voice, SR, subtype="FLOAT")
    run(["ffmpeg", "-y", "-v", "error", "-i", raw, "-af", acfg["voice_chain"], "-c:a", "pcm_f32le", proc])
    v, _ = sf.read(proc, dtype="float32")
    v_db = active_rms_db(v)
    bus = v.astype(np.float64).copy()

    paths = build_all()
    events = sfx_events(plan, style)
    cache = {}
    for e in events:
        if e["type"] not in cache:
            x, sr = sf.read(paths[e["type"]], dtype="float32")
            gain_db = v_db - acfg["sfx"][e["type"]]["db_below_voice"] - active_rms_db(x, floor_db=-60)
            cache[e["type"]] = x * 10 ** (gain_db / 20)
        x = cache[e["type"]]
        i0 = int(max(0, e["t"]) * SR)
        L = min(len(x), len(bus) - i0)
        if L > 0:
            bus[i0:i0 + L] += x[:L]

    mpath = pick_music(music if music is not None else plan["audio"].get("music"), plan["duration"], work)
    if mpath:
        m = read_audio(mpath)
        reps = int(np.ceil(len(bus) / max(1, len(m))))
        m = np.tile(m, reps)[: len(bus)]
        mc = acfg["music"]
        m *= 10 ** ((v_db - mc["level_db_below_voice"] - active_rms_db(m)) / 20)
        hop = SR // 100
        env = np.sqrt(np.convolve(v ** 2, np.ones(hop * 5) / (hop * 5), mode="same"))
        target = np.where(env > mc["duck"]["threshold"] * np.abs(v).max(), 10 ** (-6 / 20), 1.0)
        a_att = 1 - np.exp(-1 / (mc["duck"]["attack"] / 1000 * SR))
        a_rel = 1 - np.exp(-1 / (mc["duck"]["release"] / 1000 * SR))
        dec = target[::hop]  # one-pole attack/release smoothing on a 10 ms grid
        gd = np.empty_like(dec)
        gd[0] = 1.0
        for i in range(1, len(dec)):
            a = a_att * hop if dec[i] < gd[i - 1] else a_rel * hop
            gd[i] = gd[i - 1] + min(1.0, a) * (dec[i] - gd[i - 1])
        g = np.repeat(gd, hop)[: len(bus)]
        fade = np.minimum(1, np.minimum(np.arange(len(m)) / (SR * 1.0), (len(m) - np.arange(len(m))) / (SR * 1.5)))
        bus += m * g * fade
        log("music:", mpath)

    pre = os.path.join(work, "mix_pre.wav")
    sf.write(pre, bus.astype(np.float32), SR, subtype="FLOAT")
    out = os.path.join(work, "mix.wav")
    loudnorm_2pass(pre, out, acfg["loudnorm"])
    save_json({"events": events, "voice_active_rms_db": round(v_db, 2), "music": mpath}, os.path.join(work, "sfx_events.json"))
    log(f"audio: {len(events)} SFX events, music={'yes' if mpath else 'no'}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--music", default="none")
    a = ap.parse_args()
    mix(a.work, a.music)
