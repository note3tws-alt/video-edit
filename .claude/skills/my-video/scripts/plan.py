"""Steps b-e (planning): silence/filler removal, hook pick, pacing, zooms, transitions, B-roll slots,
caption groups + hero words, card suggestions.  Output: WORK/plan.json (human/Claude editable).

Usage: python plan.py --work WORK [--target-length 60] [--overrides WORK/overrides.json]

overrides.json (optional, written by Claude after reading the transcript):
  {"exclude": [{"src": "S01", "from": 12.3, "to": 15.0}],      # bad takes / repeated lines (source seconds)
   "hook": "auto" | "none" | {"src": "S01", "from": 40.1, "to": 43.0},
   "order": ["S02", "S01"]}                                      # optional clip order
"""
import argparse
import math
import os
import re

import numpy as np
import soundfile as sf

from common import (CLAUSE_END, SENT_END, has_upper, load_json, load_style, log, norm_word, save_json, script_of,
                    stopwords, visual_len)

NUM_RE = re.compile(r"\d")


# ------------------------------------------------------------------ words & sentences
def clean_words(tr, style, lang, overrides):
    fillers = set(norm_word(f) for f in style["silence_removal"]["fillers"].get(lang, []))
    excl = overrides.get("exclude", [])
    out = []
    order = overrides.get("order") or list(tr["sources"].keys())
    for sid in order:
        ws = tr["sources"][sid]["words"]
        prev = None
        for i, w in enumerate(ws):
            n = norm_word(w["w"])
            if not n:
                continue
            if n in fillers:
                continue
            if any(x["src"] == sid and x["from"] <= w["s"] < x["to"] for x in excl):
                continue
            if style["silence_removal"]["drop_immediate_repeats"] and prev is not None and norm_word(prev["w"]) == n \
                    and not SENT_END.search(prev["w"]) and w["s"] - prev["e"] < 0.8 and visual_len(n) > 1:
                out.pop()  # keep the second (usually the corrected) take of a stutter
            item = dict(w, src=sid)
            out.append(item)
            prev = item
    return out


def sentences(words):
    sents, cur = [], []
    for i, w in enumerate(words):
        if cur and (w["src"] != cur[-1]["src"] or w["s"] - cur[-1]["e"] > 0.9):
            sents.append(cur)
            cur = []
        cur.append(w)
        if SENT_END.search(w["w"]):
            sents.append(cur)
            cur = []
    if cur:
        sents.append(cur)
    return sents


def rms_energy(wavs, src, a, b):
    y, sr = wavs[src]
    seg = y[int(a * sr):int(b * sr)]
    return float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0


HOOK_WORDS = {
    "en": "you your why how never stop secret mistake truth nobody everyone most money free fast best worst".split(),
    "pt": "você seu sua por que como nunca pare segredo erro verdade ninguém todos dinheiro grátis melhor pior".split(),
    "hi": "आप आपका आपकी क्यों कैसे कभी गलती सच कोई सब पैसा मुफ्त सबसे".split(),
    "ml": "നിങ്ങൾ എന്തുകൊണ്ട് എങ്ങനെ ഒരിക്കലും തെറ്റ് സത്യം ആരും എല്ലാവരും പണം സൗജന്യം".split(),
}


def score_hook(sent, idx, total, wavs, lang, emax):
    text = " ".join(w["w"] for w in sent)
    dur = sent[-1]["e"] - sent[0]["s"]
    if dur < 1.2 or dur > 6.5:
        return -1
    s = 0.0
    s += 2.0 if text.strip().endswith("?") else 0
    s += 1.0 if NUM_RE.search(text) else 0
    hw = set(HOOK_WORDS.get(lang, HOOK_WORDS["en"]))
    s += 0.8 * min(2, sum(1 for w in sent if norm_word(w["w"]) in hw))
    s += 0.6 * (rms_energy(wavs, sent[0]["src"], sent[0]["s"], sent[-1]["e"]) / (emax + 1e-9))
    s += 0.5 if idx / max(total, 1) < 0.15 else 0
    s -= 0.5 if dur > 4.5 else 0
    return s


# ------------------------------------------------------------------ timeline
def build_segments(sents, style, fps):
    sr_cfg = style["silence_removal"]
    gap_cut, pre, post = sr_cfg["cut_gaps_longer_than_s"], sr_cfg["pad_before_word_s"], sr_cfg["pad_after_word_s"]
    pieces = []
    for sent in sents:
        for w in sent:
            if pieces and pieces[-1]["src"] == w["src"] and 0 <= w["s"] - pieces[-1]["words"][-1]["e"] <= gap_cut:
                pieces[-1]["words"].append(w)
            else:
                pieces.append({"src": w["src"], "words": [w]})
    segs, t = [], 0.0
    for p in pieces:
        a = max(0.0, p["words"][0]["s"] - pre)
        b = p["words"][-1]["e"] + post
        if segs and segs[-1]["src"] == p["src"] and a < segs[-1]["out"]:
            a = segs[-1]["out"]
        nfr = max(1, int(round((b - a) * fps)))
        b = a + nfr / fps
        seg = {"src": p["src"], "in": round(a, 4), "out": round(b, 4), "t0": round(t, 4), "t1": round(t + nfr / fps, 4),
               "words": p["words"]}
        segs.append(seg)
        t += nfr / fps
    return segs, t


def to_out_time(seg, ts):
    return seg["t0"] + (ts - seg["in"])


# ------------------------------------------------------------------ shots / zoom
def make_shots(segs, total, style, hook_end):
    """Shots = every source cut, plus 'punch' splits (zoom change, same take) where a shot exceeds
    1.6x the measured mean shot length of its third. The hook stays one shot, as in the reference."""
    pc = style["pacing"]
    shots = [{"t0": s["t0"], "t1": s["t1"], "kind": "cut"} for s in segs]
    if shots:
        shots[0]["kind"] = "start"
    bounds = sorted(to_out_time(s, w["s"]) for s in segs for w in s["words"][1:])
    out = []
    for sh in shots:
        cur = sh
        while True:
            third = min(2, int(3 * cur["t0"] / max(total, 1e-6)))
            target = pc["by_third_mean_shot_s"][third]
            if cur["t1"] - cur["t0"] <= target * pc["split_long_shot_factor"]:
                break
            lo = max(cur["t0"] + pc["shot_len_s"]["min"] * 3, hook_end if cur["t0"] < hook_end else 0)
            cands = [b for b in bounds if lo <= b < cur["t1"] - 0.8]
            if not cands:
                break
            want = max(cur["t0"] + target, hook_end if cur["t0"] < hook_end else 0)
            b = min(cands, key=lambda x: abs(x - want))
            out.append({"t0": cur["t0"], "t1": round(b, 4), "kind": cur["kind"]})
            cur = {"t0": round(b, 4), "t1": cur["t1"], "kind": "punch"}
        out.append(cur)
    return out


def _end_scale(sh):
    return sh["anim"]["to"] if sh.get("anim") else sh["zoom"]


def assign_zooms(shots, total, style, hook_end, first_heroes, broll):
    """Hook: 1.00 + push_fast on the hook hero. First 23 %: one animated template per shot, chosen so that
    every split is visible (start scale differs from the previous end scale by >= 0.03). After that:
    base/punch blocks (punch levels cycle 1.09, 1.05, 1.12, 1.03); a punch split always toggles.
    Outro: 1.00."""
    mo = style["motion"]
    outro_t = style["structure"]["outro"]["start_pct"] * total
    anim_end = mo["animated_window_pct"] * total
    tpl = {t["name"]: t for t in mo["animated_templates"]}
    order = ["push_small", "ease_back", "slow_push", "pull_back", "punch_ease"]
    levels, sizes = mo["punch_levels"], mo["punch_block_shots"]
    k_anim = lvl_i = blk_i = 0
    block_left, on_punch = 0, False
    prev_end = 1.0
    in_broll = lambda t: any(b["t0"] - 0.05 <= t < b["t1"] for b in broll)
    hook_heroes = [h for h in first_heroes if h < hook_end]
    hook_push_t = hook_heroes[-1] if hook_heroes else min(hook_end - 0.6, 2.13)
    for sh in shots:
        sh["zoom"], sh["anim"] = 1.0, None
        if sh["t0"] < hook_end:
            if sh["t0"] <= hook_push_t < sh["t1"]:
                dur = min(tpl["push_fast"]["dur_s"], sh["t1"] - hook_push_t)
                sh["anim"] = dict(tpl["push_fast"], t=round(hook_push_t, 4), dur_s=round(dur, 3))
            prev_end = _end_scale(sh)
            continue
        if sh["t0"] >= outro_t:
            prev_end = 1.0
            continue
        if in_broll((sh["t0"] + sh["t1"]) / 2):
            continue
        if sh["t0"] < anim_end:
            for k in range(len(order)):
                t = tpl[order[(k_anim + k) % len(order)]]
                if sh["kind"] != "punch" or abs(t["from"] - prev_end) >= 0.03:
                    k_anim += k + 1
                    break
            dur = min(t["dur_s"], sh["t1"] - sh["t0"])
            if t["at"] == "first_hero":
                at = next((h for h in first_heroes if sh["t0"] <= h < sh["t1"]), sh["t0"] + 0.25 * (sh["t1"] - sh["t0"]))
                at = min(at, sh["t1"] - dur)
            elif t["at"] == "shot_end_minus":
                at = max(sh["t0"], sh["t1"] - dur)
            else:
                at = sh["t0"]
            sh["zoom"] = t["from"]
            sh["anim"] = dict(t, t=round(at, 4), dur_s=round(dur, 3))
            prev_end = t["to"]
            continue
        if block_left <= 0 or sh["kind"] == "punch":  # new block; a punch split must change the zoom
            on_punch = not on_punch
            lvl_i += 1 if on_punch else 0
            block_left = sizes[blk_i % len(sizes)]
            blk_i += 1
        z = levels[(lvl_i - 1) % len(levels)] if on_punch else 1.0
        if sh["kind"] == "punch" and abs(z - prev_end) < 0.01:
            on_punch = not on_punch
            lvl_i += 1 if on_punch else 0
            z = levels[(lvl_i - 1) % len(levels)] if on_punch else 1.0
        sh["zoom"] = z
        prev_end = z
        block_left -= 1
    return shots


def place_events(total, positions, count_per_70, bounds, avoid, min_t, max_t):
    n = int(round(total * count_per_70 / 70.0))
    if n <= 0:
        return []
    pos = list(positions)
    while len(pos) < n:
        pos.append(min(0.95, pos[-1] + (1 - pos[-1]) / 2))
    picks = []
    for p in pos[:n]:
        want = p * total
        cands = [b for b in bounds if min_t <= b <= max_t and not any(a - 0.6 <= b <= z + 0.6 for a, z in avoid)
                 and all(abs(b - q) > 3.0 for q in picks)]
        if cands:
            picks.append(min(cands, key=lambda b: abs(b - want)))
    return sorted(picks)


# ------------------------------------------------------------------ captions
def hero_score(w, lang, used, acr):
    raw = w["w"]
    n = norm_word(raw)
    if not n:
        return -9
    neg = n in set(norm_word(x) for x in STYLE["languages"]["negations"].get(lang, []))
    if n in stopwords(lang) and not (NUM_RE.search(n) or neg):
        return -9
    minlen = 4 if script_of(raw) == "latin" else 3
    if visual_len(n) < minlen and not NUM_RE.search(n) and not acr and not neg:
        return -9
    s = 0.15 * min(visual_len(n), 12)
    s += 1.0 if SENT_END.search(raw) else (0.4 if CLAUSE_END.search(raw) else 0)
    s += 0.8 if raw.endswith("?") else 0
    s += 1.0 if NUM_RE.search(n) else 0
    s += 1.5 if acr else 0
    s += 0.8 if neg else 0
    if w.get("p") is not None and w["p"] < 0.5:
        s -= 0.6
    if n in used:
        s -= 1.0
    return s


def is_acronym(raw, lang):
    core = re.sub(r"[^\w]", "", raw)
    return has_upper(lang) and 2 <= len(core) <= 5 and core.isupper() and core.isalpha()


def build_captions(ow, total, style, lang, hook_end, broll):
    """ow: output-time words [{w,t,e,src,p}]. Returns caption groups with hero flags."""
    cap = style["captions"]
    hero_cfg = cap["hero"]
    outro_t = style["structure"]["outro"]["start_pct"] * total
    acr = [is_acronym(w["w"], lang) for w in ow]
    on_broll = lambda t: any(b["t0"] <= t < b["t1"] for b in broll)
    # ---- pick hero words
    heroes = {}
    used = set()
    win = total / 3.0
    scale = win / hero_cfg["per_third_window_s"]
    for third in range(3):
        n_target = int(round(hero_cfg["typographic_per_third"][third] * scale))
        if n_target <= 0:
            continue
        lo, hi = third * win, min((third + 1) * win, outro_t)
        cand = [(hero_score(w, lang, used, acr[i]) - (hero_cfg.get("broll_penalty", 0) if on_broll(w["t"]) else 0), i)
                for i, w in enumerate(ow) if max(lo, hook_end) <= w["t"] < hi]  # hook heroes are chosen separately
        cand.sort(reverse=True)
        picked = []
        for sc, i in cand:
            if sc < 0.9 or len(picked) >= n_target:
                continue
            if any(abs(ow[i]["t"] - ow[j]["t"]) < hero_cfg["min_spacing_s"][third] for j in picked + list(heroes)):
                continue
            picked.append(i)
            used.add(norm_word(ow[i]["w"]))
        for i in picked:
            heroes[i] = None
    # ---- hook: question hero on top + caps hero at the bottom
    hook_idx = [i for i, w in enumerate(ow) if w["t"] < hook_end]
    top_group = None
    if hook_idx:
        q_end = next((i for i in hook_idx if ow[i]["w"].endswith("?")), None)
        if q_end is None:
            q_end = next((i for i in hook_idx if SENT_END.search(ow[i]["w"])), hook_idx[-1])
        for i in list(heroes):
            if i <= q_end:
                del heroes[i]
        indic = script_of(ow[q_end]["w"]) != "latin"
        n_top = hero_cfg.get("hook_top_words", {}).get("indic" if indic else "latin", 3)
        if q_end - hook_idx[0] >= 3:
            top_start = max(hook_idx[0] + 2, q_end - (n_top - 1))
            top_group = (top_start, q_end)
            # hero = best content word of the question clause (Hindi/Malayalam end on an auxiliary verb)
            tops = [(hero_score(dict(ow[k], w=norm_word(ow[k]["w"])), lang, set(), acr[k]) + (0.05 if k == q_end else 0), k)
                    for k in range(top_start, q_end + 1)]  # punctuation-free: the content word wins, not the auxiliary
            top_hero = max(tops)[1] if max(tops)[0] > -9 else q_end
            heroes[top_hero] = "serif_white" if ow[q_end]["w"].endswith("?") else "serif_gold"
            negs = set(norm_word(x) for x in STYLE["languages"]["negations"].get(lang, []))
            bottom = [(hero_score(ow[i], lang, set(), False) + 0.1 * visual_len(norm_word(ow[i]["w"])), i)
                      for i in range(hook_idx[0], top_start) if not acr[i] and norm_word(ow[i]["w"]) not in negs]
            bottom = [x for x in bottom if x[0] > -9]
            if bottom:
                heroes[max(bottom)[1]] = "caps_gold" if has_upper(lang) else "serif_gold"
        else:
            sc = [(hero_score(ow[i], lang, set(), acr[i]), i) for i in range(hook_idx[0], q_end + 1)]
            heroes[max(sc)[1]] = "serif_white" if ow[q_end]["w"].endswith("?") else "serif_gold"
    # ---- pivot: first "but ..." sentence after the hook goes on top with a "1" marker (reference 7.6 s)
    pv = hero_cfg.get("pivot_top")
    pivot_group = None
    if pv:
        pw = set(norm_word(x) for x in pv["words"].get(lang, []))
        for i, w in enumerate(ow):
            if w["t"] < hook_end or w["t"] > pv["window_pct"] * total:
                continue
            if norm_word(w["w"]) in pw and (i == 0 or SENT_END.search(ow[i - 1]["w"]) or CLAUSE_END.search(ow[i - 1]["w"])):
                span = list(range(i, min(len(ow), i + pv["max_words"])))
                for k in span[1:]:
                    if CLAUSE_END.search(ow[k - 1]["w"]):
                        span = span[:span.index(k)]
                        break
                cands = [k for k in span[1:] if norm_word(ow[k]["w"]) not in stopwords(lang)]
                if cands:
                    h = cands[-1]
                    for k in list(heroes):
                        if span[0] <= k <= span[-1] + 1:
                            del heroes[k]
                    heroes[h] = pv["style"]
                    pivot_group = (span[0], h)
                break

    # ---- styles for remaining heroes
    gold_turn = True
    seen_acr = set()
    in_broll = lambda t: any(b["t0"] <= t < b["t1"] for b in broll)
    for i in sorted(heroes):
        if heroes[i]:
            continue
        w = ow[i]
        if acr[i]:
            k = norm_word(w["w"])
            heroes[i] = "caps_white" if k not in seen_acr else "caps_gold"
            seen_acr.add(k)
        elif w["w"].endswith("?") or in_broll(w["t"]):
            heroes[i] = "serif_white"
        elif SENT_END.search(w["w"]):
            heroes[i] = "serif_gold"
        else:
            heroes[i] = "serif_gold" if gold_turn else "serif_white"
            gold_turn = not gold_turn
    # ---- short hero words absorb up to 2 short preceding words of the same clause ("é que não.")
    pms = hero_cfg.get("phrase_merge", {})
    for i in sorted(list(heroes)):
        pm = pms.get("latin" if script_of(ow[i]["w"]) == "latin" else "indic", pms)
        if visual_len(norm_word(ow[i]["w"])) > pm.get("max_hero_len", 4) or acr[i]:
            continue
        j = i
        while j - 1 >= 0 and i - (j - 1) <= pm.get("max_prev_words", 2) and (j - 1) not in heroes \
                and visual_len(norm_word(ow[j - 1]["w"])) <= pm.get("max_prev_len", 3) and not CLAUSE_END.search(ow[j - 1]["w"]):
            j -= 1
        for k in range(j, i):
            heroes[k] = heroes[i]

    # ---- group words
    groups, i, N = [], 0, len(ow)
    mx, gap_br = cap["max_words_per_group"], cap["break_on_gap_s"]
    while i < N:
        if top_group and i == top_group[0]:
            groups.append({"idx": list(range(top_group[0], top_group[1] + 1)), "pos": "top"})
            i = top_group[1] + 1
            continue
        if pivot_group and i == pivot_group[0]:
            groups.append({"idx": list(range(pivot_group[0], pivot_group[1] + 1)), "pos": "top", "marker": pv["marker"]})
            i = pivot_group[1] + 1
            continue
        nxt_hero = next((h for h in sorted(heroes) if h >= i), None)
        if nxt_hero is not None and nxt_hero - i <= hero_cfg["max_lead_words"] and \
                not any(CLAUSE_END.search(ow[k]["w"]) for k in range(i, nxt_hero)) and \
                not (top_group and i < top_group[0] <= nxt_hero):
            end = nxt_hero
            while end + 1 < N and heroes.get(end + 1) and heroes.get(end + 1) == heroes[nxt_hero]:  # hero phrase
                end += 1
            hero_end = end
            while end + 1 < N and end - hero_end < hero_cfg["max_trail_words"] and not CLAUSE_END.search(ow[end]["w"]) \
                    and ow[end + 1]["t"] - ow[end]["e"] < gap_br and (end + 1) not in heroes \
                    and not (top_group and end + 1 == top_group[0]) and not (pivot_group and end + 1 == pivot_group[0]):
                end += 1
            groups.append({"idx": list(range(i, end + 1)), "pos": "bottom"})
            i = end + 1
            continue
        g = [i]
        max_chars, max_span = cap.get("max_chars_per_group", 16), cap.get("max_group_span_s", 0.9)
        while len(g) < mx and g[-1] + 1 < N and not CLAUSE_END.search(ow[g[-1]]["w"]) \
                and sum(visual_len(ow[k]["w"]) for k in g + [g[-1] + 1]) + len(g) <= max_chars \
                and ow[g[-1] + 1]["e"] - ow[g[0]]["t"] <= max_span \
                and ow[g[-1] + 1]["t"] - ow[g[-1]]["e"] < gap_br and (g[-1] + 1) not in heroes \
                and not (top_group and g[-1] + 1 == top_group[0]) and not (pivot_group and g[-1] + 1 == pivot_group[0]) \
                and not (nxt_hero is not None and g[-1] + 1 >= nxt_hero - hero_cfg["max_lead_words"] + 0
                         and nxt_hero - (g[-1] + 1) <= hero_cfg["max_lead_words"]
                         and not any(CLAUSE_END.search(ow[k]["w"]) for k in range(g[-1] + 1, nxt_hero))):
            g.append(g[-1] + 1)
        groups.append({"idx": g, "pos": "bottom"})
        i = g[-1] + 1
    # ---- timing + serialise
    out = []
    for k, g in enumerate(groups):
        ws = [ow[j] for j in g["idx"]]
        t0 = ws[0]["t"]
        nxt = groups[k + 1] if k + 1 < len(groups) else None
        t1 = ow[nxt["idx"][0]]["t"] if nxt else ws[-1]["e"] + 0.5
        if t1 - ws[-1]["e"] > 1.0:
            t1 = ws[-1]["e"] + 0.5
        items = []
        for j in g["idx"]:
            items.append({"w": ow[j]["w"], "t": round(ow[j]["t"], 3), "hero": heroes.get(j)})
        out.append({"t0": round(t0, 3), "t1": round(min(t1, total), 3), "pos": g["pos"], "words": items})
        if g.get("marker"):
            out[-1]["marker"] = g["marker"]
    # a non-hook top group (pivot) stays up while the rest of its sentence plays underneath (reference 7.6 s)
    for k, c in enumerate(out[:-1]):
        if c["pos"] == "top" and c["t0"] >= hook_end and not SENT_END.search(c["words"][-1]["w"]):
            nxt = out[k + 1]
            if nxt["pos"] == "bottom":
                c["t1"] = nxt["t1"]
    # first hook word lowercase (measured: "quem nunca")
    if out and has_upper(lang) and out[0]["t0"] < hook_end and not is_acronym(out[0]["words"][0]["w"], lang):
        w0 = out[0]["words"][0]
        w0["w"] = w0["w"][:1].lower() + w0["w"][1:]
    # the hook's bottom group stays up until the top question is fully read
    if top_group:
        tg = next((c for c in out if c["pos"] == "top"), None)
        for c in out:
            if c["pos"] == "bottom" and tg and c["t1"] <= tg["t0"] + 1e-3 and c["t0"] < hook_end:
                c["t1"] = tg["t1"]
    return out


def annotate(captions, lang, style):
    neg = set(norm_word(x) for x in style["languages"]["negations"].get(lang, []))
    ones = set(norm_word(x) for x in style["languages"]["number_one"].get(lang, []))
    did_circle = False
    did_marker = any(c.get("marker") for c in captions)
    flat = [(c, w) for c in captions for w in c["words"]]
    hook_end = captions[0]["t1"] if captions and captions[0]["pos"] == "bottom" else 0
    for k, (c, w) in enumerate(flat):
        if not w["hero"] or c["t0"] < hook_end - 1e-3:  # the circle marks the answer, not the hook
            continue
        if not did_circle and norm_word(w["w"]) in neg | {norm_word(x) for x in ["não", "no", "not"]}:
            w["annot"] = "red_circle"
            did_circle = True
        prev = [norm_word(x[1]["w"]) for x in flat[max(0, k - 4):k]]
        if not did_marker and any(p in ones for p in prev) and c["t0"] < captions[-1]["t1"] / 3:
            c["marker"] = "1"
            did_marker = True
    # negation inside a hero group (e.g. "é que não.") also gets the circle on the negation word
    if not did_circle:
        for c in captions:
            if c["t0"] >= hook_end - 1e-3 and any(w["hero"] for w in c["words"]):
                for w in c["words"]:
                    if norm_word(w["w"]) in neg:
                        w["annot"] = "red_circle"
                        return captions
    return captions


def detect_acronym_cards(ow, lang):
    cards = []
    for i, w in enumerate(ow):
        if not is_acronym(w["w"], lang):
            continue
        letters = re.sub(r"[^\w]", "", w["w"]).upper()
        for j in range(i + 1, min(len(ow), i + 14)):
            seq = [x for x in ow[j:j + 8] if norm_word(x["w"]) not in stopwords(lang)]
            if len(seq) >= len(letters) and all(norm_word(seq[k]["w"])[:1].upper() == letters[k] for k in range(len(letters))):
                words = [s["w"].strip(",.") for s in seq[:len(letters)]]
                cards.append({"type": "acronym", "t0": round(seq[0]["t"] - 0.15, 3), "t1": round(seq[len(letters) - 1]["e"] + 1.2, 3),
                              "letters": list(letters), "words": words,
                              "reveal_t": [round(s["t"], 3) for s in seq[:len(letters)]]})
                break
    return cards


def query_from(words, lang):
    sw = stopwords(lang)
    c = [norm_word(w["w"]) for w in words if norm_word(w["w"]) not in sw and visual_len(norm_word(w["w"])) > 3]
    return " ".join(c[:3])


STYLE = None


def plan(work, target_length=None, overrides_path=None):
    global STYLE
    style = STYLE = load_style()
    fps = style["format"]["fps"]
    srcs = {s["id"]: s for s in load_json(os.path.join(work, "sources.json"))["sources"]}
    tr = load_json(os.path.join(work, "transcript.json"))
    lang = (tr.get("language") or "en")[:2]
    ov = load_json(overrides_path) if overrides_path and os.path.exists(overrides_path) else {}
    wavs = {}
    for sid, s in srcs.items():
        y, sr = sf.read(s["wav16"], dtype="float32")
        wavs[sid] = (y if y.ndim == 1 else y.mean(1), sr)

    words = clean_words(tr, style, lang, ov)
    if not words:
        raise SystemExit("no speech found")
    sents = sentences(words)

    # ---- hook (step c)
    emax = max(rms_energy(wavs, s[0]["src"], s[0]["s"], s[-1]["e"]) for s in sents)
    hook_ov = ov.get("hook", "auto")
    hook_info = {"mode": "natural"}
    if isinstance(hook_ov, dict):
        hs = [s for s in sents if s[0]["src"] == hook_ov["src"] and s[0]["s"] >= hook_ov["from"] - 0.05 and s[-1]["e"] <= hook_ov["to"] + 0.05]
        if hs:
            for s in hs:
                sents.remove(s)
            sents = hs + sents
            hook_info = {"mode": "override"}
    elif hook_ov == "auto" and len(sents) > 2:
        scores = [score_hook(s, i, len(sents), wavs, lang, emax) for i, s in enumerate(sents)]
        best = int(np.argmax(scores))
        if best != 0 and scores[best] > max(scores[0], 0.1) * style["structure"]["hook"]["cold_open_if_better_by"]:
            sents.insert(0, sents.pop(best))
            hook_info = {"mode": "cold_open", "moved_from_sentence": best}
        hook_info["scores"] = [round(x, 2) for x in scores[:40]]
    hook_text = " ".join(w["w"] for w in sents[0])

    # ---- target length: drop lowest-value sentences (never the hook or the last 2 = CTA)
    if target_length:
        def dur(ss):
            return sum(s[-1]["e"] - s[0]["s"] + 0.2 for s in ss)
        removable = list(range(1, max(1, len(sents) - 2)))
        val = {i: score_hook(sents[i], i, len(sents), wavs, lang, emax) + 0.1 * len(sents[i]) for i in removable}
        dropped = []
        while dur(sents) > target_length * 1.02 and removable:
            i = min(removable, key=lambda k: val[k])
            removable.remove(i)
            dropped.append(" ".join(w["w"] for w in sents[i]))
            sents[i] = []
        sents = [s for s in sents if s]
        hook_info["dropped_for_length"] = dropped

    # ---- timeline (step b)
    segs, total = build_segments(sents, style, fps)
    ow = []
    for s in segs:
        for w in s["words"]:
            ow.append({"w": w["w"], "t": round(to_out_time(s, w["s"]), 3), "e": round(to_out_time(s, w["e"]), 3),
                       "src": w["src"], "p": w.get("p")})
    hook_end = max(style["structure"]["hook"]["duration_s"], ow[min(len(ow) - 1, len(sents[0]) - 1)]["e"]) \
        if len(sents[0]) else style["structure"]["hook"]["duration_s"]
    hook_end = min(hook_end, style["structure"]["hook"]["duration_s"] + 1.5)

    # ---- B-roll slots
    bc = style["broll"]
    word_starts = [w["t"] for w in ow]
    word_ends = [w["e"] for w in ow]
    outro_t = style["structure"]["outro"]["start_pct"] * total
    broll = []
    n_b = int(round(total * bc["per_70s"] / 70.0))
    pos = list(bc["positions_pct"])
    while len(pos) < n_b:
        pos.append(min(0.85, pos[-1] + 0.3))
    for k in range(n_b):
        want = pos[k] * total
        L = bc["lengths_s"][k % len(bc["lengths_s"])]
        starts = [t for t in word_starts if hook_end + 0.5 <= t <= outro_t - L * 0.6
                  and all(not (b["t0"] - 2 <= t <= b["t1"] + 2) for b in broll)]
        if not starts:
            continue
        # the reference lays B-roll over clusters of jump cuts (hides them): prefer windows covering more cuts
        cut_ts = [s["t0"] for s in segs[1:]]
        near = [t for t in starts if abs(t - want) <= bc.get("search_window_pct", 0.08) * total] or starts
        covered = lambda t: sum(1 for c in cut_ts if t + 0.3 < c < t + L - 0.3)
        a = max(near, key=lambda t: (covered(t) if covered(t) >= 2 else 0) - abs(t - want) / max(total, 1) * 5)
        ends = [e for e in word_ends if e >= a + L * 0.8]
        b = min(ends, key=lambda e: abs(e - (a + L))) if ends else min(a + L, total)
        b = min(b + 0.1, outro_t)
        covered = [w for w in ow if a <= w["t"] < b]
        broll.append({"t0": round(a, 3), "t1": round(b, 3), "planned_t1": round(b, 3), "treatment": bc["treatments"][k % len(bc["treatments"])],
                      "query": query_from(covered, lang), "spoken": " ".join(w["w"] for w in covered),
                      "path": None, "src_in": 0.0})

    # ---- captions + heroes (step e)
    captions = build_captions(ow, total, style, lang, hook_end, broll)
    captions = annotate(captions, lang, style)
    # B-roll never covers the pivot (top) line or the circled answer - in the reference they play on the speaker
    keep_clear = [(c["t0"], c["t1"]) for c in captions if c["t0"] >= hook_end - 0.01 and
                  (c["pos"] == "top" or any(w.get("annot") for w in c["words"]))]
    moved = False
    for b in broll:
        for a, z in keep_clear:
            if b["t0"] < z and a < b["t1"]:
                L = b["t1"] - b["t0"]
                nxt = [t for t in word_starts if t >= z - 0.01]
                if nxt and nxt[0] + L * 0.6 <= outro_t:
                    b["t0"] = round(nxt[0], 3)
                    ends = [e for e in word_ends if e >= b["t0"] + L * 0.8]
                    b["t1"] = b["planned_t1"] = round(min((min(ends, key=lambda e: abs(e - (b["t0"] + L))) if ends else b["t0"] + L) + 0.1, outro_t), 3)
                    moved = True
    if moved:
        captions = annotate(build_captions(ow, total, style, lang, hook_end, broll), lang, style)
    first_heroes = [w["t"] for c in captions for w in c["words"] if w["hero"]]

    # ---- shots, zooms, transitions (step d)
    shots = make_shots(segs, total, style, hook_end)
    cut_bounds = [s["t0"] for s in shots if s["kind"] in ("cut", "punch")]
    sent_starts = set(round(to_out_time(s, s["words"][0]["s"]), 2) for s in segs)
    tr_cfg = style["transitions"]["zoom_blur"]
    avoid = [(b["t0"], b["t1"]) for b in broll]
    zt = place_events(total, tr_cfg["positions_pct"], tr_cfg["count_per_70s"], cut_bounds, avoid,
                      style["motion"]["animated_window_pct"] * total, outro_t - 0.5)
    shots = assign_zooms(shots, total, style, hook_end, first_heroes, broll)
    for sh in shots:
        if any(abs(sh["t0"] - z) < 1e-3 for z in zt):
            sh["enter"] = "zoom_blur"
        elif sh["kind"] == "start":
            sh["enter"] = "intro"
        else:
            sh["enter"] = "cut"

    # ---- card suggestions (Claude fills content); acronym cards are automatic
    cards = detect_acronym_cards(ow, lang)
    n_cards = max(1, int(round(total * style["graphics"]["cards_per_70s"] / 70.0)))
    sugg = {"target_count": n_cards, "slots": []}
    for name, p in style["graphics"]["card_positions_pct"].items():
        m = style["graphics"]["card_types"][name]["measured_s"]
        sugg["slots"].append({"type": name, "around_t": round(p * total, 2), "typical_len_s": round(m[1] - m[0], 1)})

    plan_obj = {
        "fps": fps, "width": style["format"]["width"], "height": style["format"]["height"],
        "duration": round(total, 3), "language": lang, "hook": dict(hook_info, text=hook_text, end=round(hook_end, 3)),
        "sources": {k: {"norm": v["norm"], "face": v.get("face")} for k, v in srcs.items()},
        "segments": [{k: s[k] for k in ("src", "in", "out", "t0", "t1")} for s in segs],
        "shots": shots,
        "broll": broll,
        "captions": captions,
        "cards": cards,
        "card_suggestions": sugg,
        "audio": {"music": None},
        "words": ow,
    }
    save_json(plan_obj, os.path.join(work, "plan.json"))
    nh = sum(1 for c in captions for w in c["words"] if w["hero"])
    log(f"plan: {total:.1f}s, {len(segs)} segments, {len(shots)} shots, {len(zt)} zoom transitions, "
        f"{len(broll)} B-roll slots, {len(captions)} caption groups, {nh} hero words, hook={hook_info['mode']}")
    return plan_obj


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--target-length", type=float, default=None)
    ap.add_argument("--overrides", default=None)
    a = ap.parse_args()
    plan(a.work, a.target_length, a.overrides or os.path.join(a.work, "overrides.json"))
