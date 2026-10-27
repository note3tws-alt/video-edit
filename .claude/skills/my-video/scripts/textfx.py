"""Text rendering (Pillow + raqm shaping for Devanagari/Malayalam) and per-frame caption compositing."""
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from common import FONTS, ease, hex_rgb, lerp, norm_word, script_of

_font_cache = {}


def get_font(style, family, script, size):
    fam = style["captions"]["fonts"][script][family]
    key = (fam[0], fam[1], int(size))
    if key not in _font_cache:
        f = ImageFont.truetype(os.path.join(FONTS, fam[0]), int(size), layout_engine=ImageFont.Layout.RAQM)
        if fam[1]:
            try:
                f.set_variation_by_name(fam[1])
            except Exception:
                pass
        _font_cache[key] = f
    return _font_cache[key]


def script_cfg(style, script):
    return style["captions"]["fonts"][script]


def text_width(style, family, text, size):
    sc = script_of(text)
    f = get_font(style, family, sc, size * script_cfg(style, sc).get("size_scale", 1.0))
    l, t, r, b = f.getbbox(text, anchor="ls")
    return r - l


def render_text(style, family, text, size, fill=("#FFFFFF", "#FFFFFF"), glow=None, glow_opacity=0.0, glow_blur=0,
                shadow=None, stops=None):
    """Render text to a straight-alpha RGBA numpy array.
    Returns dict(img, ox, oy, w, asc, desc): (ox, oy) = pixel of the text origin (left, baseline) inside img."""
    sc = script_of(text)
    cfg = script_cfg(style, sc)
    size = size * cfg.get("size_scale", 1.0)
    f = get_font(style, family, sc, size)
    bold = cfg.get("faux_bold_px", 0) if family == "serif_italic" else 0
    l, t, r, b = f.getbbox(text, anchor="ls", stroke_width=bold)
    pad = int(max(glow_blur * 3, (shadow or {}).get("blur_px", 0) * 3 + 6, 12) + size * 0.15)
    W, H = int(r - l + 2 * pad), int(b - t + 2 * pad)
    ox, oy = pad - l, pad - t
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).text((ox, oy), text, font=f, fill=255, anchor="ls", stroke_width=bold, stroke_fill=255)
    shear = cfg.get("synthetic_italic_deg", 0) if family == "serif_italic" else 0
    if shear:
        k = math.tan(math.radians(shear))
        mask = mask.transform(mask.size, Image.AFFINE, (1, k, -k * oy, 0, 1, 0), resample=Image.BICUBIC)
    m = np.asarray(mask, dtype=np.float32) / 255.0
    # vertical gradient (2 or more stops) across the glyph box
    cols = np.array([hex_rgb(c) for c in fill], np.float32)
    st = np.array(stops if stops else np.linspace(0, 1, len(fill)), np.float32)
    ys = np.clip((np.arange(H) - (oy + t)) / max(1.0, (b - t)), 0, 1)
    col = np.stack([np.interp(ys, st, cols[:, c]) for c in range(3)], -1)[:, None, :]
    col = np.broadcast_to(col, (H, W, 3))
    out_rgb = np.zeros((H, W, 3), np.float32)
    out_a = np.zeros((H, W), np.float32)

    def over(rgb, a):
        nonlocal out_rgb, out_a
        na = a + out_a * (1 - a)
        out_rgb = np.where(na[..., None] > 1e-6, (rgb * a[..., None] + out_rgb * out_a[..., None] * (1 - a[..., None])) /
                           np.maximum(na[..., None], 1e-6), 0)
        out_a = na

    if shadow:
        sm = cv2.GaussianBlur(m, (0, 0), max(0.1, shadow["blur_px"] / 2))
        sm = np.roll(sm, int(shadow.get("dy_px", 0)), axis=0) * shadow["opacity"]
        over(np.broadcast_to(np.array(hex_rgb(shadow["color"]), np.float32), (H, W, 3)), sm)
    if glow and glow_opacity > 0:
        gm = np.clip(cv2.GaussianBlur(m, (0, 0), max(0.1, glow_blur / 2)) * 1.6, 0, 1) * glow_opacity
        over(np.broadcast_to(np.array(hex_rgb(glow), np.float32), (H, W, 3)), gm)
    over(col, m)
    img = np.ascontiguousarray(np.dstack([out_rgb, out_a * 255]).clip(0, 255).astype(np.uint8))
    asc = -t
    return {"img": img, "ox": ox, "oy": oy, "w": r - l, "asc": asc, "desc": b, "size": size}


def overlay(frame, rgba, x, y, alpha=1.0):
    """Alpha-composite rgba (straight alpha, RGB order) onto BGR frame at integer top-left (x, y)."""
    if alpha <= 0.003:
        return
    H, W = frame.shape[:2]
    h, w = rgba.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    src = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    a = src[..., 3:4].astype(np.float32) * (alpha / 255.0)
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    rgb = src[..., 2::-1].astype(np.float32)  # RGB -> BGR
    frame[y0:y1, x0:x1] = (rgb * a + roi * (1 - a)).astype(np.uint8)


def blur_rgba(rgba, sigma):
    if sigma < 0.4:
        return rgba
    return cv2.GaussianBlur(rgba, (0, 0), sigma)


def scale_rgba(rgba, s):
    if abs(s - 1) < 1e-3:
        return rgba
    h, w = rgba.shape[:2]
    return cv2.resize(rgba, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_LINEAR)


# ------------------------------------------------------------------ caption layout
class CaptionRenderer:
    def __init__(self, style, plan):
        self.st = style
        self.cap = style["captions"]
        self.W, self.H, self.fps = plan["width"], plan["height"], plan["fps"]
        self.broll = plan.get("broll", [])
        self.cache = {}
        self.layout_log = []

    # -- word sprites
    def _base(self, text):
        b = self.cap["base"]
        return render_text(self.st, "base", text, b["size_px"], shadow=b["shadow"])

    def _hero(self, text, style_name, lang_upper, top=False):
        """Measured sizes: serif 128 px (hook-top question 163 px); caps up to 170 px, shrunk to 0.67 W."""
        hc = self.cap["hero"]
        hs = hc["styles"][style_name]
        fam = hs["family"]
        if fam == "caps_heavy" and lang_upper:
            text = text.upper()
        if fam == "caps_heavy":
            size, fit = hc["size_px"]["caps_max"], hs.get("fit_width", 0.67) * self.W
        else:
            size, fit = hc["size_px"]["serif_hook_top" if top else "serif"], hc["max_width"] * self.W
        w = text_width(self.st, fam, text, size)
        if w > fit:
            size = size * fit / w
        return render_text(self.st, fam, text, size, fill=hs["fill"], glow=hs["glow"], glow_opacity=hs["glow_opacity"],
                           glow_blur=hs["glow_blur_px"], stops=hs.get("fill_stops"))

    def layout(self, gi, group, lang_upper):
        if gi in self.cache:
            return self.cache[gi]
        W, H = self.W, self.H
        words = group["words"]
        hero_i = next((i for i, w in enumerate(words) if w.get("hero")), None)
        sprites = []
        space = self.cap["base"]["size_px"] * self.cap["base"]["word_spacing_em"]
        on_broll = next((b for b in self.broll if b["t0"] <= group["t0"] < b["t1"] and b.get("path")), None)

        def line(idxs, cy_or_baseline, xshift=0.0, baseline=None):
            sp = [self._base(words[i]["w"]) for i in idxs]
            total = sum(s["w"] for s in sp) + space * (len(sp) - 1)
            maxw = W * (1 - 2 * self.st["format"]["safe_zone"]["side_margin_pct"] / 100)
            k = min(1.0, maxw / max(total, 1))
            if k < 1.0:
                sp = [render_text(self.st, "base", words[i]["w"], self.cap["base"]["size_px"] * k,
                                  shadow=self.cap["base"]["shadow"]) for i in idxs]
                total = sum(s["w"] for s in sp) + space * k * (len(sp) - 1)
            x = W / 2 - total / 2 + xshift * W
            cap_h = self.cap["base"]["cap_height_px"] * k
            base_y = baseline if baseline is not None else cy_or_baseline + cap_h / 2
            for i, s in zip(idxs, sp):
                sprites.append({"i": i, "s": s, "x": int(round(x - s["ox"])), "y": int(round(base_y - s["oy"])),
                                "hero": False, "bbox": (x, base_y - s["asc"], x + s["w"], base_y + s["desc"])})
                x += s["w"] + space * k
            return base_y, cap_h

        hero_idx = [i for i, w in enumerate(words) if w.get("hero")]
        hero_j = hero_idx[-1] if hero_idx else None  # consecutive hero words form one hero phrase ("é que não.")
        if hero_i is None:
            ybr = self.cap["y_center_broll"]
            yb = (ybr.get(on_broll["treatment"], 0.55) if isinstance(ybr, dict) else ybr) if on_broll else None
            cy = (yb if on_broll else self.cap["y_center"]) * H
            line(list(range(len(words))), cy)
        else:
            phrase = " ".join(words[i]["w"] for i in range(hero_i, hero_j + 1))
            hs = self._hero(phrase, words[hero_i]["hero"], lang_upper, top=group.get("pos") == "top")
            cy = (self.cap["hero"]["y_center_hook_top"] if group.get("pos") == "top" else self.cap["hero"]["y_center"]) * H
            cap_h = hs["asc"] * 0.72
            base = cy + cap_h / 2
            x = W / 2 - hs["w"] / 2
            hb = (x, base - hs["asc"], x + hs["w"], base + hs["desc"])
            sub = {}
            if hero_j > hero_i:  # per-word boxes inside the phrase (for the red circle)
                k = hs["w"] / max(1, text_width(self.st, self.cap["hero"]["styles"][words[hero_i]["hero"]]["family"], phrase, hs["size"]))
                for i in range(hero_i, hero_j + 1):
                    pre = " ".join(words[q]["w"] for q in range(hero_i, i)) + (" " if i > hero_i else "")
                    fam = self.cap["hero"]["styles"][words[hero_i]["hero"]]["family"]
                    x0 = x + text_width(self.st, fam, pre, hs["size"]) * k if pre else x
                    sub[i] = (x0, hb[1], x0 + text_width(self.st, fam, words[i]["w"], hs["size"]) * k, hb[3])
            sprites.append({"i": hero_i, "s": hs, "x": int(round(x - hs["ox"])), "y": int(round(base - hs["oy"])),
                            "hero": True, "bbox": hb, "sub": sub, "span": (hero_i, hero_j)})
            lead = list(range(0, hero_i))
            trail = list(range(hero_j + 1, len(words)))
            hero_cap = hs["size"] * 0.70  # cap height of the display faces (~0.70 em)
            if lead:  # lead-in words rest on the hero's cap line, overlapping its ascenders
                line(lead, None, baseline=base - hero_cap * self.cap["hero"].get("lead_baseline_cap_frac", 0.92))
            if trail:
                tb = base + max(hs["desc"], 0) * 0.5 + self.cap["base"]["cap_height_px"] * 1.2
                line(trail, None, xshift=self.cap["hero"]["trail_shift_x"], baseline=tb)
        if group.get("marker") and sprites:  # make room for the "1" on the left of the block
            mk = self.st["graphics"]["list_marker"]
            mw = render_text(self.st, "serif_italic", group["marker"], mk["size_px"])["w"]
            left = min(sp["bbox"][0] for sp in sprites)
            dx = max(0.0, self.W * 0.04 + mw + 10 - left)  # numeral fully left of the block, 10 px gap
            for sp in sprites:
                sp["x"] += int(dx)
                b = sp["bbox"]
                sp["bbox"] = (b[0] + dx, b[1], b[2] + dx, b[3])
                sp["sub"] = {k: (v[0] + dx, v[1], v[2] + dx, v[3]) for k, v in sp.get("sub", {}).items()}
        lay = {"sprites": sprites, "hero_i": hero_i}
        self.cache[gi] = lay
        ys = [((sp["bbox"][1] + sp["bbox"][3]) / 2) / H for sp in sprites]
        self.layout_log.append({"t0": group["t0"], "pos": group.get("pos"), "hero": hero_i is not None,
                                "y_center": round(float(np.mean(ys)), 4),
                                "max_w": round(max(sp["bbox"][2] - sp["bbox"][0] for sp in sprites) / W, 4)})
        return lay

    def draw(self, frame, t, captions, lang_upper, suppressed=()):
        fps = self.fps
        wi = self.cap["word_in"]
        hp = self.cap["hero"]["pop_in"]
        out_fr = self.cap["group_out"]["frames"]
        for gi, g in enumerate(captions):
            if not (g["t0"] - 1e-6 <= t < g["t1"]) or gi in suppressed:
                continue
            lay = self.layout(gi, g, lang_upper)
            g_alpha = min(1.0, max(0.0, (g["t1"] - t) * fps / out_fr))
            for sp in lay["sprites"]:
                w = g["words"][sp["i"]]
                dt = t - w["t"]
                if dt < 0:
                    continue
                if sp["hero"]:
                    p = ease(dt * fps / hp["frames"])
                    img = scale_rgba(blur_rgba(sp["s"]["img"], hp["blur_sigma_px"] * (1 - p)), lerp(hp["scale_from"], 1.0, p))
                    dx = (sp["s"]["img"].shape[1] - img.shape[1]) // 2
                    dy = (sp["s"]["img"].shape[0] - img.shape[0]) // 2
                    overlay(frame, img, sp["x"] + dx, sp["y"] + dy, p * g_alpha)
                else:
                    p = ease(dt * fps / wi["frames"])
                    img = blur_rgba(sp["s"]["img"], wi["blur_sigma_px"] * (1 - p))
                    overlay(frame, img, sp["x"], int(sp["y"] + wi["rise_px"] * (1 - p)), p * g_alpha)
                for j in range(sp["span"][0], sp["span"][1] + 1) if sp.get("span") else [sp["i"]]:
                    if g["words"][j].get("annot") == "red_circle":
                        self._circle(frame, sp.get("sub", {}).get(j, sp["bbox"]), t - g["words"][j]["t"], g_alpha)
            if g.get("marker") and lay["hero_i"] is not None:
                self._marker(frame, lay, g, t, g_alpha)

    def _circle(self, frame, bbox, dt, alpha):
        rc = self.st["graphics"]["red_circle"]
        p = ease(dt / rc["draw_s"])
        if p <= 0:
            return
        x0, y0, x1, y1 = bbox
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rx, ry = (x1 - x0) / 2 * 1.25 + 10, (y1 - y0) / 2 * 1.35 + 8
        n = max(2, int(60 * p))
        ang = np.linspace(-0.6 * math.pi, -0.6 * math.pi + 2.15 * math.pi * p, n)
        wob = 1 + 0.04 * np.sin(ang * 3.0)
        pts = np.stack([cx + rx * wob * np.cos(ang), cy + ry * wob * np.sin(ang)], 1).astype(np.int32)
        layer = frame.copy()
        cv2.polylines(layer, [pts], False, hex_rgb(rc["color"])[::-1], rc["width_px"], cv2.LINE_AA)
        cv2.addWeighted(layer, alpha, frame, 1 - alpha, 0, frame)

    def _marker(self, frame, lay, g, t, alpha):
        mk = self.st["graphics"]["list_marker"]
        key = ("marker", g["t0"])
        if key not in self.cache:
            self.cache[key] = render_text(self.st, "serif_italic", g["marker"], mk["size_px"], shadow=self.cap["base"]["shadow"])
        s = self.cache[key]
        hero = next(sp for sp in lay["sprites"] if sp["hero"])
        left = min(sp["bbox"][0] for sp in lay["sprites"])
        x = int(max(self.W * 0.04, left - s["w"] - 10))
        base = hero["bbox"][3] - hero["s"]["desc"]  # hero baseline
        p = ease((t - g["t0"]) * self.fps / 5)
        overlay(frame, s["img"], int(x - s["ox"]), int(base - s["oy"]), p * alpha)
