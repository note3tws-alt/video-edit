"""Info cards measured from the reference: checklist (✓/✗ rows), timeline (dots), compare (two panels + ≠), acronym.

Card schema inside plan.json "cards" (times are output seconds; any item may use "at_word" instead of "t"):
  {"type": "checklist", "t0": 48.5, "t1": 55.5, "items": [{"text": "Minimum wage", "ok": true, "t": 49.8}, ...]}
  {"type": "timeline",  "t0": 26.0, "t1": 31.9, "points": [{"label": "Min wage", "value": "$1,621", "t": 26.4}, ...]}
  {"type": "compare",   "t0": 20.2, "t1": 21.4, "panels": ["text A", "text B"], "symbol": "≠"}
  {"type": "acronym",   "t0": 16.6, "t1": 20.0, "letters": ["B","P","C"], "words": ["Benefit","Payment","Continued"], "reveal_t": [...]}
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw

from common import ease, hex_rgb, lerp, norm_word, script_of
from textfx import blur_rgba, get_font, overlay, render_text, script_cfg


def _font(style, text, size):
    sc = script_of(text)
    return get_font(style, "card", sc, size * script_cfg(style, sc).get("size_scale", 1.0))


def rounded(w, h, r, fill, shadow=None, grad=None):
    """RGBA sprite of a rounded rectangle (optional horizontal gradient) with soft drop shadow."""
    pad = int(shadow["blur_px"] * 2 + abs(shadow.get("dy_px", 0))) if shadow else 0
    W, H = w + 2 * pad, h + 2 * pad
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).rounded_rectangle([pad, pad, pad + w - 1, pad + h - 1], r, fill=255)
    m = np.asarray(m, np.float32) / 255
    if grad:
        a, b = np.array(hex_rgb(grad[0]), np.float32), np.array(hex_rgb(grad[1]), np.float32)
        xs = np.clip((np.arange(W) - pad) / max(1, w), 0, 1)[None, :, None]
        col = a * (1 - xs) + b * xs
        col = np.broadcast_to(col, (H, W, 3))
    else:
        col = np.broadcast_to(np.array(hex_rgb(fill), np.float32), (H, W, 3))
    rgb = np.array(col)
    alpha = m.copy()
    if shadow:
        sm = cv2.GaussianBlur(m, (0, 0), shadow["blur_px"] / 2)
        sm = np.roll(sm, int(shadow.get("dy_px", 0)), axis=0) * shadow["opacity"]
        a_out = m + sm * (1 - m)
        rgb = (rgb * m[..., None]) / np.maximum(a_out[..., None], 1e-6)
        alpha = a_out
    return np.ascontiguousarray(np.dstack([rgb, alpha * 255]).clip(0, 255).astype(np.uint8)), pad


def draw_text_on(img, x, y, text, style, size, color, anchor="ls"):
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    d.text((x, y), text, font=_font(style, text, size), fill=hex_rgb(color) + (255,), anchor=anchor)
    return np.array(pil)  # writable copy (cv2 draws on it next)


def icon(img, kind, cx, cy, size, color):
    c = hex_rgb(color) + (255,)
    s = size / 2
    th = max(3, int(size * 0.14))
    if kind == "ok":
        pts = np.array([[cx - s, cy], [cx - s * 0.25, cy + s * 0.7], [cx + s, cy - s * 0.75]], np.int32)
        cv2.polylines(img, [pts], False, c, th, cv2.LINE_AA)
    else:
        cv2.line(img, (int(cx - s * 0.7), int(cy - s * 0.7)), (int(cx + s * 0.7), int(cy + s * 0.7)), c, th, cv2.LINE_AA)
        cv2.line(img, (int(cx - s * 0.7), int(cy + s * 0.7)), (int(cx + s * 0.7), int(cy - s * 0.7)), c, th, cv2.LINE_AA)


def wrap(style, text, size, maxw, max_lines):
    words, lines, cur = text.split(), [], ""
    f = _font(style, text, size)
    for w in words:
        nxt = (cur + " " + w).strip()
        if f.getlength(nxt) <= maxw or not cur:
            cur = nxt
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".,") + "…"
    return lines


def ripple(frame, cx, cy, dt, cfg):
    tap = cfg["tap"]
    if not (0 <= dt < tap["s"]):
        return
    p = dt / tap["s"]
    r = int(tap["radius_px"] * (0.4 + 1.2 * ease(p)))
    layer = frame.copy()
    cv2.circle(layer, (int(cx), int(cy)), r, hex_rgb(tap["ripple_color"])[::-1], -1, cv2.LINE_AA)
    a = 0.55 * (1 - p)
    cv2.addWeighted(layer, a, frame, 1 - a, 0, frame)


class CardRenderer:
    def __init__(self, style, plan):
        self.st, self.g = style, style["graphics"]
        self.W, self.H, self.fps = plan["width"], plan["height"], plan["fps"]
        self.cards = [self._resolve(c, plan.get("words", [])) for c in plan.get("cards", [])]
        self.cache = {}
        self.log = []

    def _resolve(self, card, words):
        def find(key, after):
            k = norm_word(key.split()[0])
            for w in words:
                if w["t"] >= after - 0.05 and norm_word(w["w"]) == k:
                    return w["t"]
            return None
        for lst in ("items", "points"):
            prev = card["t0"]
            for i, it in enumerate(card.get(lst, [])):
                if "t" not in it and "at_word" in it:
                    it["t"] = find(it["at_word"], prev)
                if it.get("t") is None:
                    n = len(card[lst])
                    it["t"] = card["t0"] + 0.3 + (card["t1"] - card["t0"] - 0.8) * i / max(1, n)
                prev = it["t"]
        return card

    def acronym_active(self, t):
        return any(c["type"] == "acronym" and c["t0"] <= t < c["t1"] for c in self.cards)

    def draw(self, frame, t):
        for ci, c in enumerate(self.cards):
            if c["t0"] <= t < c["t1"]:
                getattr(self, "_" + c["type"])(frame, t, ci, c)

    # -- shared enter/exit motion for white cards
    def _motion(self, t, c):
        cc = self.g["card"]
        pin = ease((t - c["t0"]) / cc["in"]["s"])
        pout = ease((t - (c["t1"] - cc["out"]["s"])) / cc["out"]["s"]) if t > c["t1"] - cc["out"]["s"] else 0.0
        dy = cc["in"]["distance_px"] * (1 - pin) + cc["out"]["distance_px"] * pout
        return pin * (1 - pout), int(dy)

    def _checklist(self, frame, t, ci, c):
        cfg = self.g["card_types"]["checklist"]
        W, H = self.W, self.H
        rw, rh, gap = int(cfg["row_width"] * W), cfg["row_h_px"], cfg["row_gap_px"]
        x0 = (W - rw) // 2
        y0 = int(self.g["card"]["y_top"] * H)
        alpha, dy = self._motion(t, c)
        for i, it in enumerate(c["items"]):
            key = (ci, i)
            if key not in self.cache:
                ok = it.get("ok", True)
                img, pad = rounded(rw, rh, 14, None, self.g["card"]["shadow"], cfg["ok_fill"] if ok else cfg["no_fill"])
                icon(img, "ok" if ok else "no", pad + 42, pad + rh // 2, cfg["icon_px"], cfg["ok_icon"] if ok else cfg["no_icon"])
                img = draw_text_on(img, pad + 84, pad + rh // 2, it["text"], self.st, cfg["text_px"], self.g["card"]["text_color"], "lm")
                self.cache[key] = (img, pad)
            img, pad = self.cache[key]
            dt = t - it["t"]
            if dt < 0:
                continue
            p = ease(dt / cfg["row_in_s"])
            x = int(x0 - pad - 80 * (1 - p))
            y = y0 + i * (rh + gap) - pad + dy
            overlay(frame, img, x, y, p * alpha)
            ripple(frame, x0 + 42, y0 + i * (rh + gap) + rh // 2 + dy, dt, self.g["card"])
        self.log.append(("checklist", t, y0 + len(c["items"]) * (rh + gap)))

    def _timeline(self, frame, t, ci, c):
        cfg, cc = self.g["card_types"]["timeline"], self.g["card"]
        W, H = self.W, self.H
        cw, ch = int(cc["width"] * W), cfg["height_px"]
        x0, y0 = (W - cw) // 2, int(cc["y_top"] * H)
        alpha, dy = self._motion(t, c)
        n = len(c["points"])
        xs = [int(cw * (0.12 + 0.76 * i / max(1, n - 1))) for i in range(n)]
        active = sum(1 for p in c["points"] if p["t"] <= t)
        key = (ci, active)
        if key not in self.cache:
            img, pad = rounded(cw, ch, cc["radius_px"], cc["bg"], cc["shadow"])
            ly = pad + 58
            lc = hex_rgb(cfg["line_color"])[::-1]
            cv2.line(img, (pad + xs[0], ly), (pad + xs[-1], ly), (230, 230, 230, 255), 6, cv2.LINE_AA)
            if active > 1:
                cv2.line(img, (pad + xs[0], ly), (pad + xs[active - 1], ly), hex_rgb(cfg["line_color"]) + (255,), 6, cv2.LINE_AA)
            for i, p in enumerate(c["points"]):
                on = i < active
                col = hex_rgb(cfg["dot_color"]) + (255,) if on else (240, 226, 200, 255)
                cv2.circle(img, (pad + xs[i], ly), 15 if on else 12, col, -1, cv2.LINE_AA)
                img = draw_text_on(img, pad + xs[i], ly + 54, p.get("label", ""), self.st, cfg["label_px"], cc["muted_color"], "ms")
                if p.get("value"):
                    img = draw_text_on(img, pad + xs[i], ly + 86, p["value"], self.st, cfg["value_px"], cfg["value_color"], "ms")
            self.cache[key] = (img, pad)
        img, pad = self.cache[key]
        overlay(frame, img, x0 - pad, y0 - pad + dy, alpha)
        if active:
            ripple(frame, x0 + xs[active - 1], y0 + 58 + dy, t - c["points"][active - 1]["t"], cc)
        self.log.append(("timeline", t, y0 + ch))

    def _compare(self, frame, t, ci, c):
        cfg, cc = self.g["card_types"]["compare"], self.g["card"]
        W, H = self.W, self.H
        cw = int(cc["width"] * W)
        if (ci, "c") not in self.cache:
            sprites, y = [], 0
            for txt in c["panels"][:2]:
                lines = wrap(self.st, txt, cfg["text_px"], cw - 2 * cc["padding_px"], cfg["max_lines"])
                lh = int(cfg["text_px"] * 1.3)
                ph = 2 * 22 + lh * len(lines)
                img, pad = rounded(cw, ph, 16, cc["bg"], cc["shadow"])
                for k, ln in enumerate(lines):
                    img = draw_text_on(img, pad + cw // 2, pad + 22 + lh * k + int(cfg["text_px"] * 1.0), ln, self.st,
                                       cfg["text_px"], cc["text_color"], "ms")
                sprites.append((img, pad, y, ph))
                y += ph + 26
            sym = render_text(self.st, "caps_heavy", c.get("symbol", cfg["symbol"]), 72,
                              fill=(cfg["symbol_color"], cfg["symbol_color"]), shadow=self.st["captions"]["base"]["shadow"])
            self.cache[(ci, "c")] = (sprites, sym)
        sprites, sym = self.cache[(ci, "c")]
        y0 = int(cc["y_top"] * H)
        x0 = (W - cw) // 2
        p = ease((t - c["t0"]) / 0.25)
        pout = ease((t - (c["t1"] - 0.2)) / 0.2) if t > c["t1"] - 0.2 else 0
        a = p * (1 - pout)
        for k, (img, pad, yy, ph) in enumerate(sprites):
            pk = ease((t - c["t0"] - 0.12 * k) / 0.25)
            s = lerp(0.92, 1.0, pk)
            im = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s))) if s < 0.999 else img
            overlay(frame, im, x0 - pad + (img.shape[1] - im.shape[1]) // 2, y0 - pad + yy + (img.shape[0] - im.shape[0]) // 2, pk * a)
        if len(sprites) == 2:
            yb = y0 + sprites[1][2] - 13
            overlay(frame, sym["img"], int(W / 2 - sym["w"] / 2 - sym["ox"]), int(yb + sym["asc"] * 0.4 - sym["oy"]), a)
        self.log.append(("compare", t, y0 + sum(s[3] + 26 for s in sprites)))

    def _acronym(self, frame, t, ci, c):
        cfg = self.g["acronym"]
        W, H = self.W, self.H
        size = cfg["size_px"]
        if (ci, "a") not in self.cache:
            rows = []
            for L, word in zip(c["letters"], c["words"]):
                rest = word[1:] if norm_word(word[:1]) == norm_word(L) else " " + word
                ls = render_text(self.st, "base", L, size, fill=(cfg["letter_color"], cfg["letter_color"]))
                rs = render_text(self.st, "base", rest, size, shadow=self.st["captions"]["base"]["shadow"])
                rows.append((ls, rs))
            self.cache[(ci, "a")] = rows
        rows = self.cache[(ci, "a")]
        lh = int(size * 0.92) + cfg["line_gap_px"]
        y_mid = 0.69 * H
        base0 = y_mid - lh * (len(rows) - 1) / 2 + size * 0.36
        x = int(cfg["x_left"] * W)
        reveal = c.get("reveal_t") or [c["t0"] + 0.35 * i for i in range(len(rows))]
        pout = ease((t - (c["t1"] - 0.2)) / 0.2) if t > c["t1"] - 0.2 else 0
        for i, (ls, rs) in enumerate(rows):
            dt = t - reveal[i]
            if dt < 0:
                continue
            p = ease(dt * self.fps / 4)
            b = base0 + i * lh
            bx0, by0 = x - 10, int(b - size * 0.80)
            bx1, by1 = x + int(ls["w"]) + 10, int(b + size * 0.10)
            layer = frame.copy()
            cv2.rectangle(layer, (bx0, by0), (bx1, by1), hex_rgb(cfg["box_color"])[::-1], cfg["box_stroke_px"], cv2.LINE_AA)
            a = p * (1 - pout)
            cv2.addWeighted(layer, a, frame, 1 - a, 0, frame)
            overlay(frame, blur_rgba(ls["img"], 6 * (1 - p)), int(x - ls["ox"]), int(b - ls["oy"] + 30 * (1 - p)), a)
            overlay(frame, blur_rgba(rs["img"], 6 * (1 - p)), int(bx1 + 6 - rs["ox"]), int(b - rs["oy"] + 30 * (1 - p)), a)
