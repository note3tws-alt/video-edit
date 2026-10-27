"""Synthesise the reference's SFX (royalty-free, deterministic) into assets/sfx/*.wav.

shimmer: tonal partials measured in the reference (4.5/6.43/7.2/9.13/9.7 kHz), 0.55 s decay  -> gold hero words
whoosh : band-passed noise sweep, 0.18 s                                                       -> zoom-blur transitions
tap    : 40 ms click                                                                           -> card item reveals
Also: generate_music_bed() -> soft procedural pad, used only when the user asks the skill to pick music
and no track exists in assets/music/.
"""
import os

import numpy as np
import soundfile as sf

from common import ASSETS, load_style

SR = 48000


def _norm(x, peak_db=-1.0):
    return x / (np.abs(x).max() + 1e-9) * 10 ** (peak_db / 20)


def shimmer(cfg):
    n = int(cfg["dur_s"] * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(7)
    y = np.zeros(n)
    for i, f in enumerate(cfg["partials_hz"]):
        det = 1 + rng.uniform(-0.002, 0.002)
        trem = 1 + 0.25 * np.sin(2 * np.pi * (9 + i * 2.3) * t)
        y += np.sin(2 * np.pi * f * det * t + rng.uniform(0, 6.28)) * trem / (1 + 0.3 * i)
    env = (1 - np.exp(-t / 0.008)) * np.exp(-t / (cfg["dur_s"] / 3.2))
    for _ in range(14):  # sparkle grains
        c = rng.uniform(0.0, cfg["dur_s"] * 0.6)
        f = rng.uniform(8000, 12000)
        g = np.exp(-0.5 * ((t - c) / 0.006) ** 2)
        y += 0.35 * g * np.sin(2 * np.pi * f * t)
    return _norm(y * env)


def whoosh(cfg):
    n = int(cfg["dur_s"] * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(3)
    noise = rng.standard_normal(n)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    # sweep approximated by summing time-windowed band-pass slices
    y = np.zeros(n)
    k = 8
    for i in range(k):
        fc = 700 * (6 ** (i / (k - 1)))
        band = np.exp(-0.5 * (np.log(np.maximum(freqs, 1) / fc) / 0.35) ** 2)
        part = np.fft.irfft(spec * band, n)
        win = np.exp(-0.5 * ((t - cfg["dur_s"] * (i + 0.5) / k) / (cfg["dur_s"] / k)) ** 2)
        y += part * win
    env = np.sin(np.pi * np.clip(t / cfg["dur_s"], 0, 1)) ** 1.5
    return _norm(y * env)


def tap(cfg):
    n = int(cfg["dur_s"] * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(5)
    y = np.sin(2 * np.pi * 2100 * t) * np.exp(-t / 0.006) + 0.4 * rng.standard_normal(n) * np.exp(-t / 0.002)
    return _norm(y)


def build_all(force=False):
    st = load_style()["audio"]["sfx"]
    d = os.path.join(ASSETS, "sfx")
    os.makedirs(d, exist_ok=True)
    out = {}
    for name, fn in (("shimmer", shimmer), ("whoosh", whoosh), ("tap", tap)):
        p = os.path.join(d, f"{name}.wav")
        if force or not os.path.exists(p):
            sf.write(p, fn(st[name]).astype(np.float32), SR, subtype="PCM_16")
        out[name] = p
    return out


def generate_music_bed(duration, path, bpm=84, seed=11):
    """Soft, low-energy pad (I-V-vi-IV in C) as a fallback bed when the skill must pick music offline."""
    rng = np.random.default_rng(seed)
    n = int(duration * SR)
    t = np.arange(n) / SR
    chords = [[261.63, 329.63, 392.00], [196.00, 246.94, 293.66], [220.00, 261.63, 329.63], [174.61, 220.00, 261.63]]
    bar = 4 * 60 / bpm
    y = np.zeros(n)
    for k in range(int(np.ceil(duration / bar)) + 1):
        ch = chords[k % 4]
        s0, s1 = int(k * bar * SR), min(n, int((k + 1.15) * bar * SR))
        if s0 >= n:
            break
        tt = t[s0:s1] - k * bar
        env = np.minimum(1, tt / 0.6) * np.minimum(1, np.maximum(0, (bar * 1.15 - tt) / 0.8))
        for f in ch + [ch[0] / 2]:
            ph = rng.uniform(0, 6.28)
            y[s0:s1] += env * (np.sin(2 * np.pi * f * tt + ph) + 0.3 * np.sin(2 * np.pi * 2 * f * tt + ph)) / 4
    # gentle low-pass via moving average
    y = np.convolve(y, np.ones(24) / 24, mode="same")
    fade = np.minimum(1, np.minimum(t / 1.5, (duration - t) / 2.0))
    sf.write(path, _norm(y * fade, -6).astype(np.float32), SR, subtype="PCM_16")
    return path


if __name__ == "__main__":
    print(build_all(force=True))
