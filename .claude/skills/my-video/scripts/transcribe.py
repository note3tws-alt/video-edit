"""Step b (part 1): speech -> words with timestamps, per source clip.

Engine order:
  1. faster-whisper (word_timestamps=True)            -- needs huggingface.co reachable
  2. sherpa-onnx Whisper text + Omnilingual CTC forced alignment (models from GitHub releases)

Usage: python transcribe.py --work WORK [--lang en|hi|ml|auto] [--engine auto|faster-whisper|sherpa]
Writes WORK/transcript.json
"""
import argparse
import glob
import os
import tarfile
import unicodedata
import urllib.request

import numpy as np
import soundfile as sf

from common import load_json, load_style, log, save_json

SHERPA_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{}.tar.bz2"


def models_dir(style):
    d = os.path.expanduser(os.environ.get("MY_VIDEO_MODELS", style["asr"]["models_dir"]))
    os.makedirs(d, exist_ok=True)
    return d


def ensure_model(name, style):
    d = os.path.join(models_dir(style), name)
    if os.path.isdir(d) and glob.glob(os.path.join(d, "*.onnx")):
        return d
    url = SHERPA_URL.format(name)
    tmp = d + ".tar.bz2"
    log("downloading", url)
    urllib.request.urlretrieve(url, tmp)
    with tarfile.open(tmp) as t:
        t.extractall(models_dir(style), filter="data")
    os.remove(tmp)
    return d


# ------------------------------------------------------------------ engine 1: faster-whisper
def fw_transcribe(wavs, lang, style):
    from faster_whisper import WhisperModel
    m = WhisperModel(style["asr"]["faster_whisper_model"], device="auto", compute_type="int8")
    res, detected = {}, None
    for sid, wav in wavs.items():
        segs, info = m.transcribe(wav, language=None if lang == "auto" else lang, word_timestamps=True,
                                  vad_filter=False, condition_on_previous_text=False)
        detected = detected or info.language
        words = []
        for s in segs:
            for w in s.words or []:
                if w.word.strip():
                    words.append({"w": w.word.strip(), "s": round(w.start, 3), "e": round(w.end, 3),
                                  "p": round(w.probability, 3)})
        res[sid] = {"words": words}
    return res, detected


# ------------------------------------------------------------------ engine 2: sherpa whisper + CTC alignment
class CTCAligner:
    def __init__(self, mdir):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(glob.glob(os.path.join(mdir, "model*.onnx"))[0],
                                         providers=["CPUExecutionProvider"])
        toks = open(os.path.join(mdir, "tokens.txt"), encoding="utf-8").read().splitlines()
        self.vocab = {}
        for line in toks:
            sym, idx = line[:line.rfind(" ")], int(line[line.rfind(" ") + 1:])
            self.vocab[sym] = idx
        self.space = self.vocab.get(" ", 4)
        self.blank = 0

    def logprobs(self, audio):
        x = audio.astype(np.float32)[None, :]
        x = (x - x.mean()) / (x.std() + 1e-7)
        lg = self.sess.run(None, {"x": x})[0][0]
        lg = lg - lg.max(axis=1, keepdims=True)
        return lg - np.log(np.exp(lg).sum(axis=1, keepdims=True))

    def tokens_for(self, word):
        w = unicodedata.normalize("NFC", word).lower()
        return [self.vocab[c] for c in w if c in self.vocab and c != " "]

    def align(self, audio, sr, words):
        """CTC Viterbi forced alignment; returns [(start_s, end_s) or None] per word."""
        lp = self.logprobs(audio)
        T = lp.shape[0]
        stride = len(audio) / sr / T
        target, owner = [], []
        for wi, w in enumerate(words):
            ids = self.tokens_for(w)
            if not ids:
                continue
            if target:
                target.append(self.space)
                owner.append(-1)
            target += ids
            owner += [wi] * len(ids)
        if not target:
            return [None] * len(words)
        L = len(target)
        S = 2 * L + 1
        ext = np.full(S, self.blank)
        ext[1::2] = target
        if T < L:
            return [None] * len(words)
        neg = -1e30
        dp = np.full(S, neg)
        dp[0] = lp[0, self.blank]
        dp[1] = lp[0, ext[1]]
        back = np.zeros((T, S), dtype=np.int8)
        skip_ok = np.zeros(S, bool)
        skip_ok[3::2] = ext[3::2] != ext[1:-2:2]
        for t in range(1, T):
            stay = dp
            prev1 = np.concatenate([[neg], dp[:-1]])
            prev2 = np.concatenate([[neg, neg], dp[:-2]])
            prev2 = np.where(skip_ok, prev2, neg)
            cand = np.stack([stay, prev1, prev2])
            arg = cand.argmax(0)
            dp = cand[arg, np.arange(S)] + lp[t, ext]
            back[t] = arg
        s = S - 1 if dp[S - 1] >= dp[S - 2] else S - 2
        path = np.zeros(T, dtype=np.int64)
        for t in range(T - 1, -1, -1):
            path[t] = s
            s -= int(back[t, s])
        spans = {}
        for t, st in enumerate(path):
            if st % 2 == 1:
                wi = owner[st // 2]
                if wi >= 0:
                    a, b = spans.get(wi, (t, t))
                    spans[wi] = (min(a, t), max(b, t))
        return [(spans[i][0] * stride, (spans[i][1] + 1) * stride) if i in spans else None for i in range(len(words))]


def split_points(y, sr, max_len=25.0, min_len=12.0):
    hop = int(sr * 0.02)
    e = np.array([np.sqrt(np.mean(y[i:i + hop] ** 2)) for i in range(0, len(y), hop)])
    cuts, maxf, minf = [0], int(max_len * sr / hop), int(min_len * sr / hop)
    while len(e) - cuts[-1] > maxf:
        lo, hi = cuts[-1] + minf, cuts[-1] + maxf
        cuts.append(lo + int(np.argmin(e[lo:hi])))
    cuts.append(len(e))
    return [(a * hop, min(b * hop, len(y))) for a, b in zip(cuts[:-1], cuts[1:])]


def fill_missing(times, t0, t1):
    """Interpolate None spans between aligned neighbours."""
    n = len(times)
    out = list(times)
    i = 0
    while i < n:
        if out[i] is None:
            j = i
            while j < n and out[j] is None:
                j += 1
            a = out[i - 1][1] if i > 0 else t0
            b = out[j][0] if j < n else t1
            step = max(b - a, 0.01) / (j - i)
            for k in range(i, j):
                out[k] = (a + step * (k - i), a + step * (k - i + 1))
            i = j
        else:
            i += 1
    return out


def sherpa_transcribe(wavs, lang, style, whisper_name=None):
    import sherpa_onnx
    wdir = ensure_model(whisper_name or style["asr"]["fallback_whisper_model"], style)
    cdir = ensure_model(style["asr"]["fallback_ctc_model"], style)
    pick = lambda pat: (glob.glob(os.path.join(wdir, pat.replace(".onnx", ".int8.onnx"))) or glob.glob(os.path.join(wdir, pat)))[0]
    rec = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=pick("*encoder.onnx"), decoder=pick("*decoder.onnx"), tokens=glob.glob(os.path.join(wdir, "*tokens.txt"))[0],
        language="" if lang == "auto" else lang, task="transcribe", num_threads=os.cpu_count() or 4)
    aligner = CTCAligner(cdir)
    res, detected = {}, None
    for sid, wav in wavs.items():
        y, sr = sf.read(wav, dtype="float32")
        if y.ndim > 1:
            y = y.mean(1)
        words = []
        for a, b in split_points(y, sr):
            st = rec.create_stream()
            st.accept_waveform(sr, y[a:b])
            rec.decode_stream(st)
            r = st.result
            detected = detected or (getattr(r, "lang", "") or "").strip("<|>") or None
            toks = r.text.split()
            if not toks:
                continue
            spans = fill_missing(aligner.align(y[a:b], sr, toks), 0.0, (b - a) / sr)
            off = a / sr
            for w, (s, e) in zip(toks, spans):
                words.append({"w": w, "s": round(off + s, 3), "e": round(off + e, 3), "p": None})
        res[sid] = {"words": words}
        log(f"{sid}: {len(words)} words")
    return res, detected


def transcribe(work, lang="auto", engine="auto", whisper_name=None):
    style = load_style()
    srcs = load_json(os.path.join(work, "sources.json"))["sources"]
    wavs = {s["id"]: s["wav16"] for s in srcs}
    res = None
    if engine in ("auto", "faster-whisper"):
        try:
            res, det = fw_transcribe(wavs, lang, style)
            used = "faster-whisper:" + style["asr"]["faster_whisper_model"]
        except Exception as e:  # network blocked / not installed
            if engine == "faster-whisper":
                raise
            log("faster-whisper unavailable, falling back to sherpa-onnx:", str(e).splitlines()[0][:200])
    if res is None:
        res, det = sherpa_transcribe(wavs, lang, style, whisper_name)
        used = "sherpa-onnx whisper + omnilingual CTC alignment"
    language = lang if lang != "auto" else (det or "en")
    out = {"language": language, "engine": used, "sources": res}
    save_json(out, os.path.join(work, "transcript.json"))
    n = sum(len(v["words"]) for v in res.values())
    log(f"transcribed {n} words, language={language}, engine={used}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--lang", default="auto")
    ap.add_argument("--engine", default="auto", choices=["auto", "faster-whisper", "sherpa"])
    ap.add_argument("--whisper-model", default=None, help="sherpa whisper model dir name, e.g. sherpa-onnx-whisper-turbo")
    a = ap.parse_args()
    transcribe(a.work, a.lang, a.engine, a.whisper_model)
