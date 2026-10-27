"""Shared helpers for the my-video pipeline: style loading, ffmpeg wrappers, text/language utilities."""
import json
import math
import os
import re
import shutil
import subprocess
import sys
import unicodedata

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(SKILL_DIR, "assets")
FONTS = os.path.join(ASSETS, "fonts")


def load_style(path=None):
    with open(path or os.path.join(SKILL_DIR, "style.json"), encoding="utf-8") as f:
        return json.load(f)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def log(*a):
    print("[my-video]", *a, file=sys.stderr, flush=True)


def run(cmd, quiet=True, check=True):
    """Run a command list; raise with stderr tail on failure."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and p.returncode != 0:
        tail = p.stderr.decode("utf-8", "replace")[-3000:]
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(map(str, cmd))[:400]}\n{tail}")
    if not quiet:
        sys.stderr.write(p.stderr.decode("utf-8", "replace"))
    return p


def ffprobe(path):
    p = run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", path])
    return json.loads(p.stdout)


def media_info(path):
    j = ffprobe(path)
    v = next((s for s in j["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in j["streams"] if s["codec_type"] == "audio"), None)
    info = {"duration": float(j["format"].get("duration", 0)), "has_audio": a is not None, "has_video": v is not None}
    if v:
        num, den = v.get("avg_frame_rate", "0/1").split("/")
        rot = 0
        for sd in v.get("side_data_list", []) or []:
            if "rotation" in sd:
                rot = int(sd["rotation"])
        rot = int(v.get("tags", {}).get("rotate", rot))
        w, h = int(v["width"]), int(v["height"])
        if abs(rot) in (90, 270):
            w, h = h, w
        info.update(width=w, height=h, fps=float(num) / float(den) if float(den) else 0.0, vcodec=v["codec_name"],
                    pix_fmt=v.get("pix_fmt"), color_transfer=v.get("color_transfer"), color_space=v.get("color_space"))
    if a:
        info.update(sample_rate=int(a.get("sample_rate", 0)), channels=int(a.get("channels", 0)), acodec=a["codec_name"])
    return info


def have(binary):
    return shutil.which(binary) is not None


# ---------------------------------------------------------------- easing
def ease(t, kind="out"):
    t = min(1.0, max(0.0, t))
    if kind == "out":
        return 1 - (1 - t) ** 3
    if kind == "in":
        return t ** 3
    if kind == "inout":
        return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2
    return t


def lerp(a, b, t):
    return a + (b - a) * t


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# ---------------------------------------------------------------- text / language
def script_of(text):
    """Dominant script of a string: 'devanagari', 'malayalam' or 'latin'."""
    dev = sum(1 for c in text if "ऀ" <= c <= "ॿ")
    mal = sum(1 for c in text if "ഀ" <= c <= "ൿ")
    if dev == 0 and mal == 0:
        return "latin"
    return "devanagari" if dev >= mal else "malayalam"


def norm_word(w):
    """Lowercase, strip punctuation; keeps Indic combining marks."""
    w = unicodedata.normalize("NFC", w).lower()
    return "".join(c for c in w if c.isalnum() or unicodedata.category(c) in ("Mn", "Mc"))


def has_upper(lang):
    return lang not in ("hi", "ml")


def visual_len(w):
    """Grapheme-ish length (ignores combining marks) for word-length scoring."""
    return sum(1 for c in w if unicodedata.category(c) not in ("Mn", "Mc") and c.isalnum())


STOPWORDS = {
    "en": set("""a an the and or but if so of to in on at by for with from as is are was were be been being am do does did
        have has had i you he she it we they me him her us them my your his its our their this that these those there here
        what which who whom whose when where why how not no yes just very really can could will would shall should may might
        must about into over than then too also only all any some each every both more most other such own same again
        out up down off once while because until through during before after above below between under further s t
        don't can't won't it's i'm you're that's let's gonna wanna like get got go going""".split()),
    "pt": set("""a o as os um uma uns umas de do da dos das em no na nos nas por para pra com sem sob sobre e ou mas
        se que quem qual quais quando onde como porque porquê é são foi era ser estar está estão ter tem têm há ele ela
        eles elas eu tu você vocês nós me te lhe nos vos meu minha seu sua isso isto aquilo esse essa este esta também
        já não sim mais menos muito muita muitos muitas ao aos à às pelo pela pelos pelas então bom aí lá cá""".split()),
    "hi": set("""और का की के को में से पर है हैं था थी थे हो होता होती होते भी तो ही यह वह ये वो इस उस इन उन एक कि जो
        क्या क्यों कैसे कब कहाँ मैं तुम आप हम वे मेरा मेरी तेरा आपका आपकी हमारा लिए साथ बहुत अब फिर या लेकिन नहीं ना
        कर करना करते करती किया गया गई जा जाता जाती रहा रही रहे""".split()),
    "ml": set("""ഒരു ഈ ആ അത് ഇത് എന്ന് എന്ന ആണ് ആയി ഉണ്ട് ഇല്ല ഞാൻ നിങ്ങൾ നമ്മൾ അവർ അവൻ അവൾ എന്റെ നിങ്ങളുടെ
        പിന്നെ ഇനി കൂടി ഒക്കെ എന്തു എന്ത് എങ്ങനെ എപ്പോൾ എവിടെ ആര് പക്ഷേ അല്ലെങ്കിൽ ഉം ഒന്ന് വളരെ ആയിരുന്നു""".split()),
}


def stopwords(lang):
    return STOPWORDS.get(lang, STOPWORDS["en"])


SENT_END = re.compile(r"[.?!।॥]$")
CLAUSE_END = re.compile(r"[.?!,;:।॥]$")


def frames(t, fps):
    return int(round(t * fps))
