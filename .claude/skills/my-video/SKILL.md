---
name: my-video
description: Edit raw talking-head clips into the user's signature 9:16 short-form style (measured from their reference video) - word-by-word captions with gold/serif hero words, punch-in zooms, zoom-blur transitions, B-roll, info cards, warm grade, SFX. Use whenever the user says "make my video", "make my video with these footages", "edit these footages in my style", "edit this in my style", or hands over raw clips to be edited the usual way. Works for English, Hindi and Malayalam (and other Whisper languages).
---

# my-video

Every value used here lives in `style.json` (measured from the reference, never eyeballed). Follow it exactly unless the
user explicitly overrides something for this video. Only change `style.json` itself if the user says the change is permanent.

## 0. Before anything: ask exactly two questions (nothing else)

1. **Target length?** (seconds; "as is" = no trimming beyond silence removal)
2. **Are you supplying music, or should I pick?** (reference has no music; default is voice + SFX only)

Do not ask about B-roll, captions, fonts or style. B-roll: if the user attached B-roll clips/folder use them, otherwise
fetch from Pexels automatically (needs `PEXELS_API_KEY` and network access to api.pexels.com + videos.pexels.com; if
unavailable, the slot is dropped and you say so in the delivery note).

## 1. Setup (once per machine/session)

```bash
bash .claude/skills/my-video/scripts/setup.sh
```
Installs Python deps, builds SFX, and - when huggingface.co is unreachable - downloads the fallback ASR models
(Whisper ONNX + Omnilingual CTC aligner) from GitHub releases into `~/.cache/my-video/models`.

## 2. Plan (steps a-c + B-roll)

```bash
cd .claude/skills/my-video/scripts
python3 make_video.py --clips <files or folder> --work <WORK> --stage plan \
    [--target-length 60] [--broll-dir <folder>] [--lang auto|en|hi|ml]
```
- a. ingest: cover-crop to 1080x1920 (face-aware), 30 fps CFR, BT.709, HDR (HLG/PQ) tone-mapped to SDR, 48 kHz mono.
- b. transcribe: faster-whisper `large-v3-turbo` with word timestamps; fallback Whisper ONNX + Omnilingual CTC forced
  alignment. For Malayalam/Hindi on the fallback path prefer `--whisper-model sherpa-onnx-whisper-turbo` (or medium).
- c. plan: removes fillers, stutters and every pause > 0.35 s (pads 0.06/0.10 s), picks the hook (question / strongest
  claim; cold-opens a later sentence if it scores 15 % higher), trims to target length by dropping lowest-value sentences
  (never the hook or the last 2 = CTA), then builds shots, zooms, transitions, B-roll slots, captions, hero words.

## 3. Review WORK/plan.json (you, Claude - this is what makes it "edited", not "processed")

Read `plan.json` (captions, words, broll, card_suggestions) and fix it in place:
1. **Proofread captions** against the meaning: ASR errors, names, numbers, currency, Hindi/Malayalam spelling.
   Keep punctuation; sentence case. Never invent words.
2. **Hero words** (`"hero"` on a caption word: `serif_gold | serif_white | caps_gold | caps_white`): must be the
   keyword of the line (noun/claim/number/negation). Questions and B-roll lines -> white; key concepts -> gold
   (~58 % gold). Density is front-loaded: ~1 per 2.3 s in the first third, ~1 per 7.8 s in the middle, none in the
   last third. Consecutive hero words form one phrase ("é que não."). `annot: "red_circle"` on the first negation hero.
3. **Cards** (`plan["cards"]`, ~3 per 70 s near `card_suggestions`): write content ONLY from what is said, timed with
   `at_word`. Schemas: see `scripts/cards.py` docstring (`compare`, `timeline`, `checklist`; `acronym` is auto).
   Use cards where the speaker compares, lists, or gives numbers.
4. **B-roll** (`plan["broll"]`): set `query` to 2-4 concrete English words for Pexels (even for hi/ml videos), or set
   `path` to the best-matching user clip. Then `python3 broll.py --work WORK [--broll-dir DIR] --refill`.
5. **Bad takes / repeated lines**: write `WORK/overrides.json` `{"exclude":[{"src":"S01","from":12.3,"to":15.0}]}`
   (source seconds from `words`), optional `"hook": {"src","from","to"}`, then re-run the plan stage and redo 1-4.

## 4. Render + verify (steps d-g)

```bash
python3 make_video.py --work <WORK> --stage render --out <OUT>.mp4 --music none|auto|/path/track.mp3 \
    [--reference <reference.mp4> --ref-crop 356:634:267:273]
```
Produces the MP4 (1080x1920, 30 fps, H.264 CRF 18 + AAC 192k, -14 LUFS), `WORK/verify_report.md`,
`WORK/side_by_side.png` (with `--reference`). Use `--max-seconds 10` for a quick preview while iterating.

**Any FAIL in the report must be fixed before delivery** (re-plan, edit plan.json, re-render). WARNs need a one-line
explanation to the user. Deliver: the MP4 path, the verify table, and what (if anything) was dropped (e.g. B-roll).

## 5. Style checklist (verify_report.md checks these automatically)

| Area | Target (measured) |
|---|---|
| Format | 1080x1920, 9:16, 30 fps, H.264/AAC, bt709 |
| Pacing | mean shot 3.34 s (median 3.10, min 0.30, max 8.60); by thirds 2.1 / 4.7 / 4.7 s |
| Cuts | jump cuts at removed pauses; B-roll hard in/out; 3 zoom-blur transitions per 70 s at ~35/67/79 % |
| Hook (0-3.17 s) | blur-in 6 frames + zoom 1.10->1.00 (0.6 s); push 1.00->1.075 (0.37 s) on the hook hero; top serif question (163 px, y 0.245) + bottom caps-gold keyword |
| Motion | punch levels 1.03/1.05/1.09/1.12 (anchor 0.536, 0.544); animated zooms only in first 23 %; outro (last 10.6 %) static 1.00 |
| Captions | word-by-word, fade+unblur+rise 40 px in 4 frames; 1-3 words (median 2, <=16 chars, <=0.9 s); base 60 px white, y 0.66 (0.64-0.68); on B-roll y 0.59 (mono) / 0.52 (warm) |
| Hero | serif 128 px / caps <=170 px (<=0.67 W); gold gradient #FFF4DC->#F4CF92->#B98A4A + glow; first "but..." pivot on top with a "1" |
| Graphics | red circle on negation; acronym boxes; 3 cards / 70 s (white, radius 22, y 0.68, width 0.67); halo on mono B-roll |
| Color | LUT `assets/luts/my-video.cube` (= curves in style.json, luma x1.13, contrast x1.19, sat x1.36, warmth +8) |
| B-roll | 2 per 70 s at 12 % / 50 %, 4.2 s / 7.7 s (17 % of runtime), over jump-cut clusters; #1 mono + halo, #2 warm Ken Burns 1.075->1.00 |
| Audio | -14 LUFS, TP <= -1 dBTP; shimmer SFX on gold heroes/cards (-24.6 dB vs voice); whoosh on zoom-blur; no music unless asked (-20 dB, ducked) |
| Branding | none: no intro, logo, watermark or end card; spoken CTA only |

## 6. Limitations (closest approximation used)

- **Fonts**: exact reference fonts are unknown; free look-alikes: Inter (base), Playfair Display Bold Italic (serif hero),
  Montserrat ExtraBold (caps hero); Hindi: Noto Sans Devanagari + Tiro Devanagari Hindi Italic; Malayalam: Noto Sans
  Malayalam + Noto Serif Malayalam with a synthetic 12 deg slant (no Malayalam italic exists). Indic scripts have no
  upper case, so caps-tier heroes use the heavy weight instead.
- **Halo on B-roll**: placed by face detection (YuNet) every 2 frames and smoothed - no true motion tracking or
  occlusion behind the head.
- **Card hand cursor**: replaced by a tap ripple; **red circle**: procedural hand-drawn ellipse.
- **SFX**: synthesised from the measured spectra (shimmer partials 4.5-9.7 kHz, whoosh, tap), not the original samples;
  whoosh-on-transition was only "probable" in the reference.
- **Music**: the reference has none, so the -20 dB level is a default, not a measurement. "You pick" uses
  `assets/music/` if the user drops tracks there, else a generated soft pad (no online music source is reachable).
- **B-roll**: needs user clips or Pexels (API key + network). Without either, slots are dropped.
- **Creative text edits** the reference editor made (e.g. dropping a word in a hero line) are not automated - captions
  are verbatim unless you edit them in step 3. The "1" pivot marker and hook layout are generalised from single
  examples in the reference.
- **ASR**: faster-whisper needs huggingface.co; the fallback (Whisper-small ONNX + CTC alignment) is weaker for Malayalam
  - always proofread, or use a larger fallback model.
- **Color**: one global grade fitted on the reference's raw->graded pixels (MAE 5.6/255); no per-shot matching or skin
  correction for very different cameras/lighting.
- **Loudness**: reference true peak was clipped (+0.2 dBTP); output is limited to -1 dBTP on purpose.
- **Not replicated by design**: the "Antes/Depois" before/after showcase wrapper of the reference file (measurements kept
  in style.json -> reference.showcase_layout_not_replicated).

## Files

- `style.json` - every measured parameter (single source of truth)
- `scripts/make_video.py` - orchestrator; `ingest.py`, `transcribe.py`, `plan.py`, `broll.py`, `audio.py`, `render.py`,
  `verify.py` (steps a-g + checklist); `textfx.py`, `cards.py`, `sfx.py`, `faces.py`, `make_lut.py` helpers
- `assets/fonts` (OFL), `assets/luts`, `assets/sfx`, `assets/models/face_detection_yunet_2023mar.onnx` (MIT)
