"""One-command pipeline for the my-video style.

Stage "plan"  : ingest -> transcribe -> plan -> B-roll fill            (then review/edit WORK/plan.json)
Stage "render": audio mix -> render -> verify
Stage "all"   : both

Examples
  python make_video.py --clips raw/ --work work --stage plan --target-length 60
  python make_video.py --work work --stage render --out final.mp4 --music none
"""
import argparse
import os
import time

from common import load_json, load_style, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=[])
    ap.add_argument("--work", default="my_video_work")
    ap.add_argument("--out", default="my_video.mp4")
    ap.add_argument("--stage", default="all", choices=["all", "plan", "render"])
    ap.add_argument("--target-length", type=float, default=None)
    ap.add_argument("--music", default="none", help="none | auto | path to a track")
    ap.add_argument("--broll-dir", default=None)
    ap.add_argument("--no-pexels", action="store_true")
    ap.add_argument("--lang", default="auto", help="auto | en | hi | ml | ...")
    ap.add_argument("--asr", default="auto", choices=["auto", "faster-whisper", "sherpa"])
    ap.add_argument("--whisper-model", default=None, help="sherpa fallback model, e.g. sherpa-onnx-whisper-turbo")
    ap.add_argument("--fresh", action="store_true", help="re-ingest and re-transcribe even if cached")
    ap.add_argument("--reference", default=None, help="reference video for the side-by-side sheet")
    ap.add_argument("--ref-crop", default=None)
    ap.add_argument("--max-seconds", type=float, default=None, help="render only the first N seconds (preview)")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    t0 = time.time()

    if a.stage in ("all", "plan"):
        from ingest import collect, ingest
        from transcribe import transcribe
        from plan import plan
        from broll import fill
        src_json = os.path.join(a.work, "sources.json")
        files = [os.path.abspath(f) for f in collect(a.clips)]
        cached = os.path.exists(src_json) and [s["src"] for s in load_json(src_json)["sources"]] == files
        if not files and not os.path.exists(src_json):
            raise SystemExit("no clips given")
        if files and (a.fresh or not cached):
            ingest(files, a.work, load_style())
            if os.path.exists(os.path.join(a.work, "transcript.json")):
                os.remove(os.path.join(a.work, "transcript.json"))
        if a.fresh or not os.path.exists(os.path.join(a.work, "transcript.json")):
            transcribe(a.work, a.lang, a.asr, a.whisper_model)
        plan(a.work, a.target_length, os.path.join(a.work, "overrides.json"))
        fill(a.work, a.broll_dir, not a.no_pexels)
        log(f"plan stage done in {time.time() - t0:.0f}s -> review {os.path.join(a.work, 'plan.json')}")

    if a.stage in ("all", "render"):
        from audio import mix
        from render import render
        from verify import verify
        mix(a.work, a.music)
        render(a.work, a.out, a.max_seconds)
        verify(a.work, a.out, a.reference, a.ref_crop)
        log(f"done in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
