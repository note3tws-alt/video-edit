#!/usr/bin/env bash
# One-time setup for the my-video skill: Python deps, SFX assets, ASR models.
# Safe to re-run. Models go to ${MY_VIDEO_MODELS:-~/.cache/my-video/models}.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

command -v ffmpeg >/dev/null || { echo "ffmpeg is required (apt-get install ffmpeg / brew install ffmpeg)"; exit 1; }

python3 -m pip install -q numpy opencv-python-headless pillow soundfile onnxruntime sherpa-onnx
# faster-whisper is optional (needs huggingface.co at first run); failures are fine
python3 -m pip install -q faster-whisper || echo "faster-whisper not installed; the sherpa-onnx fallback will be used"

python3 -c "from PIL import features; assert features.check('raqm'), 'Pillow lacks raqm (needed for Hindi/Malayalam shaping)'"
cd "$HERE"
python3 sfx.py >/dev/null

# Fallback ASR models (GitHub releases) - downloaded only if huggingface.co is unreachable
if ! python3 - <<'EOF' 2>/dev/null
import urllib.request; urllib.request.urlopen("https://huggingface.co", timeout=8)
EOF
then
  echo "huggingface.co unreachable -> fetching sherpa-onnx fallback models"
  python3 - <<'EOF'
from common import load_style
from transcribe import ensure_model
s = load_style()
ensure_model(s["asr"]["fallback_whisper_model"], s)
ensure_model(s["asr"]["fallback_ctc_model"], s)
EOF
fi
echo "my-video setup OK"
