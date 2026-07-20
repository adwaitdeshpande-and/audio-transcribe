#!/usr/bin/env bash
set -e

echo ">>> Installing system dependencies (ffmpeg, espeak-ng for synthetic test audio)..."
sudo apt-get update -qq
sudo apt-get install -y -qq ffmpeg espeak-ng > /dev/null

echo ">>> Installing Python dependencies..."
pip install --quiet -r requirements.txt

echo ">>> Generating synthetic test audio (safe, non-evidentiary)..."
python generate_test_audio.py --outdir ./test_audio

cat <<'EOF'

============================================================
Codespace setup complete.

IMPORTANT: This is a cloud dev environment. Do NOT upload
real case/evidence audio here. Use it only to test the
pipeline against the synthetic samples in ./test_audio,
or your own non-sensitive audio.

Quick test run:
  python transcribe_pipeline.py \
      --input test_audio/sample_two_speakers.wav \
      --case-id TEST-001 \
      --outdir ./output \
      --fast-model small \
      --accurate-model medium \
      --hf-token YOUR_HF_TOKEN

(Free HF token + accept terms for pyannote/speaker-diarization-3.1
 at huggingface.co -- required once for diarization.)
============================================================
EOF
