# Forensic Call Transcription Pipeline

CPU-only pipeline tuned for a Dell OptiPlex-class workstation (13th Gen i7,
20 cores, 32GB RAM, no dedicated GPU). Designed to minimize human review time
by doing automated confidence-checking and cross-verification before
anything reaches a person.

## What it does

1. **Quality check** – estimates SNR/clipping on the original audio, no modification.
2. **Preprocessing** – normalizes and (if audio is degraded) denoises a *copy*.
   Original file is never touched.
3. **Pass 1** – fast transcription with `faster-whisper large-v3-turbo` (int8, CPU),
   VAD-filtered to skip silence and reduce hallucinations.
4. **Confidence scoring** – flags any segment with high no-speech probability,
   low average log-probability, or a high compression ratio (a hallucination signal).
5. **Pass 2** – only the *flagged* segments are re-transcribed with full
   `large-v3` for a second opinion. If pass 1 and pass 2 agree, the segment is
   auto-resolved with no human involvement. If they disagree, it's queued for
   review — this is the part that actually needs your time.
6. **Diarization** – `pyannote.audio` speaker labeling merged onto the transcript.
7. **Output** – full JSON transcript + audit trail, a readable `.txt` transcript
   with review-needed spots marked inline, and a separate `_review_queue.json`
   containing *only* the handful of segments that need a human ear.

## One-time setup

```bash
# System dependency
sudo apt-get install ffmpeg

# Python packages
pip install faster-whisper pyannote.audio soundfile numpy scipy --break-system-packages
```

**pyannote requires a free Hugging Face token** (one-time):
1. Create an account at huggingface.co
2. Accept the user agreement on the model page:
   `pyannote/speaker-diarization-3.1`
3. Generate an access token: huggingface.co/settings/tokens
4. Either run `huggingface-cli login` once, or pass `--hf-token YOUR_TOKEN`,
   or set the `HF_TOKEN` environment variable.

Models download once (~3GB total for large-v3 + large-v3-turbo) and are cached
locally after that — no internet needed for subsequent runs.

## Usage

```bash
python transcribe_pipeline.py \
    --input /path/to/call_recording.wav \
    --case-id CASE-2026-0417 \
    --outdir ./output
```

Optional flags:
- `--skip-diarization` — skip speaker labeling (faster, if you only need raw transcript)
- `--hf-token TOKEN` — pass a Hugging Face token directly instead of env var/login

## Output files

For an input `call_recording.wav`, you get:
- `call_recording_transcript.json` — full transcript with timestamps, confidence
  scores, speaker labels, and audit metadata (model versions, thresholds used,
  processing time) — this is what supports reproducibility if the methodology
  is ever challenged.
- `call_recording_transcript.txt` — human-readable transcript, review-needed
  segments marked inline with `[** REVIEW NEEDED **]` and the reason.
- `call_recording_review_queue.json` — just the segments needing a human
  listen, so review time is spent only where it's actually needed.

## Notes on accuracy vs. speed on your hardware

- With no GPU, expect roughly real-time to a few times faster than real-time
  per call using `large-v3-turbo` on your 20-core CPU — benchmark on a
  representative sample before committing to a throughput estimate for casework.
- Pass 2 only runs on flagged segments, so it adds relatively little total
  time even though `large-v3` alone is much slower than `large-v3-turbo`.
- `condition_on_previous_text=False` is intentionally set — it stops Whisper
  from letting one bad guess snowball into repeated errors later in the call,
  which is a known hallucination pattern on noisy audio.
- The confidence thresholds at the top of the script
  (`NOSPEECH_PROB_THRESHOLD`, `AVG_LOGPROB_THRESHOLD`,
  `COMPRESSION_RATIO_THRESHOLD`) are reasonable defaults — tune them against
  a batch of calls where you know the ground truth, to find the right
  precision/recall balance for how much review time you're willing to spend.

## Testing in GitHub Codespaces

This repo includes a `.devcontainer/` config for quick testing in a
Codespace. **Important: Codespaces is a cloud environment — never upload
real case/evidentiary audio there.** Use it only to test the pipeline's
behavior, output format, and diarization logic against synthetic audio.
Run production transcriptions on your local Dell workstation.

### Setup

1. Push this folder to a GitHub repo.
2. Open it in a Codespace (Code → Codespaces → Create codespace).
   Pick at least a 4-core/16GB machine type if available — the free-tier
   default (2-core/8GB) will still work but is slow even with small models.
3. `postCreateCommand` runs automatically: installs ffmpeg, espeak-ng, and
   Python deps, and generates synthetic test audio in `./test_audio/`
   (`sample_clean.wav`, `sample_two_speakers.wav`, `sample_noisy.wav` — the
   last one has added noise to test the denoise/quality-check path).

### Run a test

```bash
python transcribe_pipeline.py \
    --input test_audio/sample_two_speakers.wav \
    --case-id TEST-001 \
    --outdir ./output \
    --fast-model small \
    --accurate-model medium \
    --hf-token YOUR_HF_TOKEN
```

Use smaller models (`--fast-model small --accurate-model medium`) in
Codespaces — a 2-4 core machine will struggle with `large-v3-turbo`/`large-v3`
at usable speed. On your Dell workstation, omit these flags to use the
production defaults (`large-v3-turbo` / `large-v3`).

If you don't have an HF token set up yet, add `--skip-diarization` to test
just the transcription/confidence-flagging logic first.

### Generating more test audio

Re-run the generator any time with different content:

```bash
python generate_test_audio.py --outdir ./test_audio
```

It uses `espeak-ng` (fully offline TTS) so no external data or network calls
are needed to produce test material — safe to run repeatedly in the
cloud dev environment.

## Chain of custody reminders (not automated by this script)

- Keep the original audio file and its hash (e.g. `sha256sum call.wav`)
  recorded before any processing.
- The JSON audit trail records model versions and thresholds used — store it
  alongside the transcript as part of your case file.
- Treat the final `.txt`/`.json` transcript as a *draft pending human
  certification* until someone has reviewed the flagged segments and signed off.
