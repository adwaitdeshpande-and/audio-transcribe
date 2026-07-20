#!/usr/bin/env python3
"""
Forensic Call Transcription Pipeline
=====================================
CPU-optimized (no GPU required), designed for Dell OptiPlex-class hardware
(13th Gen i7, 20 cores, 32GB RAM).

Pipeline:
  1. Audio quality check (SNR, sample rate, clipping)
  2. Preprocessing (resample, normalize; optional denoise for low-SNR audio)
  3. Pass 1 transcription: faster-whisper large-v3-turbo (int8) + VAD
  4. Confidence scoring per segment (avg logprob + no-speech prob)
  5. Pass 2 (targeted): re-transcribe LOW-CONFIDENCE segments only with
     full large-v3 for a second opinion -> auto-resolve or flag for human
  6. Speaker diarization: pyannote.audio
  7. Merge transcript + diarization -> speaker-labeled, timestamped transcript
  8. Output: JSON (full detail + audit trail) and a human-readable .txt
     with only the genuinely uncertain spots flagged for review.

This script does NOT alter the original audio file. All processing happens
on a copy. Keep the original as your evidentiary exhibit.

Setup (one-time):
    pip install faster-whisper pyannote.audio soundfile numpy scipy --break-system-packages
    # pyannote diarization models require a (free) Hugging Face token:
    # 1) huggingface.co -> create account -> accept terms for pyannote/speaker-diarization-3.1
    # 2) huggingface-cli login   (or set HF_TOKEN env var)

Usage:
    python transcribe_pipeline.py --input call_001.wav --case-id CASE-2026-0417 --outdir ./output
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Defaults tuned for the production workstation (20-core CPU, 32GB RAM).
# Override with --fast-model/--accurate-model for lighter-weight testing,
# e.g. on GitHub Codespaces (typically 2-4 cores, 8-16GB RAM):
#   --fast-model small --accurate-model medium
FAST_MODEL_DEFAULT = "large-v3-turbo"   # pass 1: fast triage pass
ACCURATE_MODEL_DEFAULT = "large-v3"     # pass 2: re-check low-confidence segments only
COMPUTE_TYPE = "int8"                   # best for CPU-only inference
CPU_THREADS = os.cpu_count() or 8

# These get set from CLI args at runtime; module-level so all functions see them
FAST_MODEL = FAST_MODEL_DEFAULT
ACCURATE_MODEL = ACCURATE_MODEL_DEFAULT

# Segment is flagged for human review if:
NOSPEECH_PROB_THRESHOLD = 0.5     # model thinks this might not be speech at all
AVG_LOGPROB_THRESHOLD = -0.65     # low = model was "guessing"
COMPRESSION_RATIO_THRESHOLD = 2.4 # high = repetitive/garbled text (hallucination signal)

SNR_LOW_THRESHOLD_DB = 15.0       # below this, audio is treated as "degraded" -> denoise


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float
    speaker: Optional[str] = None
    pass2_text: Optional[str] = None
    pass2_agrees: Optional[bool] = None
    needs_review: bool = False
    review_reason: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Step 1: Audio quality check
# ---------------------------------------------------------------------------

def check_audio_quality(input_path: str) -> dict:
    """Estimate basic quality metrics using ffmpeg's volumedetect + a quick SNR proxy."""
    import soundfile as sf

    info = sf.info(input_path)
    data, sr = sf.read(input_path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)  # downmix to mono for analysis

    # crude SNR estimate: compare RMS of loudest vs quietest 10% of 50ms frames
    frame_len = int(sr * 0.05)
    if frame_len < 1:
        frame_len = 1
    n_frames = max(1, len(data) // frame_len)
    frame_rms = np.array([
        np.sqrt(np.mean(data[i * frame_len:(i + 1) * frame_len] ** 2) + 1e-12)
        for i in range(n_frames)
    ])
    sorted_rms = np.sort(frame_rms)
    noise_floor = np.mean(sorted_rms[: max(1, n_frames // 10)]) + 1e-9
    signal_peak = np.mean(sorted_rms[-max(1, n_frames // 10):]) + 1e-9
    snr_db = 20 * np.log10(signal_peak / noise_floor)

    clipping_ratio = float(np.mean(np.abs(data) > 0.99))

    quality = {
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "duration_sec": round(len(data) / sr, 2),
        "estimated_snr_db": round(float(snr_db), 2),
        "clipping_ratio": round(clipping_ratio, 4),
        "classification": "degraded" if snr_db < SNR_LOW_THRESHOLD_DB else "clean",
    }
    return quality


# ---------------------------------------------------------------------------
# Step 2: Preprocessing
# ---------------------------------------------------------------------------

def preprocess_audio(input_path: str, work_path: str, degraded: bool) -> str:
    """
    Produce a normalized (and, if degraded, denoised) 16kHz mono WAV for
    transcription. The original file is never modified.
    """
    filters = ["highpass=f=80", "lowpass=f=8000", "loudnorm"]
    if degraded:
        # afftdn = FFT-based denoiser, conservative settings to avoid eating speech
        filters.insert(0, "afftdn=nr=12:nf=-25")

    filter_chain = ",".join(filters)
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1",
        "-af", filter_chain,
        work_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return work_path


# ---------------------------------------------------------------------------
# Step 3/4: Pass 1 transcription with confidence scoring
# ---------------------------------------------------------------------------

def transcribe_pass1(audio_path: str) -> List[Segment]:
    from faster_whisper import WhisperModel

    model = WhisperModel(FAST_MODEL, device="cpu", compute_type=COMPUTE_TYPE,
                          cpu_threads=CPU_THREADS)

    segments_iter, info = model.transcribe(
        audio_path,
        vad_filter=True,                       # skip silence -> fewer hallucinations, faster
        vad_parameters=dict(min_silence_duration_ms=500),
        word_timestamps=False,
        beam_size=5,
        condition_on_previous_text=False,       # reduces cascading hallucination on bad audio
    )

    segments = []
    for s in segments_iter:
        segments.append(Segment(
            start=s.start, end=s.end, text=s.text.strip(),
            avg_logprob=s.avg_logprob,
            no_speech_prob=s.no_speech_prob,
            compression_ratio=s.compression_ratio,
        ))
    print(f"  Pass 1 detected language: {info.language} (p={info.language_probability:.2f})",
          file=sys.stderr)
    return segments


def flag_low_confidence(segments: List[Segment]) -> None:
    for seg in segments:
        reasons = []
        if seg.no_speech_prob > NOSPEECH_PROB_THRESHOLD:
            reasons.append("high_no_speech_prob")
        if seg.avg_logprob < AVG_LOGPROB_THRESHOLD:
            reasons.append("low_avg_logprob")
        if seg.compression_ratio > COMPRESSION_RATIO_THRESHOLD:
            reasons.append("high_compression_ratio_possible_hallucination")
        seg.needs_review = len(reasons) > 0
        seg.review_reason = reasons


# ---------------------------------------------------------------------------
# Step 5: Pass 2 - targeted re-check of flagged segments only
# ---------------------------------------------------------------------------

def transcribe_pass2(audio_path: str, segments: List[Segment]) -> None:
    """Re-run ONLY flagged segments through the larger model for a second opinion."""
    flagged = [s for s in segments if s.needs_review]
    if not flagged:
        return

    from faster_whisper import WhisperModel
    import soundfile as sf

    print(f"  Pass 2: re-checking {len(flagged)} flagged segment(s) with {ACCURATE_MODEL}...",
          file=sys.stderr)

    model = WhisperModel(ACCURATE_MODEL, device="cpu", compute_type=COMPUTE_TYPE,
                          cpu_threads=CPU_THREADS)

    data, sr = sf.read(audio_path, dtype="float32")
    for seg in flagged:
        start_sample = max(0, int((seg.start - 0.2) * sr))
        end_sample = min(len(data), int((seg.end + 0.2) * sr))
        clip = data[start_sample:end_sample]

        clip_path = "/tmp/_seg_check.wav"
        sf.write(clip_path, clip, sr)

        result_segments, _ = model.transcribe(
            clip_path, beam_size=5, condition_on_previous_text=False,
        )
        pass2_text = " ".join(s.text.strip() for s in result_segments).strip()
        seg.pass2_text = pass2_text

        # simple agreement check: normalized text overlap
        a = seg.text.lower().split()
        b = pass2_text.lower().split()
        overlap = len(set(a) & set(b)) / max(1, max(len(a), len(b)))
        seg.pass2_agrees = overlap > 0.7

        if seg.pass2_agrees:
            # both models agree -> resolve automatically, drop review flag
            seg.needs_review = False
            seg.review_reason.append("resolved_pass2_agreement")
        else:
            seg.review_reason.append("pass2_disagreement_human_review_required")


# ---------------------------------------------------------------------------
# Step 6: Diarization
# ---------------------------------------------------------------------------

def diarize(audio_path: str, hf_token: Optional[str]) -> List[dict]:
    from pyannote.audio import Pipeline

    token = hf_token or os.environ.get("HF_TOKEN")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=token
    )
    diarization = pipeline(audio_path)

    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    return turns


def assign_speakers(segments: List[Segment], turns: List[dict]) -> None:
    """Assign each transcript segment the speaker with maximum time overlap."""
    for seg in segments:
        best_speaker, best_overlap = None, 0.0
        for t in turns:
            overlap = min(seg.end, t["end"]) - max(seg.start, t["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = t["speaker"]
        seg.speaker = best_speaker or "UNKNOWN"


# ---------------------------------------------------------------------------
# Step 8: Output
# ---------------------------------------------------------------------------

def fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def write_outputs(segments: List[Segment], quality: dict, args, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]

    review_count = sum(1 for s in segments if s.needs_review)

    audit = {
        "case_id": args.case_id,
        "source_file": os.path.abspath(args.input),
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": {"pass1": FAST_MODEL, "pass2": ACCURATE_MODEL, "compute_type": COMPUTE_TYPE},
        "audio_quality": quality,
        "total_segments": len(segments),
        "segments_flagged_for_human_review": review_count,
        "thresholds": {
            "no_speech_prob": NOSPEECH_PROB_THRESHOLD,
            "avg_logprob": AVG_LOGPROB_THRESHOLD,
            "compression_ratio": COMPRESSION_RATIO_THRESHOLD,
        },
    }

    json_path = os.path.join(outdir, f"{base}_transcript.json")
    with open(json_path, "w") as f:
        json.dump({"audit": audit, "segments": [asdict(s) for s in segments]}, f, indent=2)

    txt_path = os.path.join(outdir, f"{base}_transcript.txt")
    with open(txt_path, "w") as f:
        f.write(f"FORENSIC TRANSCRIPT - Case {args.case_id}\n")
        f.write(f"Source: {audit['source_file']}\n")
        f.write(f"Processed: {audit['processed_at_utc']}\n")
        f.write(f"Audio quality: {quality['classification']} "
                f"(est. SNR {quality['estimated_snr_db']} dB)\n")
        f.write(f"Segments flagged for human review: {review_count} / {len(segments)}\n")
        f.write("=" * 70 + "\n\n")
        for seg in segments:
            flag = " [** REVIEW NEEDED **]" if seg.needs_review else ""
            f.write(f"[{fmt_ts(seg.start)} - {fmt_ts(seg.end)}] {seg.speaker}:{flag}\n")
            f.write(f"  {seg.text}\n")
            if seg.needs_review:
                f.write(f"  Reasons: {', '.join(seg.review_reason)}\n")
                if seg.pass2_text:
                    f.write(f"  Alt. reading (pass 2): {seg.pass2_text}\n")
            f.write("\n")

    review_path = os.path.join(outdir, f"{base}_review_queue.json")
    with open(review_path, "w") as f:
        json.dump([asdict(s) for s in segments if s.needs_review], f, indent=2)

    return json_path, txt_path, review_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Forensic call transcription pipeline")
    ap.add_argument("--input", required=True, help="Path to original audio file")
    ap.add_argument("--case-id", required=True, help="Case / exhibit ID for the audit log")
    ap.add_argument("--outdir", default="./output", help="Output directory")
    ap.add_argument("--hf-token", default=None, help="HuggingFace token for pyannote (or set HF_TOKEN env var)")
    ap.add_argument("--skip-diarization", action="store_true", help="Skip speaker diarization step")
    ap.add_argument("--fast-model", default=FAST_MODEL_DEFAULT,
                     help=f"Pass 1 model (default: {FAST_MODEL_DEFAULT}). "
                          f"Use 'small' or 'medium' on lighter machines like Codespaces.")
    ap.add_argument("--accurate-model", default=ACCURATE_MODEL_DEFAULT,
                     help=f"Pass 2 model (default: {ACCURATE_MODEL_DEFAULT}). "
                          f"Use 'medium' on lighter machines like Codespaces.")
    args = ap.parse_args()

    global FAST_MODEL, ACCURATE_MODEL
    FAST_MODEL = args.fast_model
    ACCURATE_MODEL = args.accurate_model

    if os.environ.get("PIPELINE_TEST_MODE") == "1":
        print("*** Running in TEST MODE (Codespaces/dev container). "
              "Do not process real evidentiary audio in this environment. ***\n",
              file=sys.stderr)

    t0 = time.time()

    print("[1/6] Checking audio quality...", file=sys.stderr)
    quality = check_audio_quality(args.input)
    print(f"      -> {quality}", file=sys.stderr)

    print("[2/6] Preprocessing audio...", file=sys.stderr)
    work_path = "/tmp/_forensic_pipeline_work.wav"
    preprocess_audio(args.input, work_path, degraded=(quality["classification"] == "degraded"))

    print(f"[3/6] Pass 1 transcription ({FAST_MODEL})...", file=sys.stderr)
    segments = transcribe_pass1(work_path)
    flag_low_confidence(segments)
    print(f"      -> {len(segments)} segments, "
          f"{sum(s.needs_review for s in segments)} flagged for pass 2", file=sys.stderr)

    print(f"[4/6] Pass 2 targeted re-check ({ACCURATE_MODEL})...", file=sys.stderr)
    transcribe_pass2(work_path, segments)

    if not args.skip_diarization:
        print("[5/6] Speaker diarization...", file=sys.stderr)
        try:
            turns = diarize(work_path, args.hf_token)
            assign_speakers(segments, turns)
        except Exception as e:
            print(f"      WARNING: diarization failed ({e}). "
                  f"Continuing without speaker labels.", file=sys.stderr)
            for s in segments:
                s.speaker = "UNKNOWN"
    else:
        for s in segments:
            s.speaker = "UNKNOWN"

    print("[6/6] Writing outputs...", file=sys.stderr)
    json_path, txt_path, review_path = write_outputs(segments, quality, args, args.outdir)

    elapsed = time.time() - t0
    review_count = sum(1 for s in segments if s.needs_review)
    print(f"\nDone in {elapsed:.1f}s. {len(segments)} segments, "
          f"{review_count} need human review.", file=sys.stderr)
    print(f"  Full transcript (JSON): {json_path}", file=sys.stderr)
    print(f"  Full transcript (TXT):  {txt_path}", file=sys.stderr)
    print(f"  Review queue only:      {review_path}", file=sys.stderr)


if __name__ == "__main__":
    main()