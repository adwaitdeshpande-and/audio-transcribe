#!/usr/bin/env python3
"""
Generate synthetic test audio for pipeline testing.

Uses offline TTS (espeak-ng) so no real call recordings or external data are
needed to test the transcription/diarization pipeline in a cloud dev
environment like GitHub Codespaces.

Produces:
  - sample_clean.wav          : single speaker, clean audio
  - sample_two_speakers.wav   : two synthetic voices alternating (diarization test)
  - sample_noisy.wav          : same content with added noise (quality/denoise test)
"""

import argparse
import os
import subprocess
import wave

import numpy as np


SCRIPT_LINES = [
    ("en+m3", "Hello, this is a test of the forensic transcription pipeline."),
    ("en+f3", "Understood. I am the second speaker in this test recording."),
    ("en+m3", "The system should separate our voices and transcribe both accurately."),
    ("en+f3", "This sentence includes a number, forty two, and a name, Alex Johnson."),
    ("en+m3", "End of the synthetic test call."),
]


def synthesize_line(voice: str, text: str, out_path: str):
    subprocess.run(
        ["espeak-ng", "-v", voice, "-s", "150", "-w", out_path, text],
        check=True, capture_output=True,
    )


def concat_wavs(paths, out_path, gap_sec=0.4):
    datas = []
    params = None
    for p in paths:
        with wave.open(p, "rb") as w:
            params = w.getparams()
            frames = w.readframes(w.getnframes())
            datas.append(np.frombuffer(frames, dtype=np.int16))

    gap = np.zeros(int(gap_sec * params.framerate), dtype=np.int16)
    combined = []
    for i, d in enumerate(datas):
        combined.append(d)
        if i < len(datas) - 1:
            combined.append(gap)
    combined = np.concatenate(combined)

    with wave.open(out_path, "wb") as w:
        w.setparams(params)
        w.writeframes(combined.tobytes())


def add_noise(in_path: str, out_path: str, snr_db: float = 10.0):
    with wave.open(in_path, "rb") as w:
        params = w.getparams()
        frames = w.readframes(w.getnframes())
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)

    signal_power = np.mean(data ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), size=data.shape)
    noisy = np.clip(data + noise, -32768, 32767).astype(np.int16)

    with wave.open(out_path, "wb") as w:
        w.setparams(params)
        w.writeframes(noisy.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="./test_audio")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tmp = os.path.join(args.outdir, "_tmp")
    os.makedirs(tmp, exist_ok=True)

    line_paths = []
    for i, (voice, text) in enumerate(SCRIPT_LINES):
        p = os.path.join(tmp, f"line_{i}.wav")
        synthesize_line(voice, text, p)
        line_paths.append(p)

    two_speaker_path = os.path.join(args.outdir, "sample_two_speakers.wav")
    concat_wavs(line_paths, two_speaker_path)

    clean_path = os.path.join(args.outdir, "sample_clean.wav")
    concat_wavs(line_paths[:1] * 3, clean_path)  # single voice, repeated lines

    noisy_path = os.path.join(args.outdir, "sample_noisy.wav")
    add_noise(two_speaker_path, noisy_path, snr_db=8.0)

    print(f"Generated test audio in {args.outdir}:")
    for f in ["sample_clean.wav", "sample_two_speakers.wav", "sample_noisy.wav"]:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
