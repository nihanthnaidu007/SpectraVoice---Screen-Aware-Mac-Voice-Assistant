#!/usr/bin/env python3
"""Measure recognizer-thread handback before/after W1's off-thread transcription.

Whisper is stubbed with a fixed sleep (default 0.8 s — roughly whisper `base`
on Apple Silicon), so this benchmark isolates the pipeline shape from the ML
model and runs deterministically on any machine, no audio required:

    python scripts/benchmark_transcription_latency.py --transcribe-seconds 0.8

What is measured:
1. Handback — how long the audio callback holds the recognizer thread per
   utterance. Inline (pre-W1) it blocks for the whole transcription; with the
   TranscriptionWorker it returns as soon as the job is queued.
2. Burst wall-clock — two near-simultaneous utterances. Pre-W1 they transcribed
   on parallel callback threads; the worker serializes them (bounded memory,
   ordered responses) at the cost of the second utterance waiting its turn.
"""

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from assistant_app.services.transcription import TranscriptionWorker


def bench_inline_handback(transcribe_seconds: float, utterances: int) -> list[float]:
    """Pre-W1 path: recognize_whisper runs inside the audio callback."""
    handbacks = []
    for _ in range(utterances):
        cb_start = time.time()
        time.sleep(transcribe_seconds)  # stubbed recognize_whisper
        handbacks.append(time.time() - cb_start)
    return handbacks


def bench_worker_handback(transcribe_seconds: float, utterances: int) -> list[float]:
    """W1 path: submit-and-return; Whisper runs on the dedicated worker."""

    def recognize(audio):
        time.sleep(transcribe_seconds)
        return "ok"

    worker = TranscriptionWorker(
        recognize=recognize,
        on_result=lambda text, job: None,
        on_error=lambda exc, job: None,
        max_queue=utterances + 1,
    ).start()
    handbacks = []
    for _ in range(utterances):
        cb_start = time.time()
        worker.submit(object())
        handbacks.append(time.time() - cb_start)
    while worker.stats.completed < utterances:
        time.sleep(0.005)
    worker.stop()
    return handbacks


def bench_burst(transcribe_seconds: float, concurrent: bool, n: int = 2) -> float:
    """Wall-clock to finish transcribing n near-simultaneous utterances.

    concurrent=True replicates the pre-W1 shape (a callback thread per
    utterance, Whisper runs overlap); concurrent=False is the W1 worker
    (serialized).
    """
    if concurrent:
        threads = []
        start = time.time()
        for _ in range(n):
            t = threading.Thread(target=lambda: time.sleep(transcribe_seconds))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        return time.time() - start

    def recognize(audio):
        time.sleep(transcribe_seconds)
        return "ok"

    worker = TranscriptionWorker(
        recognize=recognize,
        on_result=lambda text, job: None,
        on_error=lambda exc, job: None,
        max_queue=n,
    ).start()
    start = time.time()
    for _ in range(n):
        worker.submit(object())
    while worker.stats.completed < n:
        time.sleep(0.005)
    worker.stop()
    return time.time() - start


def fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f} ms"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transcribe-seconds", type=float, default=0.8,
        help="Simulated Whisper latency per utterance (default 0.8 ≈ `base` on Apple Silicon)",
    )
    parser.add_argument("--utterances", type=int, default=5, help="Handback sample count")
    args = parser.parse_args()

    inline_handbacks = bench_inline_handback(args.transcribe_seconds, args.utterances)
    worker_handbacks = bench_worker_handback(args.transcribe_seconds, args.utterances)
    inline_burst = bench_burst(args.transcribe_seconds, concurrent=True)
    worker_burst = bench_burst(args.transcribe_seconds, concurrent=False)

    print(f"Simulated Whisper latency: {args.transcribe_seconds:.2f} s, {args.utterances} utterances")
    print()
    print("Recognizer-thread handback per utterance (lower is better):")
    print(f"  inline (pre-W1):      {fmt_ms(sum(inline_handbacks) / len(inline_handbacks))} avg "
          f"(min {fmt_ms(min(inline_handbacks))}, max {fmt_ms(max(inline_handbacks))})")
    print(f"  worker (W1):          {fmt_ms(sum(worker_handbacks) / len(worker_handbacks))} avg "
          f"(min {fmt_ms(min(worker_handbacks))}, max {fmt_ms(max(worker_handbacks))})")
    print()
    print("Burst of 2 utterances — wall-clock to finish all transcriptions:")
    print(f"  inline (pre-W1, parallel Whisper runs): {fmt_ms(inline_burst)}")
    print(f"  worker (W1, serialized):                {fmt_ms(worker_burst)}")
    print()
    print("Note: the inline burst wall-clock excludes the real-world cost the worker")
    print("removes — overlapping Whisper loads contending for CPU/memory and out-of-order responses.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
