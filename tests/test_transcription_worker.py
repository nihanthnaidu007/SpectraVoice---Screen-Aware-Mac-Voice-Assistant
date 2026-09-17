"""Tests for the off-thread TranscriptionWorker (Wave 1 latency work).

Platform-independent: the Whisper recognizer is replaced by controlled stubs,
so these tests prove handback latency, error isolation, queue eviction, and
shutdown behavior without SpeechRecognition, PyAudio, or Whisper installed.
"""

import os
import sys
import threading
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))


from assistant_app.services.transcription import TranscriptionWorker


def wait_until(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


class TestHandback:
    def test_submit_returns_immediately_while_recognize_blocks(self):
        """The recognizer loop must regain control at once — this is the
        perceived-latency win of moving Whisper off the hot path."""
        release = threading.Event()

        def recognize(audio):
            release.wait(timeout=5)
            return "slow"

        worker = TranscriptionWorker(
            recognize=recognize,
            on_result=lambda text, job: None,
            on_error=lambda exc, job: None,
        ).start()

        start = time.time()
        accepted = worker.submit(object())
        handback = time.time() - start

        assert accepted is True
        assert handback < 0.1, f"submit() blocked for {handback * 1000:.0f} ms"
        worker.stop()


class TestCallbacks:
    def test_result_callback_receives_text_and_job(self):
        results = []
        worker = TranscriptionWorker(
            recognize=lambda audio: "hello world",
            on_result=lambda text, job: results.append((text, job)),
            on_error=lambda exc, job: None,
        ).start()

        worker.submit(object())
        assert wait_until(lambda: results)
        text, job = results[0]
        assert text == "hello world"
        assert job.submitted_at > 0
        worker.stop()

    def test_recognize_error_reaches_on_error_and_worker_survives(self):
        errors = []
        results = []
        calls = []

        def recognize(audio):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("model exploded")
            return "second ok"

        worker = TranscriptionWorker(
            recognize=recognize,
            on_result=lambda text, job: results.append(text),
            on_error=lambda exc, job: errors.append(exc),
        ).start()

        worker.submit(object())
        worker.submit(object())
        assert wait_until(lambda: worker.stats.completed >= 1 and worker.stats.failed >= 1)

        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
        assert results == ["second ok"]  # the worker thread survived
        worker.stop()

    def test_handler_exception_does_not_kill_worker(self):
        results = []

        def bad_handler(text, job):
            results.append(text)
            raise ValueError("handler bug")

        worker = TranscriptionWorker(
            recognize=lambda audio: "text",
            on_result=bad_handler,
            on_error=lambda exc, job: None,
        ).start()

        worker.submit(object())
        worker.submit(object())
        assert wait_until(lambda: worker.stats.completed >= 2)
        assert len(results) == 2  # handler ran for both; its exceptions were contained
        worker.stop()


class TestQueueEviction:
    def test_full_queue_evicts_oldest_utterance(self):
        """Speech arriving faster than it can be transcribed drops stale audio
        instead of growing memory or transcribing out of order."""
        started = threading.Event()
        release = threading.Event()
        dropped = []

        def recognize(audio):
            started.set()
            release.wait(timeout=5)
            return "text"

        worker = TranscriptionWorker(
            recognize=recognize,
            on_result=lambda text, job: None,
            on_error=lambda exc, job: None,
            on_drop=dropped.append,
            max_queue=1,
        ).start()

        # First job is picked up and blocks inside recognize.
        assert worker.submit("first") is True
        assert started.wait(timeout=5)

        # Queue is empty again (job in flight), so this fits (max_queue=1).
        assert worker.submit("second") is True
        # Queue is now full — this evicts the pending "second".
        assert worker.submit("third") is True

        assert wait_until(lambda: worker.stats.dropped == 1)
        release.set()
        assert wait_until(lambda: worker.stats.completed >= 2)

        assert worker.stats.submitted == 3
        assert worker.stats.dropped == 1
        assert len(dropped) == 1
        worker.stop()


class TestShutdown:
    def test_stop_finishes_in_flight_job(self):
        done = threading.Event()

        def recognize(audio):
            time.sleep(0.2)
            return "in flight"

        worker = TranscriptionWorker(
            recognize=recognize,
            on_result=lambda text, job: done.set(),
            on_error=lambda exc, job: None,
        ).start()

        worker.submit(object())
        worker.stop(timeout=5)  # must not return until the in-flight job finished

        assert done.is_set()
        assert worker.stats.completed == 1

    def test_stats_snapshot_is_isolated(self):
        worker = TranscriptionWorker(
            recognize=lambda audio: "x",
            on_result=lambda text, job: None,
            on_error=lambda exc, job: None,
        ).start()

        worker.submit(object())
        assert wait_until(lambda: worker.stats.completed == 1)

        snapshot = worker.stats
        snapshot.completed = 999  # mutating a snapshot must not corrupt real stats
        assert worker.stats.completed == 1
        worker.stop()
