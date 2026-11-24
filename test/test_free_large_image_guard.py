"""Offline tests for the free-pool large-image heavy-memory guard.

These tests do not call a live model, paid generation, or the shared size proxy.
They drive the real guard through stream materialization and `_generate_single_image`.
"""

from __future__ import annotations

import asyncio
import base64
import os
import random
import struct
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from services import free_large_image_guard as guard
from services.free_large_image_guard import (
    LARGE_PIXEL_THRESHOLD,
    LargeImageAdmissionError,
    image_inputs_are_large,
    is_large_size,
    read_stream_guarded,
    reset_for_tests,
    sniff_image_size,
)


def _png(width: int, height: int) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + ihdr + b"\x00\x00\x00\x00"


class GuardTestCase(unittest.TestCase):
    def setUp(self) -> None:
        os.environ.pop("CHATGPT2API_FREE_LARGE_IMAGE_GUARD", None)
        reset_for_tests(enabled=True)
        guard.configure_limits(
            max_active=5,
            max_waiting=10,
            wait_timeout_secs=2.0,
            admission_wait_secs=0,
        )

    def tearDown(self) -> None:
        os.environ.pop("CHATGPT2API_FREE_LARGE_IMAGE_GUARD", None)
        stats = guard.stats()
        problem = guard.consistency()
        reset_for_tests(enabled=False)
        self.assertEqual(stats["active"], 0, stats)
        self.assertEqual(stats["waiting"], 0, stats)
        self.assertEqual(stats["generating"], 0, stats)
        self.assertEqual(stats["admission_waiting"], 0, stats)
        self.assertIsNone(problem, stats)


class ClassificationTests(GuardTestCase):
    def test_known_sizes(self) -> None:
        small = [
            "1024x1024",
            "1024X1024",
            "1254x1254",
            "1024x1536",
            "1536x1024",
            "1K",
            "1k",
            "auto",
            "",
            None,
            "not-a-size",
            "10",
            "0x0",
            "-1x1024",
            "999999x10",
        ]
        for size in small:
            self.assertFalse(is_large_size(size), size)
        self.assertFalse(is_large_size((1024, 1024)))
        self.assertEqual(1024 * 1536, 1_572_864)
        self.assertLessEqual(1024 * 1536, LARGE_PIXEL_THRESHOLD)
        self.assertEqual(1254 * 1254, 1_572_516)
        self.assertLessEqual(1254 * 1254, LARGE_PIXEL_THRESHOLD)

        large = [
            "2048x2048",
            "2048X2048",
            " 2048 x 2048 ",
            "3840x2160",
            "3840×2160",
            "4096x4096",
            "2K",
            "2k",
            "4K",
            "4k",
            "size=3840x2160",
            "1792x1024",
        ]
        for size in large:
            self.assertTrue(is_large_size(size), size)
        self.assertEqual(3840 * 2160, 8_294_400)
        self.assertGreater(3840 * 2160, LARGE_PIXEL_THRESHOLD)
        self.assertGreater(2048 * 2048, LARGE_PIXEL_THRESHOLD)
        self.assertGreater(4096 * 4096, LARGE_PIXEL_THRESHOLD)
        # Just under / just over the published 1K boundary.
        self.assertFalse(is_large_size(f"1304x{LARGE_PIXEL_THRESHOLD // 1304}"))
        self.assertTrue(is_large_size("1304x1304"))
        self.assertGreater(1304 * 1304, LARGE_PIXEL_THRESHOLD)
        # n copies of a 1K frame are still not one large frame.
        self.assertFalse(is_large_size("1024x1024"))

    def test_header_sniff_does_not_need_a_full_decode(self) -> None:
        large = _png(3840, 2160)
        small = _png(1024, 1024)
        self.assertEqual(sniff_image_size(large), (3840, 2160))
        self.assertEqual(sniff_image_size(small), (1024, 1024))
        self.assertTrue(image_inputs_are_large([(large, "a.png", "image/png")]))
        self.assertFalse(image_inputs_are_large([(small, "a.png", "image/png")]))
        jpeg = b"\xff\xd8\xff\xc0\x00\x11\x08" + struct.pack(">HH", 2160, 3840)
        self.assertEqual(sniff_image_size(jpeg), (3840, 2160))
        sample = base64.b64encode(large).decode("ascii")
        self.assertGreater(guard.b64_sample_pixels(sample) or 0, LARGE_PIXEL_THRESHOLD)
        self.assertLess(guard.b64_sample_pixels(base64.b64encode(small).decode()) or 0, LARGE_PIXEL_THRESHOLD)
        bmp_large = _bmp(3840, 2160)
        bmp_small = _bmp(1024, 1024)
        self.assertEqual(sniff_image_size(bmp_large), (3840, 2160))
        self.assertEqual(sniff_image_size(bmp_small), (1024, 1024))
        self.assertTrue(image_inputs_are_large([(bmp_large, "a.bmp", "image/bmp")]))
        self.assertFalse(image_inputs_are_large([(bmp_small, "a.bmp", "image/bmp")]))
        self.assertEqual(guard.classify_image_header(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)[0], "incomplete")
        self.assertNotEqual(guard.classify_image_header(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40)[0], "small")
        self.assertEqual(guard.classify_image_header(b"????" * 16)[0], "unknown")
        self.assertTrue(guard.header_needs_guard(b"????" * 16))
        self.assertTrue(guard.header_needs_guard(b"BM" + b"\x00" * 8))
        self.assertIsNotNone(guard.b64_guard_pixels(base64.b64encode(bmp_large).decode(), "1024x1024"))
        self.assertIsNone(guard.b64_guard_pixels(base64.b64encode(small).decode(), "1024x1024"))

    def test_stream_acquires_before_the_rest_of_a_large_body(self) -> None:
        started = threading.Event()

        def chunks():
            yield _png(3840, 2160)[:32]
            self.assertGreaterEqual(guard.stats()["active"], 1)
            started.set()
            yield b"tail"

        data = read_stream_guarded(chunks(), stage="test")
        self.assertTrue(started.is_set())
        self.assertTrue(data.startswith(b"\x89PNG"))
        self.assertEqual(guard.stats()["active"], 0)
        self.assertLessEqual(guard.stats()["peak_active"], 5)

    def test_ordinary_1k_stream_does_not_take_a_permit_even_with_a_large_length(self) -> None:
        def chunks():
            yield _png(1024, 1536)
            self.assertEqual(guard.stats()["active"], 0)
            yield b"rest"

        read_stream_guarded(chunks(), content_length=9_000_000, stage="test")
        self.assertEqual(guard.stats()["peak_active"], 0)

    def test_unknown_huge_length_is_guarded_before_the_rest(self) -> None:
        def chunks():
            yield b"????" * 16
            self.assertGreaterEqual(guard.stats()["active"], 1)
            yield b"more"

        read_stream_guarded(chunks(), content_length=guard.UNKNOWN_LARGE_BYTES, stage="test")
        self.assertEqual(guard.stats()["active"], 0)


class ConcurrencyTests(GuardTestCase):
    def _hold_heavy(self, release: threading.Event, entered: list[int], errors: list[BaseException]) -> None:
        try:
            with guard.heavy_image_work(declared_size="3840x2160", stage="test"):
                with threading.Lock():
                    entered.append(1)
                release.wait(timeout=3)
        except BaseException as exc:  # noqa: BLE001 - tests collect worker failures
            errors.append(exc)

    def test_twenty_five_jobs_peak_at_five_and_sixth_waits(self) -> None:
        release = threading.Event()
        entered: list[int] = []
        errors: list[BaseException] = []
        rejected: list[str] = []
        lock = threading.Lock()

        def worker() -> None:
            try:
                with guard.heavy_image_work(declared_size="3840x2160", stage="wave"):
                    with lock:
                        entered.append(1)
                    release.wait(timeout=3)
            except LargeImageAdmissionError as exc:
                with lock:
                    rejected.append(exc.reason)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, name=f"user-{index % 5}-acct-{index}") for index in range(25)]
        for thread in threads:
            thread.start()
        self.assertTrue(self._wait_until(lambda: len(entered) == 5 and guard.stats()["active"] == 5))
        time.sleep(0.05)
        self.assertEqual(len(entered), 5, "sixth job performed heavy work")
        self.assertGreaterEqual(guard.stats()["waiting"], 1)
        self.assertLessEqual(guard.stats()["peak_active"], 5)
        self.assertTrue(self._wait_until(lambda: len(rejected) >= 1))
        self.assertIn("queue_full", rejected)
        release.set()
        for thread in threads:
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
        self.assertFalse(errors, errors)
        self.assertLessEqual(guard.stats()["peak_active"], 5)
        self.assertEqual(len(entered) + len(rejected), 25)
        self.assertGreaterEqual(len(entered), 6)

    def test_queue_full_and_timeout_do_not_start_work(self) -> None:
        release = threading.Event()
        entered: list[int] = []
        errors: list[BaseException] = []
        holders = [
            threading.Thread(target=self._hold_heavy, args=(release, entered, errors), name=f"hold-{i}")
            for i in range(5)
        ]
        for thread in holders:
            thread.start()
        self.assertTrue(self._wait_until(lambda: guard.stats()["active"] == 5))
        # Five heavy jobs do not block a new upstream generation.
        with guard.bind_large_image_job("4K"):
            self.assertEqual(guard.stats()["generating"], 1)
            self.assertEqual(guard.stats()["heavy_waiting"], 0)
        self.assertEqual(guard.stats()["generating"], 0)
        guard.configure_limits(wait_timeout_secs=0.2)
        started = time.time()
        with self.assertRaises(LargeImageAdmissionError) as timed:
            with guard.heavy_image_work(declared_size="4K", stage="heavy-timeout"):
                entered.append(1)
        self.assertEqual(timed.exception.reason, "queue_timeout")
        self.assertLess(time.time() - started, 1.5)
        self.assertEqual(len(entered), 5)
        blocked_errors: list[BaseException] = []
        blocked_entered: list[int] = []

        def blocked() -> None:
            try:
                with guard.heavy_image_work(declared_size="2048x2048", stage="wait"):
                    blocked_entered.append(1)
                    release.wait(timeout=3)
            except BaseException as exc:  # noqa: BLE001
                blocked_errors.append(exc)

        guard.configure_limits(wait_timeout_secs=2.0)
        waiters = [threading.Thread(target=blocked, name=f"wait-{i}") for i in range(10)]
        for thread in waiters:
            thread.start()
        self.assertTrue(self._wait_until(lambda: guard.stats()["waiting"] >= 10))
        with self.assertRaises(LargeImageAdmissionError) as caught:
            with guard.heavy_image_work(declared_size="4096x4096", stage="overflow"):
                blocked_entered.append(1)
        self.assertEqual(caught.exception.reason, "queue_full")
        self.assertEqual(blocked_entered, [])
        release.set()
        for thread in holders + waiters:
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
        self.assertFalse(errors, errors)
        self.assertFalse(blocked_errors, blocked_errors)
        self.assertEqual(len(blocked_entered), 10)

    def test_cancel_queued_job_never_starts(self) -> None:
        release = threading.Event()
        entered: list[int] = []
        errors: list[BaseException] = []
        holder = threading.Thread(target=self._hold_heavy, args=(release, entered, errors))
        # Fill all five permits.
        holders = [holder] + [
            threading.Thread(target=self._hold_heavy, args=(release, entered, errors))
            for _ in range(4)
        ]
        for thread in holders:
            thread.start()
        self.assertTrue(self._wait_until(lambda: guard.stats()["active"] == 5))
        started_heavy: list[str] = []
        result: list[str] = []

        def queued() -> None:
            try:
                with guard.heavy_image_work(declared_size="3840x2160", stage="queued"):
                    started_heavy.append("yes")
            except LargeImageAdmissionError as exc:
                result.append(exc.reason)

        worker = threading.Thread(target=queued)
        worker.start()
        token_id = None
        deadline = time.time() + 2
        while time.time() < deadline and token_id is None:
            for item in guard.token_snapshot():
                if item["state"] == "heavy_waiting":
                    token_id = item["id"]
                    break
            time.sleep(0.01)
        self.assertIsNotNone(token_id)
        self.assertTrue(guard.cancel_token(int(token_id)))
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(result, ["cancelled"])
        self.assertEqual(started_heavy, [])
        release.set()
        for thread in holders:
            thread.join(timeout=3)
        time.sleep(0.05)
        self.assertEqual(started_heavy, [])
        self.assertFalse(errors, errors)

    def test_exception_releases_and_cancel_does_not_release_early(self) -> None:
        try:
            with guard.heavy_image_work(declared_size="4096x4096", stage="boom"):
                raise RuntimeError("corrupt image")
        except RuntimeError:
            pass
        self.assertEqual(guard.stats()["active"], 0)
        self.assertEqual(guard.stats()["waiting"], 0)

        release = threading.Event()
        inside = threading.Event()
        holder: dict[str, object] = {}

        def running() -> None:
            with guard.heavy_image_work(declared_size="2K", stage="held") as token:
                holder["token"] = token
                inside.set()
                release.wait(timeout=3)

        thread = threading.Thread(target=running)
        thread.start()
        self.assertTrue(inside.wait(timeout=2))
        token = holder["token"]
        self.assertIsNotNone(token)
        self.assertFalse(token.cancel())  # type: ignore[union-attr]
        self.assertEqual(guard.stats()["active"], 1)
        release.set()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(guard.stats()["active"], 0)

    def test_asyncio_cancel_does_not_drop_the_permit_early(self) -> None:
        started = threading.Event()
        finished = threading.Event()
        really_done = threading.Event()

        def blocking() -> str:
            try:
                with guard.heavy_image_work(declared_size="3840x2160", stage="async"):
                    started.set()
                    finished.wait(timeout=3)
                return "done"
            finally:
                really_done.set()

        async def scenario() -> None:
            loop = asyncio.get_running_loop()
            task = loop.run_in_executor(None, blocking)
            self.assertTrue(started.wait(timeout=2))
            self.assertEqual(guard.stats()["active"], 1)
            task.cancel()
            await asyncio.sleep(0.05)
            self.assertEqual(guard.stats()["active"], 1)
            finished.set()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(scenario())
        self.assertTrue(really_done.wait(timeout=2))
        self.assertEqual(guard.stats()["active"], 0)

    def test_small_jobs_health_and_polling_do_not_use_large_permits(self) -> None:
        release = threading.Event()
        entered: list[int] = []
        errors: list[BaseException] = []
        holders = [
            threading.Thread(target=self._hold_heavy, args=(release, entered, errors))
            for _ in range(5)
        ]
        for thread in holders:
            thread.start()
        self.assertTrue(self._wait_until(lambda: guard.stats()["active"] == 5))
        small_inside = threading.Barrier(8)
        small_peak = {"value": 0}
        small_now = {"value": 0}
        small_lock = threading.Lock()

        def small() -> None:
            with guard.heavy_image_work(declared_size="1024x1024", stage="small"):
                with small_lock:
                    small_now["value"] += 1
                    small_peak["value"] = max(small_peak["value"], small_now["value"])
                small_inside.wait(timeout=2)
                with small_lock:
                    small_now["value"] -= 1

        small_threads = [threading.Thread(target=small, name=f"small-{i}") for i in range(8)]
        for thread in small_threads:
            thread.start()
        for thread in small_threads:
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
        self.assertGreaterEqual(small_peak["value"], 6)
        self.assertEqual(guard.stats()["active"], 5)
        from services.image_task_service import ImageTaskService

        service = ImageTaskService(
            Path(self.id().replace(".", "_") + "-tasks.json"),
            generation_handler=lambda _payload: {"data": [{"url": "http://example.test/small.png"}]},
            retention_days_getter=lambda: 30,
        )
        try:
            service.submit_generation(
                {"id": "owner-a", "name": "A", "role": "user"},
                client_task_id="small-1",
                prompt="cat",
                model="gpt-image-2",
                size="1024x1024",
                base_url="http://local.test",
            )
            deadline = time.time() + 2
            status = ""
            while time.time() < deadline:
                item = service.list_tasks({"id": "owner-a"}, ["small-1"])["items"][0]
                status = str(item["status"])
                if status == "success":
                    break
                time.sleep(0.02)
            self.assertEqual(status, "success")
            self.assertEqual(guard.stats()["active"], 5)
            snapshot = guard.stats()
            self.assertNotIn("prompt", snapshot)
            self.assertNotIn("key", snapshot)
            self.assertFalse(snapshot["cross_process"])
            self.assertEqual(snapshot["max_active"], 5)
        finally:
            path = service.path
            release.set()
            for thread in holders:
                thread.join(timeout=3)
            path.unlink(missing_ok=True)

    def _wait_until(self, predicate, timeout: float = 2.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return False


class IntegrationTests(GuardTestCase):
    def test_generate_single_image_blocks_before_upstream_and_does_not_duplicate(self) -> None:
        from services.protocol.conversation import (
            ConversationRequest,
            ImageGenerationError,
            _generate_single_image,
        )

        release = threading.Event()
        entered: list[int] = []
        errors: list[BaseException] = []
        holders = [
            threading.Thread(target=self._occupy, args=(release, entered, errors))
            for _ in range(5)
        ]
        for thread in holders:
            thread.start()
        self.assertTrue(ConcurrencyTests._wait_until(self, lambda: guard.stats()["active"] == 5))
        calls = {"token": 0, "stream": 0, "active": -1}

        def fake_stream(backend, request, index=1, total=1):
            calls["stream"] += 1
            from services.protocol.conversation import _materialize_image_output

            output = _materialize_image_output(
                backend, request, ["http://example.test/a.png"], index, total, "conv-1",
            )
            return [output] if output is not None else []

        class FakeBackend:
            def __init__(self, access_token: str = "", **_kwargs: object) -> None:
                self.access_token = access_token
                self.progress_callback = None

            def close(self) -> None:
                pass

            def download_image_bytes(self, urls, size=None):
                calls["active"] = guard.stats()["active"]
                return [_png(32, 32)]

        fake_backend = FakeBackend()

        def get_token(**_kwargs: object) -> str:
            calls["token"] += 1
            return "token-user-a"

        with mock.patch("services.protocol.conversation.save_image_bytes", return_value="http://example.test/out.png"), \
                mock.patch("services.protocol.conversation.stream_image_outputs", fake_stream), \
                mock.patch("services.protocol.conversation.OpenAIBackendAPI", FakeBackend), \
                mock.patch("services.protocol.conversation.account_service.get_available_access_token", get_token), \
                mock.patch("services.protocol.conversation.account_service.get_account", return_value={"email": "a@example.test"}), \
                mock.patch("services.protocol.conversation.account_service.mark_image_result", return_value=None), \
                mock.patch("services.protocol.conversation.account_service.release_image_slot", return_value=None):
            worker_done = threading.Event()
            outcome: dict[str, object] = {}

            def run_one() -> None:
                try:
                    outcome["value"] = _generate_single_image(
                        ConversationRequest(model="gpt-image-2", prompt="p", size="3840x2160", n=1),
                        1,
                        1,
                    )
                except BaseException as exc:  # noqa: BLE001
                    outcome["error"] = exc
                finally:
                    worker_done.set()

            worker = threading.Thread(target=run_one, name="acct-b")
            worker.start()
            time.sleep(0.1)
            # Heavy memory being full does not by itself block upstream admission.
            self.assertEqual(calls["token"], 1)
            self.assertEqual(calls["stream"], 1)
            self.assertFalse(worker_done.is_set())
            release.set()
            self.assertTrue(worker_done.wait(timeout=3))
            worker.join(timeout=2)
            self.assertNotIn("error", outcome, outcome.get("error"))
            self.assertEqual(calls["token"], 1)
            self.assertEqual(calls["stream"], 1)
            self.assertGreaterEqual(calls["active"], 1)

            # Queue is clear. A post-generation heavy timeout must not call upstream twice.
            calls["stream"] = 0
            calls["token"] = 0

            def timing_out_stream(backend, request, index=1, total=1):
                calls["stream"] += 1
                raise LargeImageAdmissionError("queue_timeout")

            with mock.patch("services.protocol.conversation.stream_image_outputs", timing_out_stream):
                with self.assertRaises(ImageGenerationError) as caught:
                    _generate_single_image(
                        ConversationRequest(model="gpt-image-2", prompt="p", size="2048x2048", n=1),
                        1,
                        1,
                    )
            self.assertEqual(caught.exception.status_code, 429)
            self.assertEqual(caught.exception.code, "free_large_image_queue_timeout")
            self.assertEqual(calls["stream"], 1)
            self.assertEqual(calls["token"], 1)

        for thread in holders:
            thread.join(timeout=3)
        self.assertFalse(errors, errors)
        self.assertIsNotNone(fake_backend)

    def _occupy(self, release: threading.Event, entered: list[int], errors: list[BaseException]) -> None:
        ConcurrencyTests._hold_heavy(self, release, entered, errors)

    def test_parallel_large_branches_share_five_slots(self) -> None:
        from services.protocol.conversation import ConversationRequest, stream_image_outputs_with_pool

        current = {"n": 0, "peak": 0}
        lock = threading.Lock()
        calls = {"stream": 0}

        def fake_stream(backend, request, index=1, total=1):
            with lock:
                calls["stream"] += 1
            with guard.heavy_image_work(declared_size=request.size, stage="pool"):
                with lock:
                    current["n"] += 1
                    current["peak"] = max(current["peak"], current["n"])
                time.sleep(0.15)
                with lock:
                    current["n"] -= 1
            from services.protocol.conversation import ImageOutput

            return [ImageOutput(kind="result", model=request.model, index=index, total=total, data=[{"url": "u"}])]

        class FakeBackend:
            def __init__(self, access_token: str = "", **_kwargs: object) -> None:
                self.progress_callback = None

            def close(self) -> None:
                pass

        with mock.patch("services.protocol.conversation.stream_image_outputs", fake_stream), \
                mock.patch("services.protocol.conversation.OpenAIBackendAPI", FakeBackend), \
                mock.patch("services.protocol.conversation.account_service.get_available_access_token", return_value="tok"), \
                mock.patch("services.protocol.conversation.account_service.get_account", return_value={"email": "a@example.test"}), \
                mock.patch("services.protocol.conversation.account_service.mark_image_result", return_value=None):
            # n is capped at 4 by the API, so run two pooled requests across users.
            threads = []
            errors: list[BaseException] = []

            def run(size_owner: str) -> None:
                try:
                    list(stream_image_outputs_with_pool(ConversationRequest(
                        model="gpt-image-2",
                        prompt=size_owner,
                        size="2048x2048",
                        n=4,
                    )))
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            for name in ("user-a", "user-b"):
                thread = threading.Thread(target=run, args=(name,))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join(timeout=5)
                self.assertFalse(thread.is_alive())
        self.assertFalse(errors, errors)
        self.assertLessEqual(current["peak"], 5)
        self.assertGreaterEqual(current["peak"], 2)
        self.assertEqual(calls["stream"], 8)
        self.assertGreaterEqual(guard.stats()["peak_active"], 2)
        self.assertLessEqual(guard.stats()["peak_active"], 5)

    def test_materialize_uses_real_download_guard(self) -> None:
        from services.protocol import conversation

        seen = {"active_during_download": -1}

        class Backend:
            progress_callback = None

            def download_image_bytes(self, urls, size=None):
                seen["active_during_download"] = guard.stats()["active"]
                return [_png(16, 16)]

        with mock.patch("services.protocol.conversation.save_image_bytes", return_value="http://example.test/out.png"):
            output = conversation._materialize_image_output(
                Backend(),
                conversation.ConversationRequest(
                    model="gpt-image-2",
                    prompt="p",
                    size="3840x2160",
                    response_format="b64_json",
                ),
                ["http://example.test/a.png"],
                1,
                1,
                "conv",
            )
        self.assertIsNotNone(output)
        assert output is not None
        self.assertGreaterEqual(seen["active_during_download"], 1)
        self.assertTrue(output.data[0]["b64_json"])
        self.assertEqual(output.data[0]["url"], "http://example.test/out.png")
        self.assertEqual(guard.stats()["active"], 0)

    def test_edit_corrupt_input_does_not_leak_and_decode_is_inside_the_permit(self) -> None:
        from services.protocol import openai_v1_image_edit

        seen = {"active": -1}

        def boom(*_args, **_kwargs):
            seen["active"] = guard.stats()["active"]
            raise RuntimeError("corrupt")

        payload = _png(3840, 2160) + b"not-pixels"
        with mock.patch("services.protocol.openai_v1_image_edit.Image.open", boom):
            with self.assertRaises(RuntimeError):
                openai_v1_image_edit.handle({
                    "prompt": "edit",
                    "images": [(payload, "big.png", "image/png")],
                    "mask": [(payload, "mask.png", "image/png")],
                    "model": "gpt-image-2",
                    "size": "1024x1024",
                })
        self.assertGreaterEqual(seen["active"], 1)
        self.assertEqual(guard.stats()["active"], 0)
        self.assertEqual(guard.stats()["waiting"], 0)

    def test_resume_poll_does_not_generate_again(self) -> None:
        from services.image_task_service import ImageTaskService
        from services.protocol import conversation

        downloads = {"n": 0, "active": -1}
        generations = {"n": 0}

        class FakeBackend:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def _poll_image_results(self, *_args, **_kwargs):
                return ["file"], []

            def resolve_conversation_image_urls(self, *_args, **_kwargs):
                return ["http://example.test/a.png"]

            def download_image_bytes(self, urls, size=None):
                downloads["n"] += 1
                downloads["active"] = guard.stats()["active"]
                return [_png(8, 8)]

            def close(self) -> None:
                pass

        real_generate = conversation._generate_single_image

        def spy_generate(*args, **kwargs):
            generations["n"] += 1
            return real_generate(*args, **kwargs)

        service = ImageTaskService(Path("test-resume-large.json"), retention_days_getter=lambda: 30)
        try:
            with mock.patch("services.openai_backend_api.OpenAIBackendAPI", FakeBackend), \
                    mock.patch("services.config.config.proxy_url", "", create=True), \
                    mock.patch("services.protocol.conversation._generate_single_image", spy_generate), \
                    mock.patch("services.protocol.conversation.save_image_bytes", return_value="http://example.test/resumed.png"):
                service._tasks["owner:task"] = {
                    "id": "task",
                    "owner_id": "owner",
                    "status": "error",
                    "mode": "generate",
                    "model": "gpt-image-2",
                    "size": "3840x2160",
                    "quality": "auto",
                }
                service._run_resume_poll("owner:task", "conv-1", 30.0, {"id": "owner"}, "generate", "gpt-image-2")
            task = service._tasks["owner:task"]
            self.assertEqual(task["status"], "success", task.get("error"))
            self.assertEqual(downloads["n"], 1)
            self.assertGreaterEqual(downloads["active"], 1)
            self.assertEqual(generations["n"], 0)
        finally:
            service.path.unlink(missing_ok=True)

    def test_same_async_task_poll_does_not_generate_twice(self) -> None:
        from services.image_task_service import ImageTaskService
        from services.protocol.conversation import ConversationRequest, _generate_single_image

        calls = {"n": 0}

        def handler(payload):
            calls["n"] += 1
            _generate_single_image(
                ConversationRequest(model="gpt-image-2", prompt="p", size=str(payload.get("size") or ""), n=1),
                1,
                1,
            )
            return {"data": [{"url": "http://example.test/same.png"}]}

        class FakeBackend:
            def __init__(self, access_token: str = "", **_kwargs: object) -> None:
                self.progress_callback = None

            def close(self) -> None:
                pass

        def fake_stream(backend, request, index=1, total=1):
            from services.protocol.conversation import ImageOutput

            return [ImageOutput(kind="result", model=request.model, index=index, total=total, data=[{"url": "u"}])]

        path = Path("test-same-task.json")
        service = ImageTaskService(path, generation_handler=handler, retention_days_getter=lambda: 30)
        identity = {"id": "owner-z", "name": "Z", "role": "user"}
        try:
            with mock.patch("services.protocol.conversation.stream_image_outputs", fake_stream), \
                    mock.patch("services.protocol.conversation.OpenAIBackendAPI", FakeBackend), \
                    mock.patch("services.protocol.conversation.account_service.get_available_access_token", return_value="tok"), \
                    mock.patch("services.protocol.conversation.account_service.get_account", return_value={"email": "z@example.test"}), \
                    mock.patch("services.protocol.conversation.account_service.mark_image_result", return_value=None):
                first = service.submit_generation(
                    identity,
                    client_task_id="same",
                    prompt="cat",
                    model="gpt-image-2",
                    size="2048x2048",
                    base_url="http://local.test",
                )
                for _ in range(20):
                    service.list_tasks(identity, ["same"])
                    service.submit_generation(
                        identity,
                        client_task_id="same",
                        prompt="cat",
                        model="gpt-image-2",
                        size="2048x2048",
                        base_url="http://local.test",
                    )
                deadline = time.time() + 3
                status = ""
                while time.time() < deadline:
                    status = service.list_tasks(identity, ["same"])["items"][0]["status"]
                    if status in {"success", "error"}:
                        break
                    time.sleep(0.02)
            self.assertEqual(first["id"], "same")
            self.assertEqual(status, "success")
            self.assertEqual(calls["n"], 1)
        finally:
            path.unlink(missing_ok=True)

    def test_guard_is_off_without_free_pool_and_small_jobs_are_uncapped(self) -> None:
        reset_for_tests(enabled=False)
        self.assertFalse(guard.is_enabled())
        peak = {"n": 0, "now": 0}
        lock = threading.Lock()
        barrier = threading.Barrier(8)

        def work() -> None:
            with guard.heavy_image_work(declared_size="4096x4096", stage="off"):
                with lock:
                    peak["now"] += 1
                    peak["n"] = max(peak["n"], peak["now"])
                barrier.wait(timeout=2)
                with lock:
                    peak["now"] -= 1

        threads = [threading.Thread(target=work) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)
        self.assertGreaterEqual(peak["n"], 6)
        self.assertEqual(guard.stats()["peak_active"], 0)
        guard._guard.configured = False
        os.environ["CHATGPT2API_FREE_LARGE_IMAGE_GUARD"] = "0"
        guard.install_free_large_image_guard(True)
        self.assertFalse(guard.is_enabled())
        os.environ.pop("CHATGPT2API_FREE_LARGE_IMAGE_GUARD", None)
        guard._guard.configured = False
        self.assertFalse(guard.is_enabled())
        from services.config import config

        self.assertEqual(config.image_account_concurrency, 3)

    def test_health_json_exposes_only_aggregate_counters(self) -> None:
        from fastapi.testclient import TestClient

        from api.app import create_app

        with guard.heavy_image_work(declared_size="2048x2048", stage="health"):
            with TestClient(create_app()) as client:
                response = client.get("/health?format=json")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        section = body["free_large_image_guard"]
        self.assertEqual(section["max_active"], 5)
        self.assertIn("active", section)
        self.assertIn("waiting", section)
        self.assertIn("heavy_waiting", section)
        self.assertIn("generating", section)
        self.assertEqual(section["waiting"], section["heavy_waiting"])
        self.assertIn("rejected", section)
        self.assertIn("peak_active", section)
        self.assertNotIn("prompt", section)
        self.assertNotIn("authorization", section)
        self.assertFalse(section["cross_process"])


class GenerationQueueTests(GuardTestCase):
    """Upstream generation must not consume the heavy-memory queue."""

    def test_ten_generations_with_idle_heavy_still_admit(self) -> None:
        release = threading.Event()
        started = []
        errors: list[BaseException] = []
        lock = threading.Lock()

        def occupy() -> None:
            try:
                with guard.bind_large_image_job("2048x2048"):
                    with lock:
                        started.append(1)
                    release.wait(timeout=3)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=occupy, name=f"gen-{i}") for i in range(10)]
        for thread in threads:
            thread.start()
        self.assertTrue(self._wait_until(lambda: len(started) == 10))
        stats = guard.stats()
        self.assertEqual(stats["generating"], 10)
        self.assertEqual(stats["heavy_waiting"], 0)
        self.assertEqual(stats["waiting"], 0)
        self.assertEqual(stats["active"], 0)
        self.assertEqual(stats["rejected"], 0)
        with guard.bind_large_image_job("3840x2160") as token:
            self.assertIsNotNone(token)
            self.assertEqual(guard.stats()["generating"], 11)
            self.assertEqual(guard.stats()["active"], 0)
            self.assertEqual(guard.stats()["heavy_waiting"], 0)
        guard.configure_limits(max_generating=10)
        calls = {"body": 0}
        with self.assertRaises(LargeImageAdmissionError) as busy:
            with guard.bind_large_image_job("4K"):
                calls["body"] += 1
        self.assertEqual(busy.exception.reason, "generation_busy")
        self.assertEqual(calls["body"], 0)
        self.assertEqual(guard.stats()["rejected_generation_busy"], 1)
        self.assertEqual(guard.stats()["generating"], 10)
        release.set()
        for thread in threads:
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
        self.assertFalse(errors, errors)
        self.assertEqual(guard.stats()["generating"], 0)
        self.assertEqual(guard.stats()["heavy_waiting"], 0)

    def test_saturated_heavy_queue_rejects_before_upstream(self) -> None:
        release = threading.Event()
        entered: list[int] = []
        errors: list[BaseException] = []
        holders = [
            threading.Thread(target=self._hold, args=(release, entered, errors), name=f"heavy-{i}")
            for i in range(5)
        ]
        for thread in holders:
            thread.start()
        self.assertTrue(self._wait_until(lambda: guard.stats()["active"] == 5))

        def waiter() -> None:
            try:
                with guard.heavy_image_work(declared_size="2048x2048", stage="queue"):
                    entered.append(1)
                    release.wait(timeout=3)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        waiters = [threading.Thread(target=waiter, name=f"wait-{i}") for i in range(10)]
        for thread in waiters:
            thread.start()
        self.assertTrue(self._wait_until(lambda: guard.stats()["heavy_waiting"] == 10))
        calls = {"body": 0}

        def upstream() -> None:
            calls["body"] += 1

        with self.assertRaises(LargeImageAdmissionError) as caught:
            with guard.bind_large_image_job("4K"):
                upstream()
        self.assertEqual(caught.exception.reason, "queue_full")
        self.assertEqual(calls["body"], 0)
        self.assertEqual(guard.stats()["generating"], 0)
        self.assertEqual(guard.stats()["active"], 5)
        self.assertEqual(guard.stats()["heavy_waiting"], 10)
        release.set()
        for thread in holders + waiters:
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
        self.assertFalse(errors, errors)
        self.assertEqual(guard.stats()["generating"], 0)
        self.assertEqual(guard.stats()["heavy_waiting"], 0)
        self.assertEqual(guard.stats()["active"], 0)

    def test_failed_generation_releases_reservation(self) -> None:
        with self.assertRaises(RuntimeError):
            with guard.bind_large_image_job("2K"):
                self.assertEqual(guard.stats()["generating"], 1)
                self.assertEqual(guard.stats()["heavy_waiting"], 0)
                raise RuntimeError("upstream failed")
        self.assertEqual(guard.stats()["generating"], 0)
        self.assertEqual(guard.stats()["heavy_waiting"], 0)
        self.assertEqual(guard.stats()["active"], 0)
        with guard.bind_large_image_job("2K"):
            self.assertEqual(guard.stats()["generating"], 1)

    def _hold(self, release: threading.Event, entered: list[int], errors: list[BaseException]) -> None:
        ConcurrencyTests._hold_heavy(self, release, entered, errors)

    def _wait_until(self, predicate, timeout: float = 2.0) -> bool:
        return ConcurrencyTests._wait_until(self, predicate, timeout)


class DiscoveredSizePipelineTests(GuardTestCase):
    """Declared auto/1K must not drop the permit before upscale and base64.

    The body is a real streamed download of a 3840x2160 header. Processing is
    the real ``_materialize_image_output`` path, not a bare semaphore.
    """

    def test_small_or_missing_declaration_still_caps_actual_large_processing(self) -> None:
        declared_sizes = (None, "auto", "", "1K", "1k", "1024x1024")
        for declared in declared_sizes:
            with self.subTest(declared=declared):
                reset_for_tests(enabled=True)
                guard.configure_limits(max_active=5, max_waiting=10, wait_timeout_secs=2.0)
                self._assert_processing_capped(declared)

    def test_declared_2k_still_holds_across_processing(self) -> None:
        self._assert_processing_capped("2K")
        self._assert_processing_capped("2048x2048")

    def test_processing_error_and_cancel_do_not_leak_or_duplicate(self) -> None:
        from services.protocol.conversation import (
            ConversationRequest,
            ImageGenerationError,
            _generate_single_image,
            _materialize_image_output,
        )

        seen = {"active": -1}

        def exploding_save(_data, _base_url=None):
            seen["active"] = guard.stats()["active"]
            raise RuntimeError("upscale failed")

        backend = _StreamingBackend()
        with mock.patch("services.protocol.conversation.save_image_bytes", exploding_save):
            with self.assertRaises(RuntimeError):
                _materialize_image_output(
                    backend,
                    ConversationRequest(model="gpt-image-2", prompt="p", size="1K", response_format="b64_json"),
                    ["http://example.test/a.png"],
                    1,
                    1,
                    "conv",
                )
        self.assertGreaterEqual(seen["active"], 1)
        self.assertEqual(guard.stats()["active"], 0)
        self.assertEqual(guard.stats()["waiting"], 0)

        calls = {"stream": 0}

        def fake_stream(backend, request, index=1, total=1):
            calls["stream"] += 1
            return [
                _materialize_image_output(
                    _StreamingBackend(),
                    request,
                    ["http://example.test/a.png"],
                    index,
                    total,
                    "conv",
                )
            ]

        class FakeAPI:
            def __init__(self, access_token: str = "", **_kwargs: object) -> None:
                self.progress_callback = None

            def close(self) -> None:
                pass

        with mock.patch("services.protocol.conversation.save_image_bytes", exploding_save), \
                mock.patch("services.protocol.conversation.stream_image_outputs", fake_stream), \
                mock.patch("services.protocol.conversation.OpenAIBackendAPI", FakeAPI), \
                mock.patch("services.protocol.conversation.account_service.get_available_access_token", return_value="tok"), \
                mock.patch("services.protocol.conversation.account_service.get_account", return_value={"email": "a@example.test"}), \
                mock.patch("services.protocol.conversation.account_service.mark_image_result", return_value=None):
            with self.assertRaises(ImageGenerationError):
                _generate_single_image(
                    ConversationRequest(model="gpt-image-2", prompt="p", size="auto", n=1),
                    1,
                    1,
                )
        self.assertEqual(calls["stream"], 1)
        self.assertEqual(guard.stats()["active"], 0)
        self.assertEqual(guard.stats()["waiting"], 0)

        release = threading.Event()
        entered = {"n": 0}
        lock = threading.Lock()

        def holding_save(_data, _base_url=None):
            with lock:
                entered["n"] += 1
            release.wait(timeout=3)
            return "http://example.test/out.png"

        outcomes: list[BaseException] = []

        def worker() -> None:
            try:
                _materialize_image_output(
                    _StreamingBackend(),
                    ConversationRequest(model="gpt-image-2", prompt="p", size=None, response_format="b64_json"),
                    ["http://example.test/a.png"],
                    1,
                    1,
                    "conv",
                )
            except BaseException as exc:  # noqa: BLE001
                outcomes.append(exc)

        with mock.patch("services.protocol.conversation.save_image_bytes", holding_save):
            holders = [threading.Thread(target=worker, name=f"hold-{i}") for i in range(5)]
            for thread in holders:
                thread.start()
            self.assertTrue(ConcurrencyTests._wait_until(self, lambda: entered["n"] == 5 and guard.stats()["active"] == 5))
            queued = threading.Thread(target=worker, name="queued")
            queued.start()
            token_id = None
            deadline = time.time() + 2
            while time.time() < deadline and token_id is None:
                for item in guard.token_snapshot():
                    if item["state"] == "heavy_waiting":
                        token_id = item["id"]
                        break
                time.sleep(0.01)
            self.assertIsNotNone(token_id)
            self.assertEqual(entered["n"], 5)
            self.assertTrue(guard.cancel_token(int(token_id)))
            queued.join(timeout=2)
            self.assertFalse(queued.is_alive())
            self.assertEqual(entered["n"], 5)
            release.set()
            for thread in holders:
                thread.join(timeout=3)
                self.assertFalse(thread.is_alive())
        self.assertEqual(len(outcomes), 1)
        self.assertIsInstance(outcomes[0], LargeImageAdmissionError)
        self.assertEqual(outcomes[0].reason, "cancelled")
        self.assertEqual(guard.stats()["active"], 0)
        self.assertEqual(guard.stats()["waiting"], 0)

    def _assert_processing_capped(self, declared: object) -> None:
        reset_for_tests(enabled=True)
        guard.configure_limits(max_active=5, max_waiting=10, wait_timeout_secs=2.0)
        release = threading.Event()
        lock = threading.Lock()
        state = {"inside": 0, "peak": 0, "done": 0, "active_samples": [], "tail_active": []}
        errors: list[BaseException] = []

        def save(_data, _base_url=None):
            with lock:
                state["inside"] += 1
                state["peak"] = max(state["peak"], state["inside"])
                state["active_samples"].append(guard.stats()["active"])
            release.wait(timeout=3)
            with lock:
                state["inside"] -= 1
                state["done"] += 1
            return "http://example.test/out.png"

        def worker() -> None:
            try:
                from services.protocol.conversation import ConversationRequest, _materialize_image_output

                output = _materialize_image_output(
                    _StreamingBackend(state["tail_active"]),
                    ConversationRequest(
                        model="gpt-image-2",
                        prompt="p",
                        size=declared,  # type: ignore[arg-type]
                        response_format="b64_json",
                    ),
                    ["http://example.test/a.png"],
                    1,
                    1,
                    "conv",
                )
                if output is None or not output.data:
                    errors.append(RuntimeError("missing image output"))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        try:
            with mock.patch("services.protocol.conversation.save_image_bytes", save):
                threads = [threading.Thread(target=worker, name=f"img-{index}") for index in range(6)]
                for thread in threads:
                    thread.start()
                self.assertTrue(
                    ConcurrencyTests._wait_until(
                        self,
                        lambda: state["inside"] == 5 and guard.stats()["active"] == 5,
                    ),
                    guard.stats(),
                )
                time.sleep(0.05)
                self.assertEqual(state["inside"], 5)
                self.assertEqual(state["peak"], 5)
                self.assertLessEqual(guard.stats()["peak_active"], 5)
                release.set()
                for thread in threads:
                    thread.join(timeout=3)
                    self.assertFalse(thread.is_alive())
        finally:
            release.set()
        self.assertFalse(errors, errors)
        self.assertEqual(state["done"], 6)
        self.assertEqual(state["peak"], 5)
        self.assertEqual(guard.stats()["peak_active"], 5)
        self.assertEqual(guard.stats()["active"], 0)
        self.assertEqual(guard.stats()["waiting"], 0)
        self.assertGreaterEqual(len(state["active_samples"]), 5)
        self.assertTrue(all(item >= 1 for item in state["active_samples"]), state["active_samples"])
        self.assertEqual(max(state["active_samples"]), 5)
        self.assertGreaterEqual(len(state["tail_active"]), 6)
        self.assertTrue(all(item >= 1 for item in state["tail_active"]), state["tail_active"])


def _bmp(width: int, height: int) -> bytes:
    """BITMAPINFOHEADER only. Declares the size without a pixel buffer."""
    dib = struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, 0, 0, 0, 0, 0)
    header = b"BM" + struct.pack("<IHHI", 14 + len(dib), 0, 0, 14 + len(dib))
    return header + dib


class _FakeDecodedImage:
    size = (8, 8)
    mode = "RGBA"

    def convert(self, _mode: str):
        return self

    def split(self):
        return (self, self, self, self)

    def resize(self, _size, _resample):
        return self

    def putalpha(self, _alpha) -> None:
        return None

    def save(self, buf, format: str = "PNG") -> None:  # noqa: A002
        buf.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)


class UnknownFormatEditTests(GuardTestCase):
    """BMP is accepted by Pillow. An unrecognized header must not take the 1K path."""

    def test_bmp_edit_pipeline_caps_six_threads_and_known_1k_does_not(self) -> None:
        self._assert_edit_capped(_bmp(3840, 2160), "big.bmp", "image/bmp", capped=True)
        reset_for_tests(enabled=True)
        guard.configure_limits(max_active=5, max_waiting=10, wait_timeout_secs=2.0)
        self._assert_edit_capped(_png(1024, 1024), "small.png", "image/png", capped=False)

    def test_truncated_and_unknown_headers_are_guarded_before_decode(self) -> None:
        truncated = b"\x89PNG\r\n\x1a\n" + b"\x00" * 4
        unknown = b"not-an-image-header!!" + b"\x00" * 16
        for payload, name in ((truncated, "trunc.png"), (unknown, "x.bin")):
            with self.subTest(name=name):
                reset_for_tests(enabled=True)
                guard.configure_limits(max_active=5, max_waiting=10, wait_timeout_secs=2.0)
                seen = {"active": -1}

                def open_image(_data, *args, **kwargs):
                    seen["active"] = guard.stats()["active"]
                    raise RuntimeError("malformed")

                with mock.patch("services.protocol.openai_v1_image_edit.Image.open", open_image):
                    with self.assertRaises(RuntimeError):
                        self._call_edit(payload, name, "image/png")
                self.assertGreaterEqual(seen["active"], 1)
                self.assertEqual(guard.stats()["active"], 0)
                self.assertEqual(guard.stats()["waiting"], 0)

    def test_bmp_reference_download_and_materialize_hold_permit(self) -> None:
        def chunks():
            yield _bmp(3840, 2160)[:26]
            self.assertGreaterEqual(guard.stats()["active"], 1)
            yield b"tail"

        data = guard.read_stream_guarded(chunks(), declared_size="1024x1024", stage="edit_reference_download")
        self.assertTrue(data.startswith(b"BM"))
        self.assertEqual(guard.stats()["active"], 0)

        def small_chunks():
            yield _png(1024, 1024)
            self.assertEqual(guard.stats()["active"], 0)
            yield b"rest"

        guard.read_stream_guarded(small_chunks(), declared_size="1K", stage="edit_reference_download")
        self.assertEqual(guard.stats()["peak_active"], 1)

        reset_for_tests(enabled=True)
        guard.configure_limits(max_active=5, max_waiting=10, wait_timeout_secs=2.0)
        release = threading.Event()
        lock = threading.Lock()
        state = {"inside": 0, "peak": 0, "active": []}
        errors: list[BaseException] = []

        def save(_data, _base_url=None):
            with lock:
                state["inside"] += 1
                state["peak"] = max(state["peak"], state["inside"])
                state["active"].append(guard.stats()["active"])
            release.wait(timeout=3)
            with lock:
                state["inside"] -= 1
            return "http://example.test/out.png"

        def worker() -> None:
            try:
                from services.protocol.conversation import ConversationRequest, _materialize_image_output

                output = _materialize_image_output(
                    _StreamingBackend(header=_bmp(3840, 2160)),
                    ConversationRequest(
                        model="gpt-image-2",
                        prompt="p",
                        size="1024x1024",
                        response_format="b64_json",
                    ),
                    ["http://example.test/a.bmp"],
                    1,
                    1,
                    "conv",
                )
                if output is None:
                    errors.append(RuntimeError("missing bmp output"))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        try:
            with mock.patch("services.protocol.conversation.save_image_bytes", save):
                threads = [threading.Thread(target=worker, name=f"bmp-{i}") for i in range(6)]
                for thread in threads:
                    thread.start()
                self.assertTrue(ConcurrencyTests._wait_until(self, lambda: state["inside"] == 5 and guard.stats()["active"] == 5))
                time.sleep(0.05)
                self.assertEqual(state["peak"], 5)
                release.set()
                for thread in threads:
                    thread.join(timeout=3)
                    self.assertFalse(thread.is_alive())
        finally:
            release.set()
        self.assertFalse(errors, errors)
        self.assertEqual(state["peak"], 5)
        self.assertEqual(guard.stats()["peak_active"], 5)
        self.assertTrue(all(item >= 1 for item in state["active"]))
        self.assertEqual(guard.stats()["active"], 0)

    def _assert_edit_capped(self, image: bytes, name: str, mime: str, *, capped: bool) -> None:
        release = threading.Event()
        lock = threading.Lock()
        state = {"inside": 0, "peak": 0, "active": []}
        errors: list[BaseException] = []
        local = threading.local()

        def open_image(_data, *args, **kwargs):
            if not getattr(local, "counted", False):
                local.counted = True
                with lock:
                    state["inside"] += 1
                    state["peak"] = max(state["peak"], state["inside"])
                    state["active"].append(guard.stats()["active"])
                release.wait(timeout=3)
                with lock:
                    state["inside"] -= 1
            return _FakeDecodedImage()

        def fake_pool(request):
            from services.protocol.conversation import ImageOutput

            return [ImageOutput(kind="result", model=request.model, index=1, total=1, data=[{"url": "u"}])]

        def worker() -> None:
            try:
                self._call_edit(image, name, mime)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        try:
            with mock.patch("services.protocol.openai_v1_image_edit.Image.open", open_image), \
                    mock.patch("services.protocol.openai_v1_image_edit.stream_image_outputs_with_pool", fake_pool):
                threads = [threading.Thread(target=worker, name=f"edit-{i}") for i in range(6)]
                for thread in threads:
                    thread.start()
                if capped:
                    self.assertTrue(
                        ConcurrencyTests._wait_until(self, lambda: state["inside"] == 5 and guard.stats()["active"] == 5)
                    )
                    time.sleep(0.05)
                    self.assertEqual(state["inside"], 5)
                else:
                    self.assertTrue(ConcurrencyTests._wait_until(self, lambda: state["inside"] == 6))
                    self.assertEqual(guard.stats()["active"], 0)
                    self.assertEqual(guard.stats()["peak_active"], 0)
                release.set()
                for thread in threads:
                    thread.join(timeout=3)
                    self.assertFalse(thread.is_alive())
        finally:
            release.set()
        self.assertFalse(errors, errors)
        if capped:
            self.assertEqual(state["peak"], 5)
            self.assertEqual(max(state["active"]), 5)
            self.assertTrue(all(item >= 1 for item in state["active"]))
            self.assertEqual(guard.stats()["peak_active"], 5)
        else:
            self.assertEqual(state["peak"], 6)
            self.assertEqual(guard.stats()["peak_active"], 0)
        self.assertEqual(guard.stats()["active"], 0)
        self.assertEqual(guard.stats()["waiting"], 0)

    def _call_edit(self, image: bytes, name: str, mime: str):
        from services.protocol.openai_v1_image_edit import handle

        return handle({
            "prompt": "edit",
            "images": [(image, name, mime)],
            "mask": [(_png(64, 64), "mask.png", "image/png")],
            "model": "gpt-image-2",
            "size": "1024x1024",
            "n": 1,
        })


class HandoffConservationTests(GuardTestCase):
    """Twenty generations cannot finish into a hidden wait.

    Fifteen pipeline claims are the heavy cap plus the waiting cap. The
    sixteenth request is refused before upstream. The fifteen that were
    admitted stay in generating until they finish together, then become five
    active and ten heavy_waiting, with the counters matching the threads.
    """

    def test_twenty_attempted_completions_match_thread_states(self) -> None:
        guard.configure_limits(wait_timeout_secs=5.0)
        hold = threading.Event()
        started: list[int] = []
        entered: list[int] = []
        rejected: list[str] = []
        errors: list[BaseException] = []
        problems: list[str] = []
        at_barrier: list[dict] = []
        lock = threading.Lock()
        stop = threading.Event()

        def note_barrier() -> None:
            problem = guard.consistency()
            if problem:
                problems.append("barrier " + problem)
            at_barrier.append(dict(guard.stats()))

        admitted = threading.Barrier(15, action=note_barrier, timeout=4)

        def sample() -> None:
            while not stop.is_set():
                problem = guard.consistency()
                if problem:
                    problems.append(problem)
                time.sleep(0.002)

        def worker() -> None:
            try:
                with guard.bind_large_image_job("3840x2160"):
                    with lock:
                        started.append(1)
                    admitted.wait()
                    with guard.heavy_image_work(declared_size="3840x2160", stage="handoff"):
                        with lock:
                            entered.append(1)
                        hold.wait(timeout=4)
            except LargeImageAdmissionError as exc:
                rejected.append(exc.reason)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        sampler = threading.Thread(target=sample, name="sampler", daemon=True)
        sampler.start()
        threads = [threading.Thread(target=worker, name=f"finish-{i}") for i in range(20)]
        for thread in threads:
            thread.start()
        try:
            self.assertTrue(
                self._wait_until(
                    lambda: guard.stats()["active"] == 5
                    and guard.stats()["heavy_waiting"] == 10
                    and guard.stats()["generating"] == 0
                    and len(entered) == 5,
                    timeout=4,
                ),
                guard.stats(),
            )
            settled = guard.stats()
            self.assertEqual(settled["active"], 5)
            self.assertEqual(settled["heavy_waiting"], 10)
            self.assertEqual(settled["waiting"], 10)
            self.assertEqual(settled["generating"], 0)
            self.assertLessEqual(settled["active"] + settled["heavy_waiting"] + settled["generating"], 15)
            states = sorted(item["state"] for item in guard.token_snapshot())
            self.assertEqual(states, ["active"] * 5 + ["heavy_waiting"] * 10)
            self.assertIsNone(guard.consistency())
            # Overflow is refused before the body. Once the heavy queue itself
            # is full the reason is queue_full; before that it is handoff_full.
            self.assertEqual(len(rejected), 5)
            self.assertTrue(set(rejected) <= {"handoff_full", "queue_full"})
            self.assertEqual(len(started), 15)
            self.assertEqual(at_barrier[0]["generating"], 15)
            self.assertEqual(at_barrier[0]["active"], 0)
            self.assertEqual(at_barrier[0]["heavy_waiting"], 0)
        finally:
            hold.set()
        for thread in threads:
            thread.join(timeout=4)
            self.assertFalse(thread.is_alive())
        stop.set()
        sampler.join(timeout=1)
        self.assertFalse(errors, errors)
        self.assertFalse(problems, problems)
        self.assertEqual(len(entered), 15)
        self.assertEqual(
            guard.stats()["rejected_handoff_full"] + guard.stats()["rejected_queue_full"],
            5,
        )
        self.assertLessEqual(guard.stats()["peak_active"], 5)
        self.assertLessEqual(guard.stats()["peak_heavy_waiting"], 10)
        self.assertLessEqual(guard.stats()["peak_generating"], 15)
        self.assertEqual(guard.stats()["active"], 0)
        self.assertEqual(guard.stats()["heavy_waiting"], 0)
        self.assertEqual(guard.stats()["generating"], 0)
        self.assertIsNone(guard.consistency())

    def test_cancel_exception_and_timeout_on_each_transition(self) -> None:
        from services.protocol.conversation import _admission_error_code

        self.assertEqual(_admission_error_code("handoff_full"), "free_large_image_handoff_full")
        self.assertEqual(_admission_error_code("queue_full"), "free_large_image_queue_full")
        self.assertEqual(_admission_error_code("generation_busy"), "free_large_image_generation_busy")
        self.assertEqual(_admission_error_code("queue_timeout"), "free_large_image_queue_timeout")

        with self.assertRaises(RuntimeError):
            with guard.bind_large_image_job("2K"):
                self.assertEqual(guard.stats()["generating"], 1)
                self.assertEqual(guard.stats()["heavy_waiting"], 0)
                self.assertIsNone(guard.consistency())
                raise RuntimeError("upstream failed")
        self.assertEqual(guard.stats()["generating"], 0)
        self.assertIsNone(guard.consistency())

        release = threading.Event()
        reasons: list[str] = []
        ran: list[int] = []

        def cancelled_generation() -> None:
            try:
                with guard.bind_large_image_job("2048x2048") as token:
                    assert token is not None
                    release.wait(timeout=2)
                    token.enter_heavy("download", 2)
                    ran.append(1)
            except LargeImageAdmissionError as exc:
                reasons.append(exc.reason)

        worker = threading.Thread(target=cancelled_generation)
        worker.start()
        token_id = None
        self.assertTrue(self._wait_until(lambda: any(
            item["state"] == "generating" for item in guard.token_snapshot()
        )))
        for item in guard.token_snapshot():
            if item["state"] == "generating":
                token_id = item["id"]
        self.assertIsNotNone(token_id)
        self.assertTrue(guard.cancel_token(int(token_id)))
        release.set()
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(reasons, ["cancelled"])
        self.assertEqual(ran, [])
        self.assertEqual(guard.stats()["generating"], 0)
        self.assertEqual(guard.stats()["active"], 0)
        self.assertIsNone(guard.consistency())

        with self.assertRaises(RuntimeError):
            with guard.heavy_image_work(declared_size="4096x4096", stage="boom") as token:
                self.assertEqual(guard.stats()["active"], 1)
                self.assertIsNone(guard.consistency())
                assert token is not None
                self.assertFalse(token.cancel())
                self.assertEqual(guard.stats()["active"], 1)
                raise RuntimeError("decode failed")
        self.assertEqual(guard.stats()["active"], 0)
        self.assertIsNone(guard.consistency())

        held = threading.Event()
        entered: list[int] = []
        errors: list[BaseException] = []

        def hold() -> None:
            try:
                with guard.heavy_image_work(declared_size="3840x2160", stage="held"):
                    entered.append(1)
                    held.wait(timeout=5)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        holders = [threading.Thread(target=hold, name=f"hold-{i}") for i in range(5)]
        for thread in holders:
            thread.start()
        self.assertTrue(self._wait_until(lambda: guard.stats()["active"] == 5 and len(entered) == 5))
        calls = {"upstream": 0}
        with guard.bind_large_image_job("4K") as token:
            calls["upstream"] += 1
            self.assertEqual(guard.stats()["generating"], 1)
            self.assertEqual(guard.stats()["heavy_waiting"], 0)
            assert token is not None
            with self.assertRaises(LargeImageAdmissionError) as timed:
                token.enter_heavy("download", 0.2)
            self.assertEqual(timed.exception.reason, "queue_timeout")
            self.assertIsNone(guard.consistency())
        self.assertEqual(calls["upstream"], 1)
        self.assertEqual(guard.stats()["generating"], 0)
        self.assertEqual(guard.stats()["heavy_waiting"], 0)
        self.assertEqual(guard.stats()["active"], 5)

        waiting_reasons: list[str] = []

        def queued() -> None:
            try:
                with guard.heavy_image_work(declared_size="3840x2160", stage="queued"):
                    ran.append(1)
            except LargeImageAdmissionError as exc:
                waiting_reasons.append(exc.reason)

        waiter = threading.Thread(target=queued)
        waiter.start()
        waiting_id = None
        self.assertTrue(self._wait_until(lambda: any(
            item["state"] == "heavy_waiting" for item in guard.token_snapshot()
        )))
        for item in guard.token_snapshot():
            if item["state"] == "heavy_waiting":
                waiting_id = item["id"]
        self.assertIsNotNone(waiting_id)
        self.assertTrue(guard.cancel_token(int(waiting_id)))
        waiter.join(timeout=2)
        self.assertEqual(waiting_reasons, ["cancelled"])
        self.assertEqual(ran, [])
        self.assertEqual(guard.stats()["heavy_waiting"], 0)
        self.assertIsNone(guard.consistency())
        held.set()
        for thread in holders:
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
        self.assertFalse(errors, errors)
        self.assertEqual(guard.stats()["active"], 0)
        self.assertEqual(guard.stats()["heavy_waiting"], 0)
        self.assertEqual(guard.stats()["generating"], 0)
        self.assertIsNone(guard.consistency())

    def test_owned_claim_takes_free_seat_ahead_of_full_queue(self) -> None:
        """4 active + 10 waiting + 1 owned generation, with one heavy seat free.

        The queued head has not run yet. Handoff must give that seat to the
        head and keep this claim as the new tail. It must not raise, and it
        must not count as a new rejection.
        """
        guard.configure_limits(wait_timeout_secs=5.0)
        release_one = threading.Event()
        release_all = threading.Event()
        go = threading.Event()
        errors: list[BaseException] = []
        order: list[tuple[str, int]] = []
        upstream = []
        lock = threading.Lock()

        def hold(index: int) -> None:
            try:
                with guard.heavy_image_work(declared_size="3840x2160", stage="held"):
                    if index == 0:
                        release_one.wait(timeout=5)
                    else:
                        release_all.wait(timeout=5)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def waiter(index: int) -> None:
            try:
                with guard.heavy_image_work(declared_size="2048x2048", stage="queued") as token:
                    with lock:
                        order.append(("w", index, 0 if token is None else token.id))
                    release_all.wait(timeout=5)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def generate() -> None:
            try:
                with guard.bind_large_image_job("3840x2160") as token:
                    assert token is not None
                    gen_ids.append(token.id)
                    go.wait(timeout=5)
                    upstream.append(1)
                    with guard.heavy_image_work(declared_size="3840x2160", stage="handoff") as current:
                        with lock:
                            order.append(("g", 0, 0 if current is None else current.id))
                        release_all.wait(timeout=5)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        gen_ids: list[int] = []
        holders = [threading.Thread(target=hold, args=(i,), name=f"hold-{i}") for i in range(5)]
        for thread in holders:
            thread.start()
        self.assertTrue(self._wait_until(lambda: guard.stats()["active"] == 5))
        waiters = [threading.Thread(target=waiter, args=(i,), name=f"wait-{i}") for i in range(9)]
        for thread in waiters:
            thread.start()
        self.assertTrue(self._wait_until(lambda: guard.stats()["heavy_waiting"] == 9))
        generator = threading.Thread(target=generate, name="owner")
        generator.start()
        self.assertTrue(self._wait_until(lambda: guard.stats()["generating"] == 1 and len(gen_ids) == 1))
        head_id = next(item["id"] for item in guard.token_snapshot() if item["queue_index"] == 0)
        condition = guard._guard.cv
        original_notify = condition.notify_all
        quiet = {"on": False}

        def notify_all() -> None:
            if not quiet["on"]:
                original_notify()

        condition.notify_all = notify_all  # type: ignore[method-assign]
        tenth = threading.Thread(target=waiter, args=(9,), name="wait-9")
        try:
            quiet["on"] = True
            release_one.set()
            self.assertTrue(self._wait_until(lambda: guard.stats()["active"] == 4), guard.stats())
            tenth.start()
            self.assertTrue(
                self._wait_until(
                    lambda: guard.stats()["active"] == 4
                    and guard.stats()["heavy_waiting"] == 10
                    and guard.stats()["generating"] == 1
                ),
                guard.stats(),
            )
            self.assertIsNone(guard.consistency())
            rejected_before = guard.stats()["rejected"]
            calls = {"body": 0}
            with self.assertRaises(LargeImageAdmissionError) as fresh:
                with guard.bind_large_image_job("4096x4096"):
                    calls["body"] += 1
            self.assertEqual(fresh.exception.reason, "queue_full")
            self.assertEqual(calls["body"], 0)
            self.assertEqual(guard.stats()["rejected"], rejected_before + 1)
            self.assertEqual(guard.stats()["rejected_queue_full"], 1)
            self.assertEqual(guard.stats()["rejected_handoff_full"], 0)
            go.set()
            self.assertTrue(
                self._wait_until(
                    lambda: guard.stats()["generating"] == 0
                    and guard.stats()["active"] == 5
                    and guard.stats()["heavy_waiting"] == 10
                ),
                guard.stats(),
            )
            self.assertEqual(guard.stats()["rejected"], rejected_before + 1)
            self.assertFalse(errors, errors)
            snapshot = {item["id"]: item for item in guard.token_snapshot()}
            self.assertEqual(snapshot[head_id]["state"], "active")
            self.assertIsNone(snapshot[head_id]["queue_index"])
            self.assertEqual(snapshot[gen_ids[0]]["state"], "heavy_waiting")
            self.assertEqual(snapshot[gen_ids[0]]["queue_index"], 9)
            self.assertIsNone(guard.consistency())
        finally:
            quiet["on"] = False
            condition.notify_all = original_notify  # type: ignore[method-assign]
            with condition:
                condition.notify_all()
            release_all.set()
            for thread in holders + waiters + [tenth, generator]:
                if thread.ident is not None:
                    thread.join(timeout=5)
        for thread in holders + waiters + [tenth, generator]:
            self.assertFalse(thread.is_alive(), thread.name)
        self.assertFalse(errors, errors)
        self.assertEqual(upstream, [1])
        with lock:
            seen = [item for item in order if item[0] in {"w", "g"}]
        gen_at = next(i for i, item in enumerate(seen) if item[0] == "g")
        head_at = next(i for i, item in enumerate(seen) if item[2] == head_id)
        self.assertLess(head_at, gen_at)
        self.assertEqual(guard.stats()["active"], 0)
        self.assertEqual(guard.stats()["heavy_waiting"], 0)
        self.assertEqual(guard.stats()["generating"], 0)
        self.assertIsNone(guard.consistency())

    def test_cancel_and_timeout_do_not_reject_an_owned_claim(self) -> None:
        guard.configure_limits(wait_timeout_secs=5.0)
        release = threading.Event()
        errors: list[BaseException] = []
        holders = [
            threading.Thread(
                target=self._park_heavy,
                args=(release, errors),
                name=f"hold-{i}",
            )
            for i in range(5)
        ]
        for thread in holders:
            thread.start()
        self.assertTrue(self._wait_until(lambda: guard.stats()["active"] == 5))
        waiters = [
            threading.Thread(target=self._park_heavy, args=(release, errors), name=f"wait-{i}")
            for i in range(9)
        ]
        for thread in waiters:
            thread.start()
        self.assertTrue(self._wait_until(lambda: guard.stats()["heavy_waiting"] == 9))
        outcome: list[str] = []
        upstream = []

        def owner() -> None:
            try:
                with guard.bind_large_image_job("2K") as token:
                    assert token is not None
                    upstream.append(1)
                    token.enter_heavy("download", 0.25)
                    outcome.append("entered")
                    token.leave_heavy()
            except LargeImageAdmissionError as exc:
                outcome.append(exc.reason)

        worker = threading.Thread(target=owner)
        worker.start()
        self.assertTrue(self._wait_until(lambda: guard.stats()["generating"] == 1 or outcome))
        rejected_before = guard.stats()["rejected"]
        worker.join(timeout=3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(outcome, ["queue_timeout"])
        self.assertEqual(upstream, [1])
        self.assertEqual(guard.stats()["rejected"], rejected_before)
        self.assertEqual(guard.stats()["generating"], 0)
        self.assertEqual(guard.stats()["heavy_waiting"], 9)
        self.assertEqual(guard.stats()["active"], 5)
        self.assertIsNone(guard.consistency())

        parked: list[int] = []

        def cancellable() -> None:
            try:
                with guard.bind_large_image_job("4K") as token:
                    assert token is not None
                    parked.append(token.id)
                    token.enter_heavy("download", 2)
                    outcome.append("entered-after-cancel")
                    token.leave_heavy()
            except LargeImageAdmissionError as exc:
                outcome.append(exc.reason)

        # Waiting is 9, so this owner is admitted, then cancelled before heavy work.
        other = threading.Thread(target=cancellable)
        try:
            other.start()
            self.assertTrue(self._wait_until(lambda: len(parked) == 1))
            self.assertTrue(guard.cancel_token(parked[0]) or outcome[-1:] == ["cancelled"])
            other.join(timeout=3)
            self.assertEqual(outcome, ["queue_timeout", "cancelled"])
            self.assertEqual(guard.stats()["rejected"], rejected_before)
            self.assertEqual(guard.stats()["generating"], 0)
            self.assertNotIn("entered-after-cancel", outcome)
            self.assertIsNone(guard.consistency())
        finally:
            release.set()
            other.join(timeout=3)
            for thread in holders + waiters:
                thread.join(timeout=4)
        self.assertFalse(errors, errors)
        self.assertEqual(guard.stats()["active"], 0)
        self.assertEqual(guard.stats()["heavy_waiting"], 0)
        self.assertEqual(guard.stats()["generating"], 0)

    def test_seeded_schedules_match_token_states(self) -> None:
        for seed in range(64):
            with self.subTest(seed=seed):
                reset_for_tests(enabled=True)
                guard.configure_limits(max_active=5, max_waiting=10, wait_timeout_secs=3.0)
                self._run_seed(seed)
                stats = guard.stats()
                self.assertEqual(stats["active"], 0, stats)
                self.assertEqual(stats["heavy_waiting"], 0, stats)
                self.assertEqual(stats["generating"], 0, stats)
                self.assertEqual(guard.token_snapshot(), [], guard.token_snapshot())
                self.assertIsNone(guard.consistency(), stats)

    def _run_seed(self, seed: int) -> None:
        rng = random.Random(seed)
        failures: list[str] = []
        problems: list[str] = []
        stop = threading.Event()

        def sample() -> None:
            while not stop.is_set():
                problem = guard.consistency()
                if problem:
                    problems.append(problem)
                stop.wait(0.002)

        def worker(index: int) -> None:
            local = random.Random(seed * 1000 + index)
            entered_body = False
            try:
                time.sleep(local.random() * 0.005)
                roll = local.randrange(5)
                if roll == 0:
                    with guard.heavy_image_pipeline(declared_size="2K") as pipe:
                        entered_body = True
                        if pipe is not None:
                            pipe.cover(3_840 * 2_160, stage="base64")
                        if local.random() < 0.15:
                            raise RuntimeError("decode")
                        time.sleep(local.random() * 0.004)
                elif roll == 1:
                    with guard.bind_large_image_job("3840x2160"):
                        entered_body = True
                        time.sleep(local.random() * 0.004)
                        with guard.heavy_image_work(declared_size="3840x2160", stage="seed"):
                            if local.random() < 0.15:
                                raise RuntimeError("decode")
                            time.sleep(local.random() * 0.006)
                else:
                    with guard.heavy_image_work(declared_size="2048x2048", stage="seed"):
                        entered_body = True
                        time.sleep(local.random() * 0.006)
            except LargeImageAdmissionError as exc:
                if entered_body and exc.reason == "handoff_full":
                    failures.append(f"{seed}:{index} owner handoff_full")
                elif entered_body and exc.reason not in {"queue_timeout", "cancelled"}:
                    failures.append(f"{seed}:{index} {exc.reason}")
            except RuntimeError:
                return
            except BaseException as exc:  # noqa: BLE001
                failures.append(f"{seed}:{index} {exc!r}")

        sampler = threading.Thread(target=sample, name=f"sample-{seed}", daemon=True)
        sampler.start()
        threads = [threading.Thread(target=worker, args=(i,), name=f"s{seed}-{i}") for i in range(24)]
        for thread in threads:
            thread.start()
        cancel_rng = random.Random(seed + 99)
        for _ in range(8):
            time.sleep(0.005)
            victims = [
                item for item in guard.token_snapshot()
                if item["state"] in {"generating", "heavy_waiting"}
            ]
            if victims and cancel_rng.random() < 0.7:
                guard.cancel_token(int(cancel_rng.choice(victims)["id"]))
        for thread in threads:
            thread.join(timeout=4)
            if thread.is_alive():
                failures.append(f"{seed} stuck {thread.name}")
        stop.set()
        sampler.join(timeout=1)
        self.assertFalse(failures, failures)
        self.assertFalse(problems, problems)
        # Rejections, when they happened, were new admissions only.
        stats = guard.stats()
        self.assertEqual(
            stats["rejected"],
            stats["rejected_queue_full"] + stats["rejected_generation_busy"] + stats["rejected_handoff_full"],
        )
        self.assertEqual(guard.token_snapshot(), [], guard.token_snapshot())

    def _park_heavy(self, release: threading.Event, errors: list[BaseException]) -> None:
        try:
            with guard.heavy_image_work(declared_size="3840x2160", stage="park"):
                release.wait(timeout=5)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def _wait_until(self, predicate, timeout: float = 2.0) -> bool:
        return ConcurrencyTests._wait_until(self, predicate, timeout)


class PromotedOwnershipTests(GuardTestCase):
    """A promoted slot is allocated, not owned, until this thread arms it."""

    def test_materialize_promotion_leaves_no_token(self) -> None:
        for _ in range(5):
            self._materialize_promoted_round(explode=False)
            self._assert_drained()
        with guard.heavy_image_work(declared_size="2K", stage="after-rounds"):
            self.assertEqual(guard.stats()["active"], 1)
        self._assert_drained()

    def test_materialize_exception_and_second_cover_release_the_slot(self) -> None:
        self._materialize_promoted_round(explode=True)
        self._assert_drained()
        self._pipeline_two_covers_after_promotion()
        self._assert_drained()

    def test_cancel_close_and_timeout_around_promotion(self) -> None:
        waiting = guard._begin_heavy_waiter()
        self.assertTrue(guard.cancel_token(waiting.id))
        with self.assertRaises(LargeImageAdmissionError) as cancelled:
            waiting.enter_heavy("held", 1)
        self.assertEqual(cancelled.exception.reason, "cancelled")
        waiting.close()
        self._assert_drained()

        token, holders, thread = self._promote_without_arming()
        try:
            self.assertEqual(token.state, "active")
            self.assertFalse(token.work_running)
            self.assertEqual(token.depth, 0)
            self.assertTrue(guard.cancel_token(token.id))
            self.assertNotEqual(token.state, "active")
            self.assertFalse(self._snapshot_has(token.id))
            with self.assertRaises(LargeImageAdmissionError):
                token.enter_heavy("held", 1)
        finally:
            self._release_promotion(holders, thread)
        self._assert_drained()

        token, holders, thread = self._promote_without_arming()
        try:
            token.close()
            self.assertFalse(token.work_running)
            self.assertFalse(self._snapshot_has(token.id))
            self.assertNotEqual(token.state, "active")
        finally:
            self._release_promotion(holders, thread)
        self._assert_drained()

        token, holders, thread = self._promote_without_arming()
        try:
            token.enter_heavy("download", 0.05)
            self.assertTrue(token.work_running)
            self.assertGreaterEqual(token.depth, 1)
            token.leave_heavy()
            token.close()
            self.assertFalse(self._snapshot_has(token.id))
        finally:
            self._release_promotion(holders, thread)
        self._assert_drained()

        release = threading.Event()
        errors: list[BaseException] = []
        holders = [
            threading.Thread(target=self._park, args=(release, errors), name=f"full-{i}")
            for i in range(5)
        ]
        for holder in holders:
            holder.start()
        self.assertTrue(ConcurrencyTests._wait_until(self, lambda: guard.stats()["active"] == 5))
        blocked = guard._begin_heavy_waiter()
        with self.assertRaises(LargeImageAdmissionError) as timed:
            blocked.enter_heavy("queued", 0.2)
        self.assertEqual(timed.exception.reason, "queue_timeout")
        blocked.close()
        self.assertFalse(self._snapshot_has(blocked.id))
        self.assertEqual(guard.stats()["rejected"], 0)
        release.set()
        for holder in holders:
            holder.join(timeout=3)
        self.assertFalse(errors, errors)
        self._assert_drained()

    def test_foreign_thread_cannot_reenter_or_release(self) -> None:
        release = threading.Event()
        errors: list[BaseException] = []
        shared: dict[str, guard.Token] = {}
        bodies: list[int] = []
        lock = threading.Lock()

        def hold() -> None:
            try:
                with guard.heavy_image_work(declared_size="3840x2160", stage="owner") as token:
                    assert token is not None
                    with lock:
                        bodies.append(threading.get_ident())
                        shared.setdefault("token", token)
                    release.wait(timeout=3)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        owners = [threading.Thread(target=hold, name=f"owner-{i}") for i in range(5)]
        for thread in owners:
            thread.start()
        self.assertTrue(ConcurrencyTests._wait_until(
            self, lambda: guard.stats()["active"] == 5 and len(bodies) == 5 and "token" in shared,
        ))
        token = shared["token"]
        depth = token.depth
        foreign_errors: list[BaseException] = []

        def intrude() -> None:
            try:
                token.enter_heavy("foreign", 1)
                with lock:
                    bodies.append(threading.get_ident())
            except LargeImageAdmissionError as exc:
                foreign_errors.append(exc)
            token.leave_heavy()

        intruder = threading.Thread(target=intrude, name="foreign")
        intruder.start()
        intruder.join(timeout=2)
        self.assertFalse(intruder.is_alive())
        self.assertEqual([exc.reason for exc in foreign_errors], ["foreign_thread"])
        self.assertEqual(token.depth, depth)
        self.assertEqual(guard.stats()["active"], 5)
        self.assertEqual(len(bodies), 5)
        self.assertFalse(guard.cancel_token(token.id))
        self.assertEqual(guard.stats()["active"], 5)
        token.close()
        self.assertEqual(guard.stats()["active"], 5)
        self.assertTrue(token.work_running)
        blocked = {"ran": False}

        def sixth() -> None:
            try:
                with guard.heavy_image_work(declared_size="3840x2160", stage="sixth"):
                    blocked["ran"] = True
            except LargeImageAdmissionError:
                return

        guard.configure_limits(wait_timeout_secs=0.2)
        extra = threading.Thread(target=sixth, name="sixth")
        extra.start()
        extra.join(timeout=2)
        self.assertFalse(blocked["ran"])
        self.assertEqual(guard.stats()["active"], 5)
        release.set()
        for thread in owners:
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
        self.assertFalse(errors, errors)
        self._assert_drained()

    def test_materialize_rejects_foreign_enter_and_drains(self) -> None:
        seen: dict[str, object] = {}

        def finish(raw: bytes, **_kwargs: object) -> dict[str, str]:
            current = guard._current_token()
            self.assertIsNotNone(current)
            assert current is not None
            self.assertTrue(current.owns_heavy_work())
            active = guard.stats()["active"]
            box: list[BaseException] = []

            def intrude() -> None:
                try:
                    current.enter_heavy("foreign", 1)
                except LargeImageAdmissionError as exc:
                    box.append(exc)
                current.leave_heavy()

            intruder = threading.Thread(target=intrude)
            intruder.start()
            intruder.join(timeout=2)
            self.assertFalse(intruder.is_alive())
            self.assertEqual([exc.reason for exc in box], ["foreign_thread"])
            self.assertEqual(guard.stats()["active"], active)
            self.assertTrue(current.work_running)
            seen["id"] = current.id
            return {"url": "offline://out", "revised_prompt": "p", "unused": str(len(raw))}

        from services.protocol.conversation import ConversationRequest, _materialize_image_output

        request = ConversationRequest(model="gpt-image-2", prompt="p", size="2K", response_format="url")
        with mock.patch("services.protocol.conversation._store_processed_image", finish):
            output = _materialize_image_output(
                _StreamingBackend(),
                request,
                ["offline://image"],
                1,
                1,
                "offline",
            )
        self.assertIsNotNone(output)
        self.assertFalse(self._snapshot_has(int(seen["id"])))
        self._assert_drained()

    def _materialize_promoted_round(self, explode: bool) -> None:
        guard.configure_limits(wait_timeout_secs=3.0)
        holders = [guard._begin_heavy_waiter() for _ in range(4)]
        for token in holders:
            token.enter_heavy("held", 3)
        promoted: list[guard.Token] = []
        seen: dict[str, object] = {}
        original = guard._begin_heavy_waiter
        producer_thread: list[threading.Thread] = []

        def pause_after_admission() -> guard.Token:
            token = original()
            if not promoted:
                promoted.append(token)

                def run_producer() -> None:
                    producer = guard._begin_generation()
                    try:
                        producer.enter_heavy("other-generated", 3)
                        producer.leave_heavy()
                    finally:
                        producer.close()

                worker = threading.Thread(target=run_producer, name="producer")
                producer_thread.append(worker)
                worker.start()
                deadline = time.time() + 2
                while time.time() < deadline and not (
                    token.state == "active" and not token.work_running and token.depth == 0
                ):
                    time.sleep(0.001)
                if not (token.state == "active" and not token.work_running and token.depth == 0):
                    raise AssertionError(f"promotion did not land unarmed: {token.state}")
            return token

        def finish(raw: bytes, **_kwargs: object) -> dict[str, str]:
            current = guard._current_token()
            self.assertIsNotNone(current)
            assert current is not None
            seen["state"] = current.state
            seen["work_running"] = current.work_running
            seen["depth"] = current.depth
            seen["id"] = current.id
            if explode:
                raise RuntimeError("finish failed")
            return {"url": "offline://out", "revised_prompt": "p", "unused": str(len(raw))}

        guard._begin_heavy_waiter = pause_after_admission  # type: ignore[method-assign]
        try:
            from services.protocol.conversation import ConversationRequest, _materialize_image_output

            request = ConversationRequest(
                model="gpt-image-2",
                prompt="p",
                size=None,
                response_format="url",
            )
            with mock.patch("services.protocol.conversation._store_processed_image", finish):
                if explode:
                    with self.assertRaises(RuntimeError):
                        _materialize_image_output(
                            _StreamingBackend(),
                            request,
                            ["offline://image"],
                            1,
                            1,
                            "offline",
                        )
                else:
                    output = _materialize_image_output(
                        _StreamingBackend(),
                        request,
                        ["offline://image"],
                        1,
                        1,
                        "offline",
                    )
                    self.assertIsNotNone(output)
            self.assertTrue(promoted)
            self.assertEqual(seen.get("state"), "active")
            self.assertTrue(seen.get("work_running"))
            self.assertGreaterEqual(int(seen.get("depth") or 0), 1)
            self.assertFalse(self._snapshot_has(promoted[0].id))
        finally:
            guard._begin_heavy_waiter = original  # type: ignore[method-assign]
            for token in holders:
                token.leave_heavy()
                token.close()
            for worker in producer_thread:
                worker.join(timeout=3)

    def _pipeline_two_covers_after_promotion(self) -> None:
        guard.configure_limits(wait_timeout_secs=3.0)
        holders = [guard._begin_heavy_waiter() for _ in range(4)]
        for token in holders:
            token.enter_heavy("held", 3)
        promoted: list[guard.Token] = []
        original = guard._begin_heavy_waiter
        producer_thread: list[threading.Thread] = []

        def pause_after_admission() -> guard.Token:
            token = original()
            if not promoted:
                promoted.append(token)

                def run_producer() -> None:
                    producer = guard._begin_generation()
                    try:
                        producer.enter_heavy("other-generated", 3)
                        producer.leave_heavy()
                    finally:
                        producer.close()

                worker = threading.Thread(target=run_producer, name="producer-cover")
                producer_thread.append(worker)
                worker.start()
                deadline = time.time() + 2
                while time.time() < deadline and token.state != "active":
                    time.sleep(0.001)
            return token

        guard._begin_heavy_waiter = pause_after_admission  # type: ignore[method-assign]
        try:
            with guard.heavy_image_pipeline(declared_size="2K", stage="download_decode") as pipe:
                self.assertIsNotNone(pipe)
                assert pipe is not None
                first = pipe.token
                self.assertIsNotNone(first)
                assert first is not None
                self.assertTrue(first.owns_heavy_work())
                self.assertEqual(guard.stats()["active"], 5)
                pipe.cover(3_840 * 2_160, stage="base64")
                self.assertIs(pipe.token, first)
                self.assertEqual(guard.stats()["active"], 5)
                self.assertTrue(first.owns_heavy_work())
            self.assertTrue(promoted)
            self.assertFalse(self._snapshot_has(promoted[0].id))
        finally:
            guard._begin_heavy_waiter = original  # type: ignore[method-assign]
            for token in holders:
                token.leave_heavy()
                token.close()
            for worker in producer_thread:
                worker.join(timeout=3)

    def _promote_without_arming(self) -> tuple[guard.Token, list[guard.Token], threading.Thread]:
        guard.configure_limits(wait_timeout_secs=3.0)
        holders = [guard._begin_heavy_waiter() for _ in range(4)]
        for token in holders:
            token.enter_heavy("held", 3)
        token = guard._begin_heavy_waiter()

        def run_producer() -> None:
            producer = guard._begin_generation()
            try:
                producer.enter_heavy("other-generated", 3)
                producer.leave_heavy()
            finally:
                producer.close()

        worker = threading.Thread(target=run_producer, name="promote")
        worker.start()
        self.assertTrue(
            ConcurrencyTests._wait_until(
                self,
                lambda: token.state == "active" and not token.work_running and token.depth == 0,
            )
        )
        return token, holders, worker

    def _release_promotion(self, holders: list[guard.Token], worker: threading.Thread) -> None:
        for token in holders:
            if token.state == "active":
                token.leave_heavy()
            token.close()
        worker.join(timeout=3)

    def _park(self, release: threading.Event, errors: list[BaseException]) -> None:
        try:
            with guard.heavy_image_work(declared_size="3840x2160", stage="park"):
                release.wait(timeout=3)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def _snapshot_has(self, token_id: int) -> bool:
        return any(item["id"] == token_id for item in guard.token_snapshot())

    def _assert_drained(self) -> None:
        self.assertEqual(guard.stats()["active"], 0, guard.stats())
        self.assertEqual(guard.stats()["heavy_waiting"], 0, guard.stats())
        self.assertEqual(guard.stats()["generating"], 0, guard.stats())
        self.assertEqual(guard.token_snapshot(), [], guard.token_snapshot())
        self.assertIsNone(guard.consistency(), guard.stats())

    def _wait_until(self, predicate, timeout: float = 2.0) -> bool:
        return ConcurrencyTests._wait_until(self, predicate, timeout)


class AdmissionWaitTests(GuardTestCase):
    """Claim-full requests wait before upstream. The line is not a heavy claim."""

    def test_sixteenth_waits_and_does_not_call_upstream_until_a_claim_frees(self) -> None:
        guard.configure_limits(admission_wait_secs=3)
        release = threading.Event()
        started: list[int] = []
        errors: list[BaseException] = []

        def occupy() -> None:
            try:
                with guard.bind_large_image_job("3840x2160"):
                    started.append(1)
                    release.wait(timeout=3)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        holders = [threading.Thread(target=occupy, name=f"claim-{i}") for i in range(15)]
        for thread in holders:
            thread.start()
        self.assertTrue(self._wait_until(lambda: guard.stats()["generating"] == 15))
        calls = {"body": 0}
        entered = threading.Event()

        def late() -> None:
            try:
                with guard.bind_large_image_job("3840x2160"):
                    calls["body"] += 1
                    entered.set()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        waiter = threading.Thread(target=late, name="waiter")
        waiter.start()
        self.assertTrue(self._wait_until(lambda: guard.stats()["admission_waiting"] == 1))
        self.assertEqual(calls["body"], 0)
        self.assertEqual(guard.stats()["generating"], 15)
        self.assertEqual(guard.stats()["active"], 0)
        self.assertEqual(guard.stats()["heavy_waiting"], 0)
        self.assertIsNone(guard.consistency())
        release.set()
        self.assertTrue(entered.wait(timeout=3))
        self.assertEqual(calls["body"], 1)
        waiter.join(timeout=3)
        for thread in holders:
            thread.join(timeout=3)
        self.assertFalse(errors, errors)
        self._assert_drained()

    def test_admission_full_timeout_cancel_and_restart_do_not_start_upstream(self) -> None:
        guard.configure_limits(
            max_active=1,
            max_heavy_waiting=1,
            max_generating=1,
            max_admission_waiting=1,
            admission_wait_secs=2,
        )
        release = threading.Event()
        holder_started = threading.Event()

        def hold() -> None:
            with guard.bind_large_image_job("2K"):
                holder_started.set()
                release.wait(timeout=3)

        holder = threading.Thread(target=hold)
        holder.start()
        self.assertTrue(holder_started.wait(timeout=2))
        queued = threading.Event()
        outcome: list[str] = []

        def wait_for_slot() -> None:
            try:
                with guard.bind_large_image_job("2K"):
                    outcome.append("body")
            except LargeImageAdmissionError as exc:
                outcome.append(exc.reason)
            queued.set()

        first = threading.Thread(target=wait_for_slot)
        first.start()
        self.assertTrue(self._wait_until(lambda: guard.stats()["admission_waiting"] == 1))
        calls = {"body": 0}
        with self.assertRaises(LargeImageAdmissionError) as full:
            with guard.bind_large_image_job("4K"):
                calls["body"] += 1
        self.assertEqual(full.exception.reason, "admission_full")
        self.assertEqual(calls["body"], 0)
        self.assertEqual(guard.stats()["rejected_admission_full"], 1)
        token_id = next(item["id"] for item in guard.token_snapshot() if item["state"] == "admission_waiting")
        self.assertTrue(guard.cancel_token(token_id))
        self.assertTrue(queued.wait(timeout=2))
        self.assertEqual(outcome, ["cancelled"])
        first.join(timeout=2)
        release.set()
        holder.join(timeout=2)

        guard.configure_limits(admission_wait_secs=0.2)
        release.clear()
        holder_started.clear()
        holder = threading.Thread(target=hold)
        holder.start()
        self.assertTrue(holder_started.wait(timeout=2))
        timed: list[str] = []
        body = {"n": 0}

        def expire() -> None:
            try:
                with guard.bind_large_image_job("2K"):
                    body["n"] += 1
            except LargeImageAdmissionError as exc:
                timed.append(exc.reason)

        worker = threading.Thread(target=expire)
        worker.start()
        worker.join(timeout=2)
        self.assertEqual(timed, ["admission_timeout"])
        self.assertEqual(body["n"], 0)
        self.assertGreaterEqual(guard.stats()["admission_timed_out"], 1)
        release.set()
        holder.join(timeout=2)

        guard.configure_limits(admission_wait_secs=2)
        release.clear()
        holder_started.clear()
        holder = threading.Thread(target=hold)
        holder.start()
        self.assertTrue(holder_started.wait(timeout=2))
        restarted = {"body": 0, "reason": ""}

        def stale() -> None:
            try:
                with guard.bind_large_image_job("2K"):
                    restarted["body"] += 1
            except LargeImageAdmissionError as exc:
                restarted["reason"] = exc.reason

        stale_thread = threading.Thread(target=stale)
        stale_thread.start()
        self.assertTrue(self._wait_until(lambda: guard.stats()["admission_waiting"] == 1))
        reset_for_tests(enabled=True)
        guard.configure_limits(
            max_active=1,
            max_heavy_waiting=1,
            max_generating=1,
            admission_wait_secs=0,
        )
        stale_thread.join(timeout=2)
        self.assertEqual(restarted["body"], 0)
        self.assertEqual(restarted["reason"], "cancelled")
        release.set()
        holder.join(timeout=2)
        self._assert_drained()

    def test_fifo_and_parallel_n_do_not_deadlock_or_double_start(self) -> None:
        guard.configure_limits(
            max_active=1,
            max_heavy_waiting=1,
            max_generating=2,
            admission_wait_secs=3,
        )
        order: list[int] = []
        lock = threading.Lock()
        release = threading.Event()
        errors: list[BaseException] = []

        def job(index: int) -> None:
            try:
                with guard.bind_large_image_job("3840x2160"):
                    with lock:
                        order.append(index)
                    release.wait(timeout=3)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=job, args=(i,), name=f"n-{i}") for i in range(4)]
        for thread in threads:
            thread.start()
            time.sleep(0.02)
        self.assertTrue(self._wait_until(lambda: guard.stats()["generating"] == 2))
        self.assertEqual(order, [0, 1])
        self.assertGreaterEqual(guard.stats()["admission_waiting"], 1)
        self.assertLessEqual(guard.stats()["generating"], 2)
        release.set()
        for thread in threads:
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
        self.assertFalse(errors, errors)
        self.assertEqual(order, [0, 1, 2, 3])
        self._assert_drained()

    def test_cancelled_http_task_does_not_start_upstream(self) -> None:
        import anyio

        guard.configure_limits(
            max_active=1,
            max_heavy_waiting=1,
            max_generating=1,
            admission_wait_secs=3,
        )
        release = threading.Event()

        def hold() -> None:
            with guard.bind_large_image_job("2K"):
                release.wait(timeout=3)

        holder = threading.Thread(target=hold)
        holder.start()
        self.assertTrue(self._wait_until(lambda: guard.stats()["generating"] == 1))
        body = {"n": 0}

        async def main() -> None:
            async with anyio.create_task_group() as group:
                async def waiter() -> None:
                    def blocking() -> None:
                        try:
                            with guard.bind_large_image_job("2K"):
                                body["n"] += 1
                        except LargeImageAdmissionError:
                            return

                    await anyio.to_thread.run_sync(blocking)

                group.start_soon(waiter)
                await anyio.sleep(0.3)
                group.cancel_scope.cancel()

        anyio.run(main)
        self.assertEqual(body["n"], 0)
        self.assertEqual(guard.stats()["admission_waiting"], 0)
        release.set()
        holder.join(timeout=2)
        self._assert_drained()

    def _assert_drained(self) -> None:
        self.assertEqual(guard.stats()["admission_waiting"], 0, guard.stats())
        self.assertEqual(guard.stats()["generating"], 0, guard.stats())
        self.assertEqual(guard.stats()["active"], 0, guard.stats())
        self.assertEqual(guard.stats()["heavy_waiting"], 0, guard.stats())
        self.assertEqual(guard.token_snapshot(), [])
        self.assertIsNone(guard.consistency())

    def _wait_until(self, predicate, timeout: float = 2.0) -> bool:
        return ConcurrencyTests._wait_until(self, predicate, timeout)


class RealRequestAdmissionTests(GuardTestCase):
    """Cancellation has to be the inbound generation request, not an AnyIO-only scope."""

    def test_http_cancel_and_disconnect_do_not_start_upstream(self) -> None:
        import json

        import httpx
        import main

        for n, disconnect in ((1, False), (4, False), (1, True), (4, True)):
            with self.subTest(n=n, disconnect=disconnect):
                reset_for_tests(enabled=True)
                guard.configure_limits(
                    max_active=1,
                    max_heavy_waiting=1,
                    max_generating=1,
                    admission_wait_secs=3,
                )
                release = threading.Event()
                calls: list[int] = []

                def hold() -> None:
                    with guard.bind_large_image_job("2K"):
                        release.wait(timeout=8)

                holder = threading.Thread(target=hold)
                holder.start()
                self.assertTrue(self._wait_until(lambda: guard.stats()["generating"] == 1))

                def body(req, index, total):
                    calls.append(index)
                    from services.protocol.conversation import ImageOutput
                    return [ImageOutput(
                        kind="result", model=req.model, index=index, total=total,
                        data=[{"b64_json": "b2ZmbGluZQ=="}],
                    )]

                async def scenario() -> None:
                    with mock.patch("api.ai.require_identity", return_value={"id": "offline", "name": "offline", "role": "user"}), \
                            mock.patch("services.protocol.conversation._generate_single_image_body", body):
                        if not disconnect:
                            async with httpx.AsyncClient(
                                transport=httpx.ASGITransport(app=main.app),
                                base_url="http://offline.test",
                            ) as client:
                                task = asyncio.create_task(client.post(
                                    "/v1/images/generations",
                                    json={"model": "gpt-image-2", "prompt": "offline", "size": "2048x2048", "n": n, "stream": False},
                                ))
                                self.assertTrue(await self._until(lambda: guard.stats()["admission_waiting"] == n))
                                task.cancel()
                                try:
                                    await task
                                except asyncio.CancelledError:
                                    pass
                        else:
                            disconnected = asyncio.Event()
                            received = {"done": False}
                            payload = json.dumps({
                                "model": "gpt-image-2", "prompt": "offline", "size": "2048x2048", "n": n, "stream": False,
                            }).encode()

                            async def receive():
                                if not received["done"]:
                                    received["done"] = True
                                    return {"type": "http.request", "body": payload, "more_body": False}
                                await disconnected.wait()
                                return {"type": "http.disconnect"}

                            async def send(_message):
                                return None

                            scope = {
                                "type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
                                "http_version": "1.1", "method": "POST", "scheme": "http",
                                "path": "/v1/images/generations", "raw_path": b"/v1/images/generations",
                                "query_string": b"", "root_path": "",
                                "headers": [
                                    (b"host", b"offline.test"),
                                    (b"content-type", b"application/json"),
                                    (b"content-length", str(len(payload)).encode()),
                                ],
                                "server": ("offline.test", 80), "client": ("127.0.0.1", 1),
                            }
                            task = asyncio.create_task(main.app(scope, receive, send))
                            self.assertTrue(await self._until(lambda: guard.stats()["admission_waiting"] == n))
                            disconnected.set()
                        await asyncio.sleep(0.45)
                        self.assertEqual(calls, [])
                        self.assertEqual(guard.stats()["admission_waiting"], 0)
                        release.set()
                        holder.join(timeout=2)
                        await asyncio.sleep(0.3)
                        self.assertEqual(calls, [])

                asyncio.run(scenario())
                self._assert_drained()

    def test_cancel_keeps_armed_permit_and_does_not_start_the_rest(self) -> None:
        import httpx
        import main
        from services.protocol.conversation import ImageOutput

        reset_for_tests(enabled=True)
        guard.configure_limits(
            max_active=1, max_heavy_waiting=1, max_generating=1, admission_wait_secs=3,
        )
        finish = threading.Event()
        started = threading.Event()
        calls: list[int] = []

        def body(req, index, total):
            calls.append(index)
            started.set()
            finish.wait(timeout=5)
            return [ImageOutput(
                kind="result", model=req.model, index=index, total=total,
                data=[{"b64_json": "b2ZmbGluZQ=="}],
            )]

        async def scenario() -> None:
            with mock.patch("api.ai.require_identity", return_value={"id": "offline", "name": "offline", "role": "user"}), \
                    mock.patch("services.protocol.conversation._generate_single_image_body", body):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=main.app),
                    base_url="http://offline.test",
                ) as client:
                    task = asyncio.create_task(client.post(
                        "/v1/images/generations",
                        json={"model": "gpt-image-2", "prompt": "offline", "size": "2048x2048", "n": 4, "stream": False},
                    ))
                    try:
                        self.assertTrue(await self._until(lambda: started.is_set() and guard.stats()["admission_waiting"] == 3))
                        self.assertEqual(len(calls), 1)
                        self.assertEqual(guard.stats()["generating"], 1)
                        task.cancel()
                        await asyncio.sleep(0.4)
                        self.assertEqual(len(calls), 1)
                        self.assertEqual(guard.stats()["admission_waiting"], 0)
                        self.assertEqual(guard.stats()["generating"], 1)
                        finish.set()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                        await asyncio.sleep(0.3)
                        self.assertEqual(len(calls), 1)
                    finally:
                        finish.set()

        asyncio.run(scenario())
        self._assert_drained()

    def test_n4_capacity_errors_stay_429_and_partial_success_is_not_rerun(self) -> None:
        import httpx

        import main
        from services.protocol.conversation import ImageGenerationError, ImageOutput

        cases = (
            ("full", True, None, 16, "free_large_image_admission_full"),
            ("timeout", False, None, 0, "free_large_image_admission_timeout"),
            ("mixed", False, 1, 0, "free_large_image_admission_full"),
        )
        for name, full, admission_cap, fillers_n, expected in cases:
            with self.subTest(name=name):
                reset_for_tests(enabled=True)
                guard.configure_limits(
                    max_active=1, max_heavy_waiting=1, max_generating=1,
                    admission_wait_secs=2 if full else 0.08,
                    **({} if admission_cap is None else {"max_admission_waiting": admission_cap}),
                )
                release = threading.Event()

                def hold() -> None:
                    with guard.bind_large_image_job("2K"):
                        release.wait(timeout=5)

                holder = threading.Thread(target=hold)
                holder.start()
                self.assertTrue(self._wait_until(lambda: guard.stats()["generating"] == 1))
                fillers = []
                if fillers_n:
                    def fill() -> None:
                        try:
                            with guard.bind_large_image_job("2K"):
                                pass
                        except LargeImageAdmissionError:
                            return

                    fillers = [threading.Thread(target=fill) for _ in range(fillers_n)]
                    for thread in fillers:
                        thread.start()
                    self.assertTrue(self._wait_until(lambda: guard.stats()["admission_waiting"] == fillers_n))
                calls: list[int] = []

                def body(*_args):
                    calls.append(1)
                    return []

                async def post(n: int):
                    with mock.patch("api.ai.require_identity", return_value={"id": "offline", "name": "offline", "role": "user"}), \
                            mock.patch("services.protocol.conversation._generate_single_image_body", body):
                        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://offline.test") as client:
                            response = await client.post(
                                "/v1/images/generations",
                                json={"model": "gpt-image-2", "prompt": "offline", "size": "2048x2048", "n": n, "stream": False},
                            )
                    return response

                try:
                    response = asyncio.run(post(4))
                    self.assertEqual(response.status_code, 429, response.text)
                    self.assertEqual(response.json()["error"]["code"], expected)
                    self.assertEqual(calls, [])
                finally:
                    release.set()
                    holder.join(timeout=2)
                    for thread in fillers:
                        thread.join(timeout=2)

        reset_for_tests(enabled=True)
        guard.configure_limits(admission_wait_secs=2)
        seen: list[int] = []

        def mixed(req, index, total):
            seen.append(index)
            if index == 2:
                raise ImageGenerationError("decode failed", status_code=500, error_type="server_error", code="decode_failed")
            return [ImageOutput(kind="result", model=req.model, index=index, total=total, data=[{"b64_json": "b2ZmbGluZQ=="}])]

        async def partial():
            with mock.patch("api.ai.require_identity", return_value={"id": "offline", "name": "offline", "role": "user"}), \
                    mock.patch("services.protocol.conversation._generate_single_image_body", mixed):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://offline.test") as client:
                    return await client.post(
                        "/v1/images/generations",
                        json={"model": "gpt-image-2", "prompt": "offline", "size": "1024x1024", "n": 4, "stream": False},
                    )

        response = asyncio.run(partial())
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(response.json()["data"]), 3)
        self.assertEqual(sorted(seen), [1, 2, 3, 4])
        self._assert_drained()

    def test_large_edit_waits_before_decode_and_known_1k_bypasses(self) -> None:
        import httpx

        import main
        from services.protocol import openai_v1_image_edit as edit

        reset_for_tests(enabled=True)
        guard.configure_limits(admission_wait_secs=3)
        release = threading.Event()

        def hold() -> None:
            with guard.bind_large_image_job("2K"):
                release.wait(timeout=6)

        holders = [threading.Thread(target=hold) for _ in range(15)]
        for thread in holders:
            thread.start()
        self.assertTrue(self._wait_until(lambda: guard.stats()["generating"] == 15))
        encoded_at: list[int] = []
        composed: list[int] = []

        def encode(images):
            encoded_at.append(guard.stats()["admission_waiting"])
            return ["encoded"]

        def composite(images, masks):
            composed.append(guard.stats()["admission_waiting"])
            return images

        def body(req, index, total):
            from services.protocol.conversation import ImageOutput
            return [ImageOutput(kind="result", model=req.model, index=index, total=total, data=[{"b64_json": "b2ZmbGluZQ=="}])]

        async def large_case():
            with mock.patch("api.ai.require_identity", return_value={"id": "offline", "name": "offline", "role": "user"}), \
                    mock.patch.object(edit, "encode_images", encode), \
                    mock.patch.object(edit, "_composite_mask", composite), \
                    mock.patch("services.protocol.conversation._generate_single_image_body", body):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://offline.test") as client:
                    task = asyncio.create_task(client.post(
                        "/v1/images/edits",
                        data={"model": "gpt-image-2", "prompt": "offline", "size": "2048x2048", "n": "1", "stream": "false"},
                        files={"image": ("offline.png", _png(3840, 2160), "image/png")},
                    ))
                    try:
                        self.assertTrue(await self._until(lambda: guard.stats()["admission_waiting"] == 1 or task.done()))
                        self.assertFalse(task.done(), "edit finished before admission")
                        self.assertEqual(composed, [])
                        self.assertEqual(encoded_at, [])
                        self.assertEqual(guard.stats()["admission_waiting"], 1)
                        release.set()
                        for thread in holders:
                            thread.join(timeout=2)
                        return await asyncio.wait_for(task, 3)
                    finally:
                        release.set()

        try:
            response = asyncio.run(large_case())
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(composed, [0])
            self.assertEqual(encoded_at, [0])
            self._assert_drained()
        finally:
            release.set()
            for thread in holders:
                thread.join(timeout=2)

        reset_for_tests(enabled=True)
        guard.configure_limits(admission_wait_secs=0)

        async def small_case():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://offline.test") as client:
                with mock.patch("api.ai.require_identity", return_value={"id": "offline", "name": "offline", "role": "user"}), \
                        mock.patch.object(edit, "encode_images", lambda images: ["small"]), \
                        mock.patch.object(edit, "_composite_mask", lambda images, masks: images), \
                        mock.patch("services.protocol.conversation._generate_single_image_body", body):
                    response = await client.post(
                        "/v1/images/edits",
                        data={"model": "gpt-image-2", "prompt": "offline", "size": "1024x1024", "n": "1", "stream": "false"},
                        files={"image": ("small.png", _png(1024, 1024), "image/png")},
                    )
            return response

        response = asyncio.run(small_case())
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(guard.stats()["peak_active"], 0)
        self.assertEqual(guard.stats()["peak_admission_waiting"], 0)
        self._assert_drained()

    def test_edit_cancel_does_not_start_upstream_or_pending_encode(self) -> None:
        import httpx
        import main
        from services.protocol import openai_v1_image_edit as edit
        from services.protocol.conversation import ImageOutput

        def image_body(req, index, total):
            return [ImageOutput(
                kind="result", model=req.model, index=index, total=total,
                data=[{"b64_json": "b2ZmbGluZQ=="}],
            )]

        for n in (1, 4):
            with self.subTest(stage="encode", n=n):
                reset_for_tests(enabled=True)
                guard.configure_limits(admission_wait_secs=3, wait_timeout_secs=5)
                release = threading.Event()
                entered = threading.Event()
                calls: list[int] = []
                encoded = {"n": 0}

                def encode(images):
                    encoded["n"] += 1
                    entered.set()
                    release.wait(timeout=6)
                    return ["encoded"]

                def run_body(req, index, total):
                    calls.append(index)
                    return image_body(req, index, total)

                async def scenario() -> None:
                    with mock.patch("api.ai.require_identity", return_value={"id": "offline", "name": "offline", "role": "user"}), \
                            mock.patch.object(edit, "encode_images", encode), \
                            mock.patch.object(edit, "_composite_mask", lambda images, masks: images), \
                            mock.patch("services.protocol.conversation._generate_single_image_body", run_body):
                        async with httpx.AsyncClient(
                            transport=httpx.ASGITransport(app=main.app),
                            base_url="http://offline.test",
                        ) as client:
                            task = asyncio.create_task(client.post(
                                "/v1/images/edits",
                                data={"model": "gpt-image-2", "prompt": "offline", "size": "2048x2048", "n": str(n), "stream": "false"},
                                files={"image": ("offline.png", _png(3840, 2160), "image/png")},
                            ))
                            try:
                                self.assertTrue(await self._until(entered.is_set))
                                self.assertEqual(calls, [])
                                self.assertEqual(guard.stats()["active"], 1)
                                task.cancel()
                                await asyncio.sleep(0.25)
                                self.assertEqual(calls, [])
                                self.assertEqual(guard.stats()["active"], 1)
                                release.set()
                                with self.assertRaises(asyncio.CancelledError):
                                    await task
                                self.assertTrue(await self._until(
                                    lambda: guard.stats()["active"] == 0 and guard.stats()["generating"] == 0,
                                ))
                                self.assertEqual(calls, [])
                                self.assertEqual(encoded["n"], 1)
                            finally:
                                release.set()

                asyncio.run(scenario())
                self._assert_drained()

        for n in (1, 4):
            with self.subTest(stage="before_encode", n=n):
                reset_for_tests(enabled=True)
                guard.configure_limits(admission_wait_secs=3, wait_timeout_secs=5)
                release = threading.Event()
                entered = threading.Event()
                calls: list[int] = []
                encoded = {"n": 0}

                def composite(images, masks):
                    entered.set()
                    release.wait(timeout=6)
                    return images

                async def scenario() -> None:
                    with mock.patch("api.ai.require_identity", return_value={"id": "offline", "name": "offline", "role": "user"}), \
                            mock.patch.object(edit, "encode_images", lambda images: encoded.__setitem__("n", encoded["n"] + 1) or ["encoded"]), \
                            mock.patch.object(edit, "_composite_mask", composite), \
                            mock.patch("services.protocol.conversation._generate_single_image_body", lambda *args: calls.append(1) or image_body(*args)):
                        async with httpx.AsyncClient(
                            transport=httpx.ASGITransport(app=main.app),
                            base_url="http://offline.test",
                        ) as client:
                            task = asyncio.create_task(client.post(
                                "/v1/images/edits",
                                data={"model": "gpt-image-2", "prompt": "offline", "size": "2048x2048", "n": str(n), "stream": "false"},
                                files={"image": ("offline.png", _png(3840, 2160), "image/png")},
                            ))
                            try:
                                self.assertTrue(await self._until(entered.is_set))
                                self.assertEqual(encoded["n"], 0)
                                self.assertEqual(guard.stats()["active"], 1)
                                task.cancel()
                                await asyncio.sleep(0.25)
                                self.assertEqual(guard.stats()["active"], 1)
                                self.assertEqual(encoded["n"], 0)
                                release.set()
                                with self.assertRaises(asyncio.CancelledError):
                                    await task
                                self.assertTrue(await self._until(
                                    lambda: guard.stats()["active"] == 0 and guard.stats()["generating"] == 0,
                                ))
                                self.assertEqual(encoded["n"], 0)
                                self.assertEqual(calls, [])
                            finally:
                                release.set()

                asyncio.run(scenario())
                self._assert_drained()

        for n in (1, 4):
            with self.subTest(stage="heavy_wait_disconnect", n=n):
                reset_for_tests(enabled=True)
                guard.configure_limits(admission_wait_secs=3, wait_timeout_secs=5)
                release = threading.Event()
                calls: list[int] = []
                encoded = {"n": 0}

                def hold() -> None:
                    with guard.heavy_image_work(discovered_pixels=3840 * 2160, stage="holder"):
                        release.wait(timeout=8)

                holders = [threading.Thread(target=hold) for _ in range(5)]
                for thread in holders:
                    thread.start()
                self.assertTrue(self._wait_until(lambda: guard.stats()["active"] == 5))

                async def scenario() -> None:
                    request = httpx.Request(
                        "POST",
                        "http://offline.test/v1/images/edits",
                        data={"model": "gpt-image-2", "prompt": "offline", "size": "2048x2048", "n": str(n), "stream": "false"},
                        files={"image": ("offline.png", _png(3840, 2160), "image/png")},
                    )
                    payload = await request.aread()
                    disconnected = asyncio.Event()
                    received = {"done": False}
                    statuses: list[int] = []

                    async def receive():
                        if not received["done"]:
                            received["done"] = True
                            return {"type": "http.request", "body": payload, "more_body": False}
                        await disconnected.wait()
                        return {"type": "http.disconnect"}

                    async def send(message):
                        if message["type"] == "http.response.start":
                            statuses.append(message["status"])

                    scope = {
                        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
                        "http_version": "1.1", "method": "POST", "scheme": "http",
                        "path": "/v1/images/edits", "raw_path": b"/v1/images/edits",
                        "query_string": b"", "root_path": "",
                        "headers": [(key.lower(), value) for key, value in request.headers.raw],
                        "server": ("offline.test", 80), "client": ("127.0.0.1", 1),
                    }
                    with mock.patch("api.ai.require_identity", return_value={"id": "offline", "name": "offline", "role": "user"}), \
                            mock.patch.object(edit, "encode_images", lambda images: encoded.__setitem__("n", encoded["n"] + 1) or ["encoded"]), \
                            mock.patch.object(edit, "_composite_mask", lambda images, masks: images), \
                            mock.patch("services.protocol.conversation._generate_single_image_body", lambda *args: calls.append(1) or image_body(*args)):
                        task = asyncio.create_task(main.app(scope, receive, send))
                        try:
                            self.assertTrue(await self._until(lambda: guard.stats()["heavy_waiting"] == 1))
                            self.assertEqual(encoded["n"], 0)
                            self.assertEqual(calls, [])
                            disconnected.set()
                            await asyncio.sleep(0.45)
                            self.assertEqual(encoded["n"], 0)
                            self.assertEqual(calls, [])
                            self.assertEqual(guard.stats()["heavy_waiting"], 0)
                            release.set()
                            for thread in holders:
                                thread.join(timeout=2)
                            await asyncio.wait_for(task, 4)
                            self.assertEqual(statuses, [429])
                            self.assertEqual(encoded["n"], 0)
                            self.assertEqual(calls, [])
                        finally:
                            disconnected.set()
                            release.set()

                try:
                    asyncio.run(scenario())
                    self._assert_drained()
                finally:
                    release.set()
                    for thread in holders:
                        thread.join(timeout=2)

    async def _until(self, predicate, timeout: float = 3.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            await asyncio.sleep(0.005)
        return bool(predicate())

    def _assert_drained(self) -> None:
        self.assertEqual(guard.token_snapshot(), [], guard.token_snapshot())
        self.assertEqual(guard.stats()["admission_waiting"], 0)
        self.assertEqual(guard.stats()["generating"], 0)
        self.assertEqual(guard.stats()["active"], 0)
        self.assertIsNone(guard.consistency())

    def _wait_until(self, predicate, timeout: float = 2.0) -> bool:
        return ConcurrencyTests._wait_until(self, predicate, timeout)


class _StreamingBackend:
    """Session shaped like the real client. ``download_image_bytes`` is the production method."""

    def __init__(self, tail_active: list[int] | None = None, header: bytes | None = None) -> None:
        self.session = self
        self.tail_active = tail_active if tail_active is not None else []
        self.header = header if header is not None else _png(3840, 2160)

    def get(self, url, timeout=120, stream=True):  # noqa: ARG002
        tail_active = self.tail_active
        header = self.header

        class _Response:
            status_code = 200
            headers: dict[str, str] = {}

            def iter_content(self, chunk_size=65536):  # noqa: ARG002
                yield header[:64]
                tail_active.append(guard.stats()["active"])
                yield b"tail"

            def close(self) -> None:
                return None

        return _Response()

    def download_image_bytes(self, urls, size=None):
        from services.openai_backend_api import OpenAIBackendAPI

        return OpenAIBackendAPI.download_image_bytes(self, urls, size=size)

    def close(self) -> None:
        return None


if __name__ == "__main__":
    unittest.main()
