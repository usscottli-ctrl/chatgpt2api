"""Offline tests for the free-pool large-image heavy-memory guard.

These tests do not call a live model, paid generation, or the shared size proxy.
They drive the real guard through stream materialization and `_generate_single_image`.
"""

from __future__ import annotations

import asyncio
import base64
import os
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
        guard.configure_limits(max_active=5, max_waiting=10, wait_timeout_secs=2.0)

    def tearDown(self) -> None:
        os.environ.pop("CHATGPT2API_FREE_LARGE_IMAGE_GUARD", None)
        stats = guard.stats()
        reset_for_tests(enabled=False)
        self.assertEqual(stats["active"], 0, stats)
        self.assertEqual(stats["waiting"], 0, stats)


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
