import contextlib
import io
import json
import unittest
import urllib.error
from collections.abc import Callable

from adapters.outbound.upstream.google_cloudcode import GoogleCloudCodeAdapter
from config import UpstreamConfig
from core.domain.entities import (
    ChatMessage,
    ChatRequest,
    GenerationParameters,
    ImagePart,
    Role,
    ToolCallPart,
    ToolDefinition,
    ToolResultPart,
)
from core.domain.exceptions import UpstreamServiceError, UpstreamTimeoutError


@contextlib.contextmanager
def patched_urlopen(handler: Callable[[int], object]):
    """Replace urllib.request.urlopen with a counting stub.

    The handler receives the 1-based attempt number, letting each test assert
    exactly how many requests were made.
    """
    import adapters.outbound.upstream.google_cloudcode as module

    calls = {"n": 0}
    original = module.urllib.request.urlopen

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        return handler(calls["n"])

    module.urllib.request.urlopen = fake_urlopen
    try:
        yield calls
    finally:
        module.urllib.request.urlopen = original


def make_adapter(**overrides) -> GoogleCloudCodeAdapter:
    defaults = {
        "base_url": "https://mock.google.com",
        "project_id": "test-project",
        "user_agent": "test-agent",
        "max_retries": 0,
    }
    defaults.update(overrides)
    return GoogleCloudCodeAdapter(UpstreamConfig(**defaults), sleep=lambda _: None)


class TestPayloadMapping(unittest.TestCase):
    def setUp(self):
        self.adapter = make_adapter()

    def test_system_and_user_messages(self):
        request = ChatRequest(
            model="gemini-3.6-flash-low",
            messages=(
                ChatMessage.from_text(Role.SYSTEM, "You are a helper."),
                ChatMessage.from_text(Role.USER, "What is 2+2?"),
            ),
            parameters=GenerationParameters(temperature=0.2, max_output_tokens=100),
        )
        payload = self.adapter._map_to_upstream_payload(request)
        self.assertEqual(payload["project"], "test-project")
        self.assertEqual(payload["model"], "gemini-3.6-flash-low")

        inner = payload["request"]
        self.assertEqual(inner["systemInstruction"]["parts"][0]["text"], "You are a helper.")
        self.assertEqual(inner["contents"][0]["role"], "user")
        self.assertEqual(inner["generationConfig"]["maxOutputTokens"], 100)

    def test_assistant_role_maps_to_model(self):
        request = ChatRequest(
            model="m",
            messages=(
                ChatMessage.from_text(Role.USER, "a"),
                ChatMessage.from_text(Role.ASSISTANT, "b"),
            ),
        )
        contents = self.adapter._map_to_upstream_payload(request)["request"]["contents"]
        self.assertEqual([entry["role"] for entry in contents], ["user", "model"])

    def test_image_part(self):
        request = ChatRequest(
            model="m",
            messages=(ChatMessage(role=Role.USER, parts=(ImagePart("image/png", "QUJD"),)),),
        )
        parts = self.adapter._map_to_upstream_payload(request)["request"]["contents"][0]["parts"]
        self.assertEqual(parts[0]["inlineData"], {"mimeType": "image/png", "data": "QUJD"})

    def test_tool_call_and_result_mapping(self):
        request = ChatRequest(
            model="m",
            messages=(
                ChatMessage(role=Role.ASSISTANT, parts=(ToolCallPart("c1", "f", '{"a":1}'),)),
                ChatMessage(role=Role.TOOL, parts=(ToolResultPart("c1", "f", "ok"),)),
            ),
        )
        contents = self.adapter._map_to_upstream_payload(request)["request"]["contents"]
        self.assertEqual(contents[0]["parts"][0]["functionCall"], {"name": "f", "args": {"a": 1}})
        self.assertEqual(
            contents[1]["parts"][0]["functionResponse"],
            {"name": "f", "response": {"result": "ok"}},
        )

    def test_malformed_tool_arguments_do_not_crash(self):
        request = ChatRequest(
            model="m",
            messages=(
                ChatMessage(role=Role.ASSISTANT, parts=(ToolCallPart("c1", "f", "not json"),)),
            ),
        )
        parts = self.adapter._map_to_upstream_payload(request)["request"]["contents"][0]["parts"]
        self.assertEqual(parts[0]["functionCall"]["args"], {})

    def test_tools_are_declared(self):
        request = ChatRequest(
            model="m",
            messages=(ChatMessage.from_text(Role.USER, "x"),),
            tools=(ToolDefinition("f", "does f", {"type": "object"}),),
        )
        tools = self.adapter._map_to_upstream_payload(request)["request"]["tools"]
        self.assertEqual(tools[0]["functionDeclarations"][0]["name"], "f")

    def test_no_tools_key_when_absent(self):
        request = ChatRequest(model="m", messages=(ChatMessage.from_text(Role.USER, "x"),))
        self.assertNotIn("tools", self.adapter._map_to_upstream_payload(request)["request"])

    def test_base_url_has_no_trailing_slash(self):
        adapter = make_adapter(base_url="https://x.test/")
        self.assertEqual(adapter._config.base_url, "https://x.test/")


class TestSSEFraming(unittest.TestCase):
    """Regression coverage for the SSE terminator that used to raise a 502."""

    def setUp(self):
        self.adapter = make_adapter()

    def test_done_sentinel_yields_no_chunk(self):
        payloads = list(self.adapter._iter_sse_payloads("data: [DONE]"))
        self.assertEqual(payloads, ["[DONE]"])

    def test_parses_standard_data_lines(self):
        self.assertEqual(list(self.adapter._iter_sse_payloads('data: {"a":1}')), ['{"a":1}'])

    def test_tolerates_missing_space(self):
        self.assertEqual(list(self.adapter._iter_sse_payloads("data:{}")), ["{}"])

    def test_ignores_comments_and_event_lines(self):
        self.assertEqual(list(self.adapter._iter_sse_payloads(": keep-alive")), [])
        self.assertEqual(list(self.adapter._iter_sse_payloads("event: message")), [])
        self.assertEqual(list(self.adapter._iter_sse_payloads("")), [])

    def test_ignores_id_and_retry_fields(self):
        self.assertEqual(list(self.adapter._iter_sse_payloads("id: 42")), [])
        self.assertEqual(list(self.adapter._iter_sse_payloads("retry: 1000")), [])


class TestChunkParsing(unittest.TestCase):
    def setUp(self):
        self.adapter = make_adapter()

    def test_text_finish_and_usage(self):
        delta = self.adapter._parse_chunk(
            {
                "response": {
                    "candidates": [
                        {
                            "content": {"role": "model", "parts": [{"text": "Hello"}]},
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 5,
                        "candidatesTokenCount": 1,
                        "totalTokenCount": 6,
                        "thoughtsTokenCount": 2,
                    },
                }
            }
        )
        self.assertEqual(delta.text, "Hello")
        self.assertEqual(delta.finish_reason, "stop")
        self.assertEqual(delta.usage.total_tokens, 6)
        self.assertEqual(delta.usage.thinking_tokens, 2)

    def test_thought_parts_become_reasoning_not_content(self):
        delta = self.adapter._parse_chunk(
            {
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "thinking", "thought": True},
                                    {"text": "answer"},
                                ]
                            }
                        }
                    ]
                }
            }
        )
        self.assertEqual(delta.reasoning, "thinking")
        self.assertEqual(delta.text, "answer")

    def test_function_call_becomes_tool_call(self):
        delta = self.adapter._parse_chunk(
            {
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "functionCall": {
                                            "id": "c1",
                                            "name": "f",
                                            "args": {"a": 1},
                                        }
                                    }
                                ]
                            },
                            "finishReason": "STOP",
                        }
                    ]
                }
            }
        )
        self.assertEqual(delta.tool_calls[0].function_name, "f")
        self.assertEqual(delta.tool_calls[0].arguments, '{"a": 1}')

    def test_finish_reason_vocabulary_is_mapped_to_openai(self):
        cases = {
            "STOP": "stop",
            "MAX_TOKENS": "length",
            "SAFETY": "content_filter",
            "RECITATION": "content_filter",
        }
        for upstream, expected in cases.items():
            delta = self.adapter._parse_chunk(
                {
                    "response": {
                        "candidates": [
                            {"content": {"parts": [{"text": "x"}]}, "finishReason": upstream}
                        ]
                    }
                }
            )
            self.assertEqual(delta.finish_reason, expected, upstream)

    def test_empty_chunk_returns_none(self):
        self.assertIsNone(self.adapter._parse_chunk({"response": {}}))
        self.assertIsNone(self.adapter._parse_chunk({}))

    def test_multiple_candidates_are_merged(self):
        delta = self.adapter._parse_chunk(
            {
                "response": {
                    "candidates": [
                        {"content": {"parts": [{"text": "a"}]}},
                        {"content": {"parts": [{"text": "b"}]}},
                    ]
                }
            }
        )
        self.assertEqual(delta.text, "ab")


class TestModelMapping(unittest.TestCase):
    def setUp(self):
        self.adapter = make_adapter()

    def test_maps_catalog_fields(self):
        models = self.adapter._map_models(
            {
                "models": {
                    "gemini-x": {
                        "displayName": "Gemini X",
                        "modelProvider": "google",
                        "inputTokenLimit": 1000,
                        "maxOutputTokens": 200,
                        "supportsThinking": True,
                        "quotaInfo": {"remainingFraction": 0.25, "resetTime": "later"},
                    }
                }
            }
        )
        model = models[0]
        self.assertEqual(model.context_window, 1000)
        self.assertEqual(model.max_output_tokens, 200)
        self.assertTrue(model.supports_thinking)
        self.assertEqual(model.quota_info.remaining_fraction, 0.25)

    def test_handles_missing_fields(self):
        model = self.adapter._map_models({"models": {"m": {}}})[0]
        self.assertEqual(model.display_name, "m")
        self.assertIsNone(model.quota_info)

    def test_handles_empty_catalog(self):
        self.assertEqual(self.adapter._map_models({}), ())
        self.assertEqual(self.adapter._map_models({"models": {}}), ())


class FakeResponse:
    """Stands in for an urlopen response.

    Accepts either a list of text lines (SSE streaming) or raw bytes (a buffered
    JSON body), because the two upstream calls are read differently.
    """

    def __init__(self, lines):
        if isinstance(lines, (bytes, bytearray)):
            payload = bytes(lines)
        else:
            payload = "".join(lines).encode("utf-8")
        self._buffer = io.BytesIO(payload)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        yield from self._buffer

    def read(self) -> bytes:
        return self._buffer.read()


class TestRetryPolicy(unittest.TestCase):
    @staticmethod
    def _request():
        return ChatRequest(model="m", messages=(ChatMessage.from_text(Role.USER, "x"),))

    def test_retries_on_503_then_succeeds(self):
        adapter = make_adapter(max_retries=2)
        slept: list[float] = []
        adapter._sleep = slept.append
        attempts = {"n": 0}

        def handler(_attempt: int) -> FakeResponse:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise urllib.error.HTTPError("u", 503, "busy", {}, io.BytesIO(b""))
            return FakeResponse(
                ['data: {"response": {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}}\n']
            )

        with patched_urlopen(handler):
            deltas = list(adapter.stream_generate("tok", self._request()))

        self.assertEqual(deltas[0].text, "ok")
        self.assertEqual(attempts["n"], 3)
        self.assertEqual(len(slept), 2, "should back off twice")
        self.assertGreater(slept[1], slept[0], "backoff must grow exponentially")

    def test_does_not_retry_on_400(self):
        adapter = make_adapter(max_retries=3)
        attempts = {"n": 0}

        def handler(_attempt: int):
            attempts["n"] += 1
            raise urllib.error.HTTPError("u", 400, "bad", {}, io.BytesIO(b"nope"))

        with patched_urlopen(handler), self.assertRaises(UpstreamServiceError):
            adapter.fetch_models("tok")

        self.assertEqual(attempts["n"], 1, "client errors must not be retried")

    def test_gives_up_after_max_retries(self):
        adapter = make_adapter(max_retries=1)
        adapter._sleep = lambda _: None
        attempts = {"n": 0}

        def handler(_attempt: int):
            attempts["n"] += 1
            raise urllib.error.HTTPError("u", 429, "slow down", {}, io.BytesIO(b""))

        with patched_urlopen(handler), self.assertRaises(UpstreamServiceError):
            adapter.fetch_models("tok")

        self.assertEqual(attempts["n"], 2, "one retry means two attempts")

    def test_backoff_is_capped(self):
        adapter = make_adapter(max_retries=8, retry_base_delay=1.0, retry_max_delay=4.0)
        slept: list[float] = []
        adapter._sleep = slept.append

        error = urllib.error.HTTPError("u", 503, "", {}, io.BytesIO(b""))
        with patched_urlopen(lambda _a: _raise(error)), self.assertRaises(UpstreamServiceError):
            adapter.fetch_models("tok")

        self.assertTrue(all(delay <= 4.0 for delay in slept), slept)
        self.assertEqual(slept, sorted(slept), "delays must be non-decreasing")

    def test_timeout_error_is_retryable(self):
        error = UpstreamTimeoutError("too slow", timeout=1.0)
        self.assertTrue(error.retryable)
        self.assertEqual(error.status_code, 504)


def _raise(error: Exception):
    raise error


class TestCatalogCache(unittest.TestCase):
    """Regression: every model read cost a live round trip to Google."""

    @staticmethod
    def _adapter(ttl: float, clock):
        return GoogleCloudCodeAdapter(
            UpstreamConfig(base_url="https://mock", project_id="p", max_retries=0, catalog_ttl=ttl),
            sleep=lambda _: None,
            monotonic=clock,
        )

    def test_repeat_reads_hit_the_upstream_once(self):
        calls = {"n": 0}
        payload = {"models": {"m": {"displayName": "M"}}}

        def handler(_attempt):
            calls["n"] += 1
            return FakeResponse(json.dumps(payload).encode())

        adapter = self._adapter(ttl=300.0, clock=lambda: 0.0)
        with patched_urlopen(handler):
            first = adapter.fetch_models("tok")
            second = adapter.fetch_models("tok")
        self.assertEqual(first, second)
        self.assertEqual(calls["n"], 1, "second read should have been served from cache")

    def test_cache_expires_after_the_ttl(self):
        calls = {"n": 0}
        payload = {"models": {"m": {"displayName": "M"}}}
        now = {"t": 0.0}

        def handler(_attempt):
            calls["n"] += 1
            return FakeResponse(json.dumps(payload).encode())

        adapter = self._adapter(ttl=10.0, clock=lambda: now["t"])
        with patched_urlopen(handler):
            adapter.fetch_models("tok")
            now["t"] = 5.0
            adapter.fetch_models("tok")
            self.assertEqual(calls["n"], 1, "still inside the TTL")
            now["t"] = 11.0
            adapter.fetch_models("tok")
        self.assertEqual(calls["n"], 2, "TTL elapsed, should re-fetch")

    def test_failed_read_is_not_cached(self):
        calls = {"n": 0}
        payload = {"models": {"m": {"displayName": "M"}}}

        def handler(_attempt):
            calls["n"] += 1
            if calls["n"] == 1:
                raise urllib.error.HTTPError("u", 400, "bad", {}, io.BytesIO(b""))
            return FakeResponse(json.dumps(payload).encode())

        adapter = self._adapter(ttl=300.0, clock=lambda: 0.0)
        with patched_urlopen(handler):
            with self.assertRaises(UpstreamServiceError):
                adapter.fetch_models("tok")
            self.assertEqual(adapter.fetch_models("tok")[0].id, "m")
        self.assertEqual(calls["n"], 2, "a failure must not poison the cache")

    def test_zero_ttl_disables_caching(self):
        calls = {"n": 0}
        payload = {"models": {"m": {"displayName": "M"}}}

        def handler(_attempt):
            calls["n"] += 1
            return FakeResponse(json.dumps(payload).encode())

        adapter = self._adapter(ttl=0.0, clock=lambda: 0.0)
        with patched_urlopen(handler):
            adapter.fetch_models("tok")
            adapter.fetch_models("tok")
        self.assertEqual(calls["n"], 2)


class TestSseEndToEnd(unittest.TestCase):
    @staticmethod
    def _request():
        return ChatRequest(model="m", messages=(ChatMessage.from_text(Role.USER, "x"),))

    def test_done_terminates_stream_without_error(self):
        """Regression: a `data: [DONE]` sentinel used to become a 502."""
        adapter = make_adapter()
        response = FakeResponse(
            [
                'data: {"response": {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}}\n',
                "\n",
                "data: [DONE]\n",
                "\n",
            ]
        )
        with patched_urlopen(lambda _a: response):
            deltas = list(adapter.stream_generate("tok", self._request()))
        self.assertEqual([delta.text for delta in deltas], ["hi"])

    def test_malformed_chunk_is_skipped_not_fatal(self):
        adapter = make_adapter()
        response = FakeResponse(
            [
                "data: {not json}\n",
                'data: {"response": {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}}\n',
            ]
        )
        with patched_urlopen(lambda _a: response):
            deltas = list(adapter.stream_generate("tok", self._request()))
        self.assertEqual([delta.text for delta in deltas], ["ok"])


if __name__ == "__main__":
    unittest.main()
