import json
import socket
import threading
import unittest
from collections.abc import Iterator

from adapters.inbound.http.openai_adapter import OpenAIProtocolTranslator
from adapters.inbound.http.server import GatewayHandler, create_http_server, status_for
from config import ProtocolConfig, ServerConfig
from core.domain.entities import (
    ChatResponse,
    ModelInfo,
    QuotaInfo,
    StreamDelta,
    TokenUsage,
)
from core.domain.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ModelNotFoundError,
    UpstreamServiceError,
    ValidationError,
)
from core.ports.inbound import AuthUseCase, ChatUseCase, ModelCatalogUseCase

CHAT = "/v1/chat/completions"


class FakeAuth(AuthUseCase):
    def ensure_authenticated(self) -> str:
        return "token"

    def login_interactive(self) -> None:
        pass

    def logout(self) -> None:
        pass


class WorkingChat(ChatUseCase):
    def complete(self, request) -> ChatResponse:
        return ChatResponse(
            id="chatcmpl-test",
            model=request.model,
            content="Hello world",
            created_at=1700000000,
            finish_reason="stop",
            usage=TokenUsage(5, 2, 7),
        )

    def complete_stream(self, request) -> Iterator[StreamDelta]:
        def generate():
            for piece in ("Hello", " ", "world"):
                yield StreamDelta(text=piece)
            yield StreamDelta(finish_reason="stop", usage=TokenUsage(5, 2, 7))

        return generate()


class FailingChat(ChatUseCase):
    """Fails eagerly at call time, the way ChatService.complete_stream does."""

    def __init__(self, error: Exception):
        self.error = error

    def complete(self, request):
        raise self.error

    def complete_stream(self, request):
        raise self.error


class MidStreamFailingChat(ChatUseCase):
    def complete(self, request):
        raise AssertionError("sync path is not exercised here")

    def complete_stream(self, request):
        def generate():
            yield StreamDelta(text="partial")
            raise UpstreamServiceError("upstream connection reset", status_code=502)

        return generate()


class FakeCatalog(ModelCatalogUseCase):
    def __init__(self, models: tuple[ModelInfo, ...] | None = None, error=None):
        self.models = models or (
            ModelInfo(id="test-model", display_name="Test", provider="google"),
            ModelInfo(
                id="quota-model",
                display_name="Quota",
                provider="google",
                quota_info=QuotaInfo(0.5, "later"),
            ),
        )
        self.error = error

    def list_models(self):
        if self.error:
            raise self.error
        return self.models

    def get_model(self, model_id: str) -> ModelInfo:
        for model in self.list_models():
            if model.id == model_id:
                return model
        raise ModelNotFoundError(model_id)


class RawClient:
    """Speaks HTTP over a raw socket so framing bugs cannot hide behind a client."""

    def __init__(self, port: int):
        self.port = port

    def request(
        self,
        method: str = "POST",
        path: str = CHAT,
        body: bytes | None = None,
        headers: dict | None = None,
        timeout: float = 5.0,
    ):
        lines = [f"{method} {path} HTTP/1.1", f"Host: 127.0.0.1:{self.port}"]
        for key, value in (headers or {}).items():
            lines.append(f"{key}: {value}")
        if body is not None:
            lines.append(f"Content-Length: {len(body)}")
        elif method in ("POST", "PUT", "PATCH"):
            lines.append("Content-Length: 0")
        request_bytes = ("\r\n".join(lines) + "\r\n\r\n").encode() + (body or b"")

        received = bytearray()
        with socket.create_connection(("127.0.0.1", self.port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall(request_bytes)
            # Respect Content-Length when present (HTTP/1.1 keep-alive means
            # there is no EOF to wait for); otherwise read until the server
            # closes, which is how close-delimited SSE responses terminate.
            header_end = -1
            content_length: int | None = None

            def body_so_far() -> int:
                return len(received) - (header_end + 4) if header_end != -1 else 0

            while True:
                if content_length is not None and body_so_far() >= content_length:
                    break
                data = sock.recv(4096)
                if not data:
                    break
                received.extend(data)
                if content_length is None and header_end == -1:
                    header_end = received.find(b"\r\n\r\n")
                    if header_end != -1:
                        head = received[:header_end].decode("latin-1")
                        for line in head.split("\r\n")[1:]:
                            if line.lower().startswith("content-length:"):
                                content_length = int(line.split(":", 1)[1].strip())
        return _Response(bytes(received))

    def post_json(self, payload: dict, headers: dict | None = None, **kwargs):
        merged = {"Content-Type": "application/json", **(headers or {})}
        return self.request(body=json.dumps(payload).encode(), headers=merged, **kwargs)


class _Response:
    def __init__(self, raw: bytes):
        self.raw = raw
        head, _, body = raw.partition(b"\r\n\r\n")
        self.head = head.decode("utf-8", errors="replace")
        self.body = body.decode("utf-8", errors="replace")

    @property
    def status(self) -> int:
        return int(self.head.split()[1])

    @property
    def response_count(self) -> int:
        return self.raw.count(b"HTTP/1.")

    def json(self):
        return json.loads(self.body)

    def events(self):
        return [
            json.loads(line[5:].strip())
            for line in self.body.splitlines()
            if line.startswith("data: {")
        ]


class ServerTestCase(unittest.TestCase):
    chat: ChatUseCase = WorkingChat()
    catalog: ModelCatalogUseCase = FakeCatalog()
    server_config: ServerConfig = ServerConfig()

    def setUp(self):
        self.translator = OpenAIProtocolTranslator(ProtocolConfig())
        self.server = create_http_server(
            host="127.0.0.1",
            port=0,
            translator=self.translator,
            chat_use_case=self.chat,
            catalog_use_case=self.catalog,
            config=self.server_config,
        )
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = RawClient(self.port)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


# --------------------------------------------------------------------------
# The regressions that motivated this suite.
# --------------------------------------------------------------------------


class TestStreamTermination(ServerTestCase):
    """Regression: every streamed response used to hang forever."""

    def test_stream_closes_and_delivers_done(self):
        response = self.client.post_json(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}
        )
        self.assertEqual(response.status, 200)
        self.assertIn("text/event-stream", response.head)
        self.assertNotIn("keep-alive", response.head.lower())
        self.assertEqual(response.response_count, 1)
        self.assertTrue(response.body.rstrip().endswith("data: [DONE]"))

    def test_stream_content_is_complete(self):
        response = self.client.post_json(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}
        )
        text = "".join(
            event["choices"][0]["delta"].get("content", "") for event in response.events()
        )
        self.assertEqual(text, "Hello world")

    def test_stream_reports_usage_on_final_chunk(self):
        response = self.client.post_json(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}
        )
        self.assertEqual(response.events()[-1]["usage"]["total_tokens"], 7)

    def test_include_usage_adds_a_trailing_usage_chunk(self):
        response = self.client.post_json(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        )
        events = response.events()
        self.assertEqual(events[-1]["choices"], [])
        self.assertEqual(events[-1]["usage"]["total_tokens"], 7)


class TestStreamPreflightFailure(ServerTestCase):
    chat = FailingChat(AuthenticationError("OAuth token has expired"))

    def test_auth_failure_is_a_single_401(self):
        response = self.client.post_json(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}
        )
        self.assertEqual(response.status, 401)
        self.assertEqual(response.response_count, 1, "must not append a second response")
        self.assertNotIn("text/event-stream", response.head)
        self.assertEqual(response.json()["error"]["type"], "authentication_error")


class TestMidStreamFailure(ServerTestCase):
    chat = MidStreamFailingChat()

    def test_failure_after_commit_is_reported_in_band(self):
        response = self.client.post_json(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}
        )
        self.assertEqual(response.status, 200)
        self.assertEqual(response.response_count, 1)
        events = response.events()
        self.assertEqual(events[0]["choices"][0]["delta"]["content"], "partial")
        self.assertIn("error", events[-1])
        self.assertIn("upstream connection reset", events[-1]["error"]["message"])
        self.assertTrue(response.body.rstrip().endswith("data: [DONE]"))


# --------------------------------------------------------------------------
# Security
# --------------------------------------------------------------------------


class TestApiKeyAuth(ServerTestCase):
    server_config = ServerConfig(api_key="s3cret")

    def test_health_needs_no_key(self):
        self.assertEqual(self.client.request("GET", "/health").status, 200)

    def test_missing_key_is_rejected(self):
        response = self.client.post_json(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        )
        self.assertEqual(response.status, 401)
        self.assertEqual(response.json()["error"]["type"], "invalid_api_key")

    def test_wrong_key_is_rejected(self):
        response = self.client.post_json(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer nope"},
        )
        self.assertEqual(response.status, 401)

    def test_correct_key_is_accepted(self):
        response = self.client.post_json(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer s3cret"},
        )
        self.assertEqual(response.status, 200)

    def test_models_endpoint_is_also_protected(self):
        self.assertEqual(self.client.request("GET", "/v1/models").status, 401)
        self.assertEqual(
            self.client.request(
                "GET", "/v1/models", headers={"Authorization": "Bearer s3cret"}
            ).status,
            200,
        )


# --------------------------------------------------------------------------
# Framing, limits, routing
# --------------------------------------------------------------------------


class TestBodyLimits(ServerTestCase):
    def test_non_integer_content_length(self):
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as sock:
            sock.sendall(f"POST {CHAT} HTTP/1.1\r\nHost: x\r\nContent-Length: abc\r\n\r\n".encode())
            sock.settimeout(5)
            data = b""
            while b"\r\n\r\n" not in data or len(data.split(b"\r\n\r\n", 1)[1]) < 115:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
        self.assertIn(b"400", data.split(b"\r\n")[0])
        self.assertIn(b"validation_error", data)

    def test_empty_body_is_rejected(self):
        response = self.client.request(body=b"")
        self.assertEqual(response.status, 400)

    def test_invalid_json_is_400(self):
        response = self.client.request(
            body=b"not json!", headers={"Content-Type": "application/json"}
        )
        self.assertEqual(response.status, 400)
        self.assertEqual(response.json()["error"]["type"], "validation_error")

    def test_oversized_body_against_constrained_server(self):
        server = create_http_server(
            "127.0.0.1",
            0,
            self.translator,
            self.chat,
            self.catalog,
            ServerConfig(max_request_bytes=32),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = RawClient(server.server_address[1])
            body = json.dumps(
                {"model": "m", "messages": [{"role": "user", "content": "x"}]}
            ).encode()
            response = client.request(body=body)
            self.assertEqual(response.status, 400)
            self.assertIn("limit", response.json()["error"]["message"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


class TestRouting(ServerTestCase):
    def test_health(self):
        self.assertEqual(self.client.request("GET", "/health").json(), {"status": "ok"})

    def test_models_list(self):
        response = self.client.request("GET", "/v1/models")
        self.assertEqual(response.status, 200)
        self.assertEqual(
            {entry["id"] for entry in response.json()["data"]}, {"test-model", "quota-model"}
        )

    def test_models_list_tolerates_query_string(self):
        self.assertEqual(self.client.request("GET", "/v1/models?limit=1").status, 200)

    def test_gateway_model_details_route(self):
        response = self.client.request("GET", "/v1/gateway/models")
        self.assertEqual(response.status, 200)
        payload = response.json()
        self.assertEqual({m["id"] for m in payload["models"]}, {"test-model", "quota-model"})
        entry = next(m for m in payload["models"] if m["id"] == "quota-model")
        self.assertEqual(entry["quota"]["remainingFraction"], 0.5)
        self.assertIn("contextWindow", entry)

    def test_single_model_lookup(self):
        response = self.client.request("GET", "/v1/models/test-model")
        self.assertEqual(response.json()["id"], "test-model")

    def test_unknown_model_is_404(self):
        response = self.client.request("GET", "/v1/models/nope")
        self.assertEqual(response.status, 404)
        self.assertEqual(response.json()["error"]["type"], "model_not_found")

    def test_unversioned_aliases(self):
        self.assertEqual(self.client.request("GET", "/models").status, 200)

    def test_unknown_path_is_404(self):
        self.assertEqual(self.client.request("GET", "/nope").status, 404)

    def test_wrong_method_is_405(self):
        self.assertEqual(self.client.request("GET", CHAT).status, 405)

    def test_options_preflight(self):
        self.assertEqual(self.client.request("OPTIONS", CHAT).status, 204)


class TestSyncPath(ServerTestCase):
    def test_completion(self):
        response = self.client.post_json(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        )
        self.assertEqual(response.status, 200)
        self.assertIn("Content-Length", response.head)
        self.assertEqual(response.response_count, 1)
        payload = response.json()
        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(payload["choices"][0]["message"]["content"], "Hello world")

    def test_error_body_never_leaks_internals(self):
        class Exploding(WorkingChat):
            def complete(self, request):
                raise RuntimeError("secret internal detail")

        server = create_http_server("127.0.0.1", 0, self.translator, Exploding(), self.catalog)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = RawClient(server.server_address[1])
            response = client.post_json(
                {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
            )
            self.assertEqual(response.status, 500)
            self.assertNotIn("secret internal detail", response.body)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


class TestRequestLogging(ServerTestCase):
    """Regression: log_message passed 2 values to a 3-placeholder format.

    stdlib supplies a format string plus a variable number of arguments, so
    interpolating first and logging the result as one value is the only correct
    form. The broken version raised inside logging on every single request.
    """

    def test_standard_library_call_shapes_do_not_raise(self):
        with self.assertLogs("adapters.inbound.http.server", level="INFO") as captured:
            self.client.request("GET", "/health")
            self.client.request("GET", "/v1/models")
            self.client.request("GET", "/nope")
        self.assertTrue(captured.output)

    def test_log_message_tolerates_variadic_arguments(self):
        handler = GatewayHandler.__new__(GatewayHandler)
        handler.address_string = lambda: "127.0.0.1"  # type: ignore[method-assign]
        with self.assertLogs("adapters.inbound.http.server", level="INFO"):
            handler.log_message('"%s" %s %s', "GET / HTTP/1.1", 200, 5)
            handler.log_message("code %d, message %s", 404, "missing")

    def test_no_internal_error_is_reported_by_the_request_log(self):
        with self.assertLogs("adapters.inbound.http.server", level="INFO") as captured:
            self.client.post_json({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        joined = "\n".join(captured.output)
        self.assertNotIn("--- Logging error ---", joined)
        self.assertNotIn("Traceback", joined)


class TestStatusMapping(unittest.TestCase):
    def test_domain_errors_map_to_expected_codes(self):
        cases = [
            (ValidationError("x"), 400),
            (AuthorizationError(), 401),
            (AuthenticationError("x"), 401),
            (ModelNotFoundError("m"), 404),
            (UpstreamServiceError("x"), 502),
        ]
        for error, expected in cases:
            self.assertEqual(status_for(error), expected, type(error).__name__)

    def test_upstream_401_is_not_relayed_as_401(self):
        """A stale Google token must not look like a rejected client key."""
        error = UpstreamServiceError("Upstream returned HTTP 401", status_code=502)
        self.assertEqual(status_for(error), 502)


if __name__ == "__main__":
    unittest.main()
