"""HTTP transport for the gateway.

This module is deliberately thin: routing, authentication, body limits and
status-code mapping only. All protocol knowledge lives in a
ProtocolTranslatorPort implementation and all orchestration in the use cases,
so adding a second wire protocol touches neither this file nor the domain.
"""

import hmac
import logging
import time
import urllib.parse
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from config import ServerConfig
from core.domain.entities import ChatRequest, StreamDelta
from core.domain.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    DomainException,
    ModelNotFoundError,
    TokenExpiredError,
    UpstreamServiceError,
    ValidationError,
)
from core.ports.inbound import (
    ChatUseCase,
    ModelCatalogUseCase,
    ProtocolTranslatorPort,
)
from core.services.chat_service import new_chat_id

logger = logging.getLogger(__name__)

# Maps domain failures onto HTTP status codes. Upstream statuses are NOT passed
# through verbatim: a 401 from Google means *our* token is stale, and relaying
# it would make clients believe their own API key was rejected.
STATUS_BY_ERROR: tuple[tuple[type, int], ...] = (
    (ValidationError, 400),
    (AuthorizationError, 401),
    (AuthenticationError, 401),
    (TokenExpiredError, 401),
    (ModelNotFoundError, 404),
    (ConfigurationError, 500),
    (UpstreamServiceError, 502),
)


def status_for(error: DomainException) -> int:
    for error_type, status in STATUS_BY_ERROR:
        if isinstance(error, error_type):
            return status
    return 500


class Route:
    __slots__ = ("handler", "method", "pattern", "requires_auth")

    def __init__(self, method: str, pattern: str, handler: str, requires_auth: bool = True):
        self.method = method
        self.pattern = pattern
        self.handler = handler
        self.requires_auth = requires_auth

    def match(self, path: str) -> dict | None:
        expected = self.pattern.strip("/").split("/")
        actual = path.strip("/").split("/")
        if len(expected) != len(actual):
            return None
        params = {}
        for want, got in zip(expected, actual, strict=True):
            if want.startswith("{") and want.endswith("}"):
                params[want[1:-1]] = urllib.parse.unquote(got)
            elif want != got:
                return None
        return params


ROUTES = (
    Route("GET", "/health", "health", requires_auth=False),
    Route("GET", "/v1/models", "list_models"),
    Route("GET", "/v1/gateway/models", "model_details"),
    Route("GET", "/v1/models/{model_id}", "get_model"),
    Route("POST", "/v1/chat/completions", "chat"),
    # Unversioned aliases kept for existing harnesses.
    Route("GET", "/models", "list_models"),
    Route("POST", "/chat/completions", "chat"),
)


class GatewayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "antidapter"

    translator: ProtocolTranslatorPort
    chat_use_case: ChatUseCase
    catalog_use_case: ModelCatalogUseCase
    config: ServerConfig
    on_ready: Callable[[], None]

    def do_GET(self) -> None:
        route, params = self._resolve("GET")
        if route is None:
            return
        if route.requires_auth and not self._authorize():
            return
        if route.handler == "health":
            self._send_json(b'{"status":"ok"}')
        elif route.handler == "list_models":
            self._guard(
                lambda: self._send_json(
                    self.translator.serialize_models_list(self.catalog_use_case.list_models())
                )
            )
        elif route.handler == "model_details":
            self._guard(
                lambda: self._send_json(
                    self.translator.serialize_model_details(self.catalog_use_case.list_models())
                )
            )
        elif route.handler == "get_model":
            self._guard(
                lambda: self._send_json(
                    self.translator.serialize_model(
                        self.catalog_use_case.get_model(params["model_id"])
                    )
                )
            )

    def do_POST(self) -> None:
        route, _params = self._resolve("POST")
        if route is None:
            return
        if route.requires_auth and not self._authorize():
            return
        if route.handler != "chat":
            self._send_status(404, "Not Found", "not_found")
            return

        try:
            body = self._read_body()
        except DomainException as exc:
            self._send_error(exc)
            return

        try:
            request = self.translator.parse_chat_request(body)
        except DomainException as exc:
            self._send_error(exc)
            return

        if request.stream:
            self._stream_response(request, self.translator.wants_stream_usage(body))
        else:
            self._guard(
                lambda: self._send_json(
                    self.translator.serialize_chat_response(self.chat_use_case.complete(request))
                )
            )

    # -- routing -----------------------------------------------------------

    def _resolve(self, method: str) -> tuple[Route | None, dict[str, str]]:
        path = urllib.parse.urlparse(self.path).path
        allowed = False
        for route in ROUTES:
            params = route.match(path)
            if params is None:
                continue
            if route.method != method:
                allowed = True
                continue
            return route, params
        if allowed:
            self._send_status(405, "Method Not Allowed", "method_not_allowed")
        else:
            self._send_status(404, "Not Found", "not_found")
        return None, {}

    def _authorize(self) -> bool:
        expected = self.config.api_key
        if not expected:
            return True
        header = self.headers.get("Authorization", "")
        scheme, _, presented = header.partition(" ")
        if scheme.lower() != "bearer" or not presented:
            self._send_error(AuthorizationError("Missing bearer token"))
            return False
        if not hmac.compare_digest(presented.strip(), expected):
            self._send_error(AuthorizationError())
            return False
        return True

    def _read_body(self) -> bytes:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValidationError("Content-Length header is required")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValidationError("Content-Length must be an integer") from exc
        if length < 0:
            raise ValidationError("Content-Length must not be negative")
        if length > self.config.max_request_bytes:
            raise ValidationError(
                f"Request body exceeds the {self.config.max_request_bytes} byte limit"
            )
        if length == 0:
            raise ValidationError("Request body is empty")
        return self.rfile.read(length)

    # -- responses ---------------------------------------------------------

    def _stream_response(self, request: ChatRequest, include_usage: bool) -> None:
        # Resolve the iterator before committing a response: complete_stream
        # authenticates eagerly, so pre-flight failures still map to real
        # status codes instead of corrupting an already-sent 200.
        try:
            delta_iter = self.chat_use_case.complete_stream(request)
        except DomainException as exc:
            self._send_error(exc)
            return
        except Exception:
            logger.exception("Unexpected pre-stream failure")
            self._send_status(500, "Internal Server Error", "internal_error")
            return

        chat_id = new_chat_id()
        created_at = int(time.time())
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        # Framed by connection close: no Content-Length and no chunked encoding.
        # Claiming keep-alive would make the server await another request while
        # the client awaits EOF.
        self.send_header("Connection", "close")
        self._set_cors_headers()
        self.end_headers()
        self.close_connection = True

        last_usage = None
        try:
            for delta in delta_iter:
                if delta.usage is not None:
                    last_usage = delta.usage
                self._write(
                    self.translator.serialize_stream_chunk(
                        chat_id=chat_id, model=request.model, delta=delta, created_at=created_at
                    )
                )
        except (BrokenPipeError, ConnectionResetError):
            logger.info("Client disconnected mid-stream (%s)", chat_id)
            return
        except DomainException as exc:
            # The 200 is already on the wire; report in-band.
            logger.error("Stream %s failed after commit: %s", chat_id, exc.message)
            self._write(self.translator.serialize_stream_error(exc.message, exc.code))
        except Exception as exc:
            logger.exception("Unexpected error during stream %s", chat_id)
            self._write(self.translator.serialize_stream_error(str(exc), "internal_error"))

        if include_usage and last_usage is not None:
            self._write(
                self.translator.serialize_stream_chunk(
                    chat_id=chat_id,
                    model=request.model,
                    delta=StreamDelta(usage=last_usage),
                    created_at=created_at,
                )
            )
        self._write(self.translator.serialize_stream_done())

    def _write(self, payload: bytes) -> None:
        try:
            self.wfile.write(payload)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            raise

    def _guard(self, action: Callable[[], None]) -> None:
        try:
            action()
        except DomainException as exc:
            self._send_error(exc)
        except (BrokenPipeError, ConnectionResetError):
            # The client hung up before we finished writing. That is a normal
            # event (a cancelled fetch, a timed-out probe), not a server fault,
            # and there is nobody left to send a 500 to.
            logger.info("Client disconnected before the response was written")
        except Exception:
            logger.exception("Unexpected server error")
            self._send_status(500, "Internal Server Error", "internal_error")

    def _send_json(self, body: bytes, status: int = 200) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            logger.info("Client disconnected while sending a %d response", status)

    def _send_error(self, exc: DomainException) -> None:
        status = status_for(exc)
        if status >= 500:
            logger.error("%s -> HTTP %d: %s", type(exc).__name__, status, exc.message)
        self._send_json(self.translator.serialize_error(exc.message, exc.code, status), status)

    def _send_status(self, status: int, phrase: str, error_type: str) -> None:
        self._send_json(self.translator.serialize_error(phrase, error_type, status), status)

    def _set_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self._set_cors_headers()
        self.end_headers()

    def log_message(self, fmt: str, *args: object) -> None:
        # stdlib supplies a format string plus a variable number of arguments,
        # so interpolate first and then log the result as a single value.
        logger.info("%s %s", self.address_string(), fmt % args)


def create_http_server(
    host: str,
    port: int,
    translator: ProtocolTranslatorPort,
    chat_use_case: ChatUseCase,
    catalog_use_case: ModelCatalogUseCase,
    config: ServerConfig | None = None,
    on_ready: Callable[[], None] | None = None,
) -> ThreadingHTTPServer:
    """Build the HTTP server with all dependencies injected.

    BaseHTTPRequestHandler instantiates handlers without arguments, so the
    per-server dependencies are bound onto a per-server subclass here rather
    than being passed through a constructor.
    """
    handler_cls = type(
        "ConfiguredGatewayHandler",
        (GatewayHandler,),
        {
            "translator": translator,
            "chat_use_case": chat_use_case,
            "catalog_use_case": catalog_use_case,
            "config": config or ServerConfig(),
            "on_ready": on_ready or (lambda: None),
        },
    )
    return ThreadingHTTPServer((host, port), handler_cls)
