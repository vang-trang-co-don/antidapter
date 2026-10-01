import contextlib
import itertools
import json
import logging
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from typing import Any, TypeVar, cast

from config import UpstreamConfig
from core.domain.entities import (
    ChatRequest,
    ImagePart,
    ModelInfo,
    QuotaInfo,
    Role,
    StreamDelta,
    TextPart,
    TokenUsage,
    ToolCall,
    ToolCallPart,
    ToolDefinition,
    ToolResultPart,
)
from core.domain.exceptions import (
    DomainException,
    UpstreamRejectedError,
    UpstreamServiceError,
    UpstreamTimeoutError,
)
from core.ports.outbound import UpstreamModelPort

logger = logging.getLogger(__name__)

DONE_SENTINEL = "[DONE]"

FINISH_REASON_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
    "MALFORMED_FUNCTION_CALL": "error",
}

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

_T = TypeVar("_T")
_END_OF_STREAM = object()


class GoogleCloudCodeAdapter(UpstreamModelPort):
    """Translates domain requests into Google Cloud Code payloads and back.

    HTTP transport, retry policy, SSE framing and vocabulary mapping all live
    here so the domain never learns that Google exists.
    """

    def __init__(
        self,
        config: UpstreamConfig,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self._config = config
        self._sleep = sleep
        self._monotonic = monotonic
        # (fetched_at, models); None until the first successful read.
        self._catalog_cache: tuple[float, tuple[ModelInfo, ...]] | None = None

    def stream_generate(
        self,
        token: str,
        request: ChatRequest,
    ) -> Iterator[StreamDelta]:
        url = f"{self._config.base_url}{self._config.stream_generate_path}"
        payload = self._map_to_upstream_payload(request)
        body = json.dumps(payload).encode("utf-8")

        def attempt() -> Iterator[StreamDelta]:
            import os as _os

            if _os.environ.get("ANTIDAPTER_DEBUG_TOOLS"):
                with open("/tmp/opencode/tools_sent.json", "w") as _f:
                    json.dump(payload.get("request", {}).get("tools"), _f, indent=1)
            req = urllib.request.Request(
                url,
                data=body,
                headers=self._headers(token),
                method="POST",
            )
            with self._open(req) as resp:
                for line in resp:
                    for raw in self._iter_sse_payloads(line.decode("utf-8", errors="replace")):
                        if raw == DONE_SENTINEL:
                            return
                        try:
                            chunk = json.loads(raw)
                        except json.JSONDecodeError:
                            logger.warning("Discarding malformed SSE payload: %.200s", raw)
                            continue
                        delta = self._parse_chunk(chunk)
                        if delta is not None:
                            yield delta

        return self._stream_with_retry(attempt)

    def fetch_models(self, token: str) -> tuple[ModelInfo, ...]:
        """Return the catalog, served from a short-lived cache when warm.

        The catalog changes rarely, but every read costs a round trip to Google
        and a live request can take seconds. Clients (the pi extension, model
        pickers) poll this on startup, so a brief cache keeps those reads cheap
        and fast without pinning a stale list for long.
        """
        now = self._monotonic()
        cached = self._catalog_cache
        if cached is not None and now - cached[0] < self._config.catalog_ttl:
            return cached[1]

        url = f"{self._config.base_url}{self._config.fetch_models_path}"

        def attempt() -> tuple[ModelInfo, ...]:
            req = urllib.request.Request(
                url, data=b"{}", headers=self._headers(token), method="POST"
            )
            with self._open(req) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            return self._map_models(raw)

        models = self._with_retry(attempt)
        self._catalog_cache = (self._monotonic(), models)
        return models

    def _headers(self, token: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": self._config.user_agent,
            "Accept": "text/event-stream",
        }

    @contextlib.contextmanager
    def _open(self, request: urllib.request.Request) -> Iterator[Any]:
        """Perform the request, translating transport failures to domain errors.

        Keeping this in one place guarantees that every exit path out of the
        adapter raises a domain exception, so callers never have to know that
        urllib is involved and the retry policy sees a uniform error type.
        """
        try:
            with urllib.request.urlopen(request, timeout=self._config.request_timeout) as response:
                yield response
        except urllib.error.HTTPError as exc:
            body = ""
            with contextlib.suppress(Exception):
                body = exc.read().decode("utf-8", errors="replace")
            with contextlib.suppress(Exception):
                exc.close()
            detail = _summarize(body)
            logger.error("Upstream returned HTTP %s: %s", exc.code, detail)
            if 400 <= exc.code < 500 and exc.code not in RETRYABLE_STATUS:
                # Most 4xx mean the caller's payload is wrong, which is not a
                # gateway fault. Surface them as-is so the client gets an
                # actionable message instead of a 502 it will dutifully retry.
                # The retryable 4xx (408/425/429) are rate limits and
                # timeouts, not client mistakes, so they keep the retry path.
                raise UpstreamRejectedError(
                    _rejection_message(exc.code, detail),
                    upstream_status=exc.code,
                    details=detail,
                ) from exc
            raise UpstreamServiceError(
                f"Upstream returned HTTP {exc.code}",
                status_code=502,
                details=detail,
                retryable=exc.code in RETRYABLE_STATUS,
            ) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise UpstreamTimeoutError(
                    f"Upstream did not respond within {self._config.request_timeout:.0f}s",
                    timeout=self._config.request_timeout,
                ) from exc
            raise UpstreamServiceError(
                f"Could not reach the upstream: {exc.reason}",
                status_code=504,
                retryable=True,
            ) from exc
        except TimeoutError as exc:
            raise UpstreamTimeoutError(
                f"Upstream did not respond within {self._config.request_timeout:.0f}s",
                timeout=self._config.request_timeout,
            ) from exc
        except OSError as exc:
            raise UpstreamServiceError(
                f"Upstream connection failed: {exc}", status_code=504, retryable=True
            ) from exc
        except json.JSONDecodeError as exc:
            raise UpstreamServiceError(
                "Upstream returned a malformed JSON body", status_code=502
            ) from exc

    def _with_retry(self, operation: Callable[[], _T]) -> _T:
        """Run an eager operation, retrying transient failures with backoff."""
        for attempt_index in range(self._attempts):
            try:
                return operation()
            except DomainException as exc:
                if not getattr(exc, "retryable", False) or attempt_index == self._attempts - 1:
                    raise
                logger.warning(
                    "Upstream call failed (%s); retry %d/%d",
                    exc.message,
                    attempt_index + 1,
                    self._attempts,
                )
                self._backoff(attempt_index, exc.message)
        raise UpstreamServiceError("Upstream request failed", status_code=502)

    def _stream_with_retry(
        self, factory: Callable[[], Iterator[StreamDelta]]
    ) -> Iterator[StreamDelta]:
        """Retry a streaming call only until the first chunk is committed.

        ``factory()`` returns a generator, so no I/O happens until the first
        ``next()``. Priming it inside the retry loop is what lets connection
        and auth failures be retried; once a delta has been handed to the
        caller the stream cannot be safely replayed.
        """
        for attempt_index in range(self._attempts):
            iterator = factory()
            try:
                first = next(iterator, _END_OF_STREAM)
            except DomainException as exc:
                if not getattr(exc, "retryable", False) or attempt_index == self._attempts - 1:
                    raise
                logger.warning(
                    "Upstream stream failed to open (%s); retry %d/%d",
                    exc.message,
                    attempt_index + 1,
                    self._attempts,
                )
                self._backoff(attempt_index, exc.message)
                continue
            if first is _END_OF_STREAM:
                return iter(())
            return itertools.chain((cast(StreamDelta, first),), iterator)
        raise UpstreamServiceError("Upstream stream failed to open", status_code=502)

    @property
    def _attempts(self) -> int:
        return self._config.max_retries + 1

    def _backoff(self, attempt_index: int, reason: str) -> None:
        delay = min(
            self._config.retry_base_delay * (2**attempt_index),
            self._config.retry_max_delay,
        )
        logger.info("Backing off %.2fs after %s", delay, reason)
        self._sleep(delay)

    @staticmethod
    def _iter_sse_payloads(line: str) -> Iterator[str]:
        """Yield the data payload of a single SSE line.

        Handles the `data:` prefix, tolerates an optional space, and treats
        `[DONE]` and comment/field lines as absent so callers never have to
        special-case them.
        """
        stripped = line.strip()
        if not stripped or stripped.startswith(":"):
            return
        if not stripped.startswith("data:"):
            return
        yield stripped[5:].strip()

    def _map_models(self, raw: dict[str, Any]) -> tuple[ModelInfo, ...]:
        models: list[ModelInfo] = []
        for model_id, details in (raw.get("models") or {}).items():
            raw_quota = details.get("quotaInfo") or {}
            quota = (
                QuotaInfo(
                    remaining_fraction=raw_quota.get("remainingFraction"),
                    reset_time=raw_quota.get("resetTime"),
                )
                if raw_quota
                else None
            )
            models.append(
                ModelInfo(
                    id=model_id,
                    display_name=details.get("displayName") or model_id,
                    provider=details.get("modelProvider", "google"),
                    context_window=details.get("inputTokenLimit", 1048576),
                    max_output_tokens=details.get("maxOutputTokens", 65536),
                    supports_thinking=bool(details.get("supportsThinking", False)),
                    supports_tools=bool(details.get("supportsTools", True)),
                    quota_info=quota,
                )
            )
        return tuple(models)

    def _map_to_upstream_payload(self, request: ChatRequest) -> dict[str, Any]:
        contents: list[dict[str, Any]] = []
        system_parts: list[dict[str, Any]] = []

        for msg in request.messages:
            parts = self._map_parts(msg.parts)
            if not parts:
                continue

            if msg.role in (Role.SYSTEM,):
                system_parts.extend(parts)
            elif msg.role is Role.USER:
                contents.append({"role": "user", "parts": parts})
            elif msg.role is Role.ASSISTANT:
                contents.append({"role": "model", "parts": parts})
            elif msg.role is Role.TOOL:
                contents.append({"role": "user", "parts": parts})

        gemini_request: dict[str, Any] = {"contents": contents}
        if system_parts:
            gemini_request["systemInstruction"] = {"parts": system_parts}

        gen_config: dict[str, Any] = {}
        if request.parameters.temperature is not None:
            gen_config["temperature"] = request.parameters.temperature
        if request.parameters.max_output_tokens is not None:
            gen_config["maxOutputTokens"] = request.parameters.max_output_tokens
        if request.parameters.top_p is not None:
            gen_config["topP"] = request.parameters.top_p
        if gen_config:
            gemini_request["generationConfig"] = gen_config

        tools = self._map_tools(request.tools)
        if tools:
            gemini_request["tools"] = tools

        return {
            "project": self._config.project_id,
            "model": request.model,
            "request": gemini_request,
        }

    @staticmethod
    def _map_tools(tools: Iterable[ToolDefinition]) -> list[dict[str, Any]]:
        declarations = []
        for tool in tools:
            declaration: dict[str, Any] = {"name": tool.name}
            if tool.description:
                declaration["description"] = tool.description
            if tool.parameters:
                declaration["parameters"] = tool.parameters
            declarations.append(declaration)
        return [{"functionDeclarations": declarations}] if declarations else []

    @staticmethod
    def _map_parts(parts: Iterable[Any]) -> list[dict[str, Any]]:
        mapped: list[dict[str, Any]] = []
        for part in parts:
            if isinstance(part, TextPart):
                mapped.append({"text": part.text})
            elif isinstance(part, ImagePart):
                mapped.append(
                    {"inlineData": {"mimeType": part.mime_type, "data": part.base64_data}}
                )
            elif isinstance(part, ToolCallPart):
                mapped.append(
                    {
                        "functionCall": {
                            "name": part.function_name,
                            "args": _safe_json_object(part.arguments),
                        }
                    }
                )
            elif isinstance(part, ToolResultPart):
                mapped.append(
                    {
                        "functionResponse": {
                            "name": part.function_name,
                            "response": {"result": part.content},
                        }
                    }
                )
        return mapped

    def _parse_chunk(self, chunk: dict[str, Any]) -> StreamDelta | None:
        response = chunk.get("response") or {}
        candidates = response.get("candidates") or []

        text_pieces: list[str] = []
        reasoning_pieces: list[str] = []
        tool_calls: list[ToolCall] = []
        finish_reason: str | None = None

        for candidate in candidates:
            for part in (candidate.get("content") or {}).get("parts") or []:
                if part.get("text"):
                    if part.get("thought"):
                        reasoning_pieces.append(part["text"])
                    else:
                        text_pieces.append(part["text"])
                call = part.get("functionCall")
                if call:
                    tool_calls.append(
                        ToolCall(
                            call_id=call.get("id") or f"call_{len(tool_calls)}",
                            function_name=call.get("name", ""),
                            arguments=json.dumps(call.get("args") or {}),
                        )
                    )
            if candidate.get("finishReason"):
                finish_reason = FINISH_REASON_MAP.get(
                    str(candidate["finishReason"]).upper(),
                    str(candidate["finishReason"]).lower(),
                )

        usage = None
        raw_usage = response.get("usageMetadata")
        if raw_usage:
            usage = TokenUsage(
                prompt_tokens=raw_usage.get("promptTokenCount", 0) or 0,
                completion_tokens=raw_usage.get("candidatesTokenCount", 0) or 0,
                total_tokens=raw_usage.get("totalTokenCount", 0) or 0,
                thinking_tokens=raw_usage.get("thoughtsTokenCount", 0) or 0,
            )

        has_payload = bool(text_pieces or reasoning_pieces or tool_calls or finish_reason or usage)
        if not has_payload:
            return None
        return StreamDelta(
            text="".join(text_pieces),
            reasoning="".join(reasoning_pieces),
            finish_reason=finish_reason,
            usage=usage,
            tool_calls=tuple(tool_calls),
        )


def _summarize(body: str, limit: int = 300) -> str:
    flattened = " ".join(body.split())
    return flattened[:limit] + ("..." if len(flattened) > limit else "")


def _rejection_message(status: int, detail: str) -> str:
    """Pull the upstream's own reason out of its envelope, when present."""
    reason = detail
    try:
        parsed = json.loads(detail)
        message = parsed.get("error", {}).get("message")
        if isinstance(message, str):
            reason = message
            # The upstream double-encodes the reason as a JSON string.
            try:
                inner = json.loads(message)
                if isinstance(inner, dict) and isinstance(inner.get("error"), dict):
                    reason = inner["error"].get("message") or message
            except json.JSONDecodeError:
                pass
    except (json.JSONDecodeError, AttributeError):
        pass
    return f"Upstream rejected the request (HTTP {status}): {_summarize(reason, 400)}"


def _safe_json_object(arguments: str) -> dict[str, Any]:
    """Parse tool-call arguments, tolerating empty or malformed strings."""
    if not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        logger.warning("Tool call arguments were not valid JSON; sending empty object")
        return {}
    return parsed if isinstance(parsed, dict) else {"value": parsed}
