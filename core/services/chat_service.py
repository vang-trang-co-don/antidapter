import time
import uuid
from collections.abc import Iterator

from core.domain.entities import (
    ChatRequest,
    ChatResponse,
    StreamDelta,
    TokenUsage,
    ToolCall,
)
from core.ports.inbound import AuthUseCase, ChatUseCase
from core.ports.outbound import UpstreamModelPort


class ChatService(ChatUseCase):
    """Orchestrates authentication and delegates generation to the upstream port.

    Request validation is intentionally *not* repeated here: ChatRequest is a
    frozen entity that enforces its own invariants in __post_init__, so a
    ChatRequest instance reaching this service is valid by construction.
    """

    def __init__(
        self,
        auth_use_case: AuthUseCase,
        upstream_model: UpstreamModelPort,
    ):
        self._auth = auth_use_case
        self._upstream = upstream_model

    def complete(self, request: ChatRequest) -> ChatResponse:
        token = self._auth.ensure_authenticated()

        deltas = list(self._upstream.stream_generate(token, request))

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        finish_reason = "stop"
        usage: TokenUsage | None = None

        for delta in deltas:
            if delta.text:
                text_parts.append(delta.text)
            if delta.reasoning:
                reasoning_parts.append(delta.reasoning)
            tool_calls.extend(delta.tool_calls)
            if delta.finish_reason:
                finish_reason = delta.finish_reason
            # Upstream reports cumulative usage on its final chunk, so the last
            # non-null value wins rather than being summed.
            if delta.usage is not None:
                usage = delta.usage

        return ChatResponse(
            id=new_chat_id(),
            model=request.model,
            content="".join(text_parts),
            created_at=int(time.time()),
            finish_reason="tool_calls" if tool_calls and finish_reason == "stop" else finish_reason,
            usage=usage,
            reasoning="".join(reasoning_parts),
            tool_calls=tuple(tool_calls),
        )

    def complete_stream(self, request: ChatRequest) -> Iterator[StreamDelta]:
        # Intentionally NOT a generator function. Validation and authentication
        # must run eagerly, at call time, so failures surface to the caller
        # before any inbound adapter commits an HTTP response. Deferring them
        # into the first next() would raise mid-stream, after the 200 has
        # already been written.
        token = self._auth.ensure_authenticated()
        return self._upstream.stream_generate(token, request)


def new_chat_id() -> str:
    """Single source of truth for completion identifiers."""
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"
