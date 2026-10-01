import json
import logging
import time
from collections.abc import Iterable
from typing import Any

from config import ProtocolConfig
from core.domain.entities import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ContentPart,
    GenerationParameters,
    ImagePart,
    ModelInfo,
    Role,
    StreamDelta,
    TextPart,
    TokenUsage,
    ToolCallPart,
    ToolDefinition,
    ToolResultPart,
)
from core.domain.exceptions import ValidationError
from core.ports.inbound import ProtocolTranslatorPort

logger = logging.getLogger(__name__)

# Roles accepted on the wire. Anything else is rejected rather than coerced:
# silently relabelling a `tool` message as `user` corrupts agent transcripts.
ROLE_MAP = {
    "system": Role.SYSTEM,
    "developer": Role.SYSTEM,
    "user": Role.USER,
    "assistant": Role.ASSISTANT,
    "tool": Role.TOOL,
    "function": Role.TOOL,
}


class OpenAIProtocolTranslator(ProtocolTranslatorPort):
    """OpenAI Chat Completions <-> domain entity translation."""

    def __init__(self, config: ProtocolConfig | None = None):
        self._config = config or ProtocolConfig()

    def parse_chat_request(self, body: bytes) -> ChatRequest:
        data = self._load_object(body)

        raw_messages = data.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise ValidationError("'messages' must be a non-empty array")

        messages = [
            message
            for message in (
                self._parse_message(item, index) for index, item in enumerate(raw_messages)
            )
            if message is not None
        ]
        if not messages:
            raise ValidationError("At least one message with usable content is required")

        stream = data.get("stream", False)
        if not isinstance(stream, bool):
            raise ValidationError("'stream' must be a boolean")

        return ChatRequest(
            model=self._parse_model(data.get("model")),
            messages=tuple(messages),
            stream=stream,
            parameters=self._parse_parameters(data),
            tools=tuple(self._parse_tools(data.get("tools"))),
        )

    def serialize_chat_response(self, response: ChatResponse) -> bytes:
        message: dict[str, Any] = {"role": "assistant", "content": response.content or None}
        if response.reasoning:
            message["reasoning_content"] = response.reasoning
        if response.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {"name": call.function_name, "arguments": call.arguments},
                }
                for call in response.tool_calls
            ]

        payload: dict[str, Any] = {
            "id": response.id,
            "object": "chat.completion",
            "created": response.created_at,
            "model": response.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "logprobs": None,
                    "finish_reason": response.finish_reason,
                }
            ],
        }
        if response.usage is not None:
            payload["usage"] = _usage_payload(response.usage)
        return _encode(payload)

    def serialize_stream_chunk(
        self,
        chat_id: str,
        model: str,
        delta: StreamDelta,
        created_at: int,
    ) -> bytes:
        body: dict[str, Any] = {}
        if delta.text:
            body["content"] = delta.text
        if delta.reasoning:
            body["reasoning_content"] = delta.reasoning
        if delta.tool_calls:
            body["tool_calls"] = [
                {
                    "index": index,
                    "id": call.call_id,
                    "type": "function",
                    "function": {"name": call.function_name, "arguments": call.arguments},
                }
                for index, call in enumerate(delta.tool_calls)
            ]

        chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created_at,
            "model": model,
            # A usage-only chunk (stream_options.include_usage) carries an
            # empty choices array, matching OpenAI's wire format.
            "choices": (
                []
                if not (delta.text or delta.reasoning or delta.tool_calls or delta.finish_reason)
                else [
                    {
                        "index": 0,
                        "delta": body,
                        "logprobs": None,
                        "finish_reason": delta.finish_reason,
                    }
                ]
            ),
        }
        if delta.usage is not None:
            chunk["usage"] = _usage_payload(delta.usage)
        return _sse(chunk)

    def serialize_stream_done(self) -> bytes:
        return b"data: [DONE]\n\n"

    def serialize_stream_error(self, message: str, error_type: str) -> bytes:
        return _sse({"error": {"message": message, "type": error_type, "code": error_type}})

    def serialize_models_list(self, models: Iterable[ModelInfo]) -> bytes:
        created = int(time.time())
        return _encode(
            {
                "object": "list",
                "data": [
                    {
                        "id": model.id,
                        "object": "model",
                        "created": created,
                        "owned_by": model.provider,
                    }
                    for model in models
                ],
            }
        )

    def serialize_model(self, model: ModelInfo) -> bytes:
        return _encode(
            {
                "id": model.id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": model.provider,
            }
        )

    def serialize_model_details(self, models: Iterable[ModelInfo]) -> bytes:
        """Render the catalog with capability metadata for rich clients."""
        return _encode(
            {
                "object": "list",
                "models": [
                    {
                        "id": model.id,
                        "name": model.display_name or model.id,
                        "provider": model.provider,
                        "input": ["text", "image"],
                        "contextWindow": model.context_window,
                        "maxTokens": model.max_output_tokens,
                        "reasoning": model.supports_thinking,
                        "supportsTools": model.supports_tools,
                        "quota": (
                            {
                                "remainingFraction": model.quota_info.remaining_fraction,
                                "resetTime": model.quota_info.reset_time,
                            }
                            if model.quota_info
                            else None
                        ),
                    }
                    for model in models
                ],
            }
        )

    def serialize_error(self, message: str, error_type: str, status: int) -> bytes:
        return _encode(
            {"error": {"message": message, "type": error_type, "code": status, "param": None}}
        )

    def wants_stream_usage(self, body: bytes) -> bool:
        """Honour OpenAI's stream_options.include_usage."""
        try:
            data = self._load_object(body)
        except ValidationError:
            return False
        options = data.get("stream_options")
        return isinstance(options, dict) and bool(options.get("include_usage"))

    def _parse_model(self, raw: Any) -> str:
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return self._config.default_model
        if not isinstance(raw, str):
            raise ValidationError("'model' must be a string")
        return raw

    def _load_object(self, body: bytes) -> dict[str, Any]:
        try:
            data = json.loads(body.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ValidationError("Request body must be UTF-8 encoded") from exc
        except json.JSONDecodeError as exc:
            raise ValidationError(f"Request body is not valid JSON: {exc.msg}") from exc
        if not isinstance(data, dict):
            raise ValidationError("Request body must be a JSON object")
        return data

    def _parse_message(self, raw: Any, index: int) -> ChatMessage | None:
        if not isinstance(raw, dict):
            raise ValidationError(f"messages[{index}] must be an object")

        raw_role = raw.get("role")
        if not isinstance(raw_role, str) or raw_role not in ROLE_MAP:
            raise ValidationError(
                f"messages[{index}].role {raw_role!r} is not supported; "
                f"expected one of {sorted(ROLE_MAP)}"
            )
        role = ROLE_MAP[raw_role]

        parts: list[ContentPart] = []
        if role is Role.TOOL:
            # For a tool turn the body *is* the result; it must not also be
            # parsed as ordinary text or the transcript gains a phantom turn.
            parts.append(self._parse_tool_result(raw, index))
        else:
            parts.extend(self._parse_content(raw.get("content"), index))
            parts.extend(self._parse_tool_calls(raw, index))
        if not parts:
            return None
        return ChatMessage(role=role, parts=tuple(parts))

    def _parse_content(self, raw: Any, index: int) -> Iterable[ContentPart]:
        if raw is None:
            return
        if isinstance(raw, str):
            if raw:
                yield TextPart(text=raw)
            return
        if not isinstance(raw, list):
            raise ValidationError(f"messages[{index}].content must be a string or an array")

        for position, block in enumerate(raw):
            if isinstance(block, str):
                if block:
                    yield TextPart(text=block)
                continue
            if not isinstance(block, dict):
                raise ValidationError(f"messages[{index}].content[{position}] must be an object")
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text", "")
                if not isinstance(text, str):
                    raise ValidationError(
                        f"messages[{index}].content[{position}].text must be a string"
                    )
                if text:
                    yield TextPart(text=text)
            elif block_type == "image_url":
                part = self._parse_image(block, index, position)
                if part is not None:
                    yield part
            elif block_type in ("input_audio", "file", "refusal"):
                raise ValidationError(
                    f"messages[{index}].content[{position}].type {block_type!r} is not supported"
                )

    def _parse_image(self, block: dict[str, Any], index: int, position: int) -> ImagePart | None:
        raw_url = (block.get("image_url") or {}).get("url")
        if not isinstance(raw_url, str) or not raw_url:
            raise ValidationError(
                f"messages[{index}].content[{position}].image_url.url is required"
            )
        if not raw_url.startswith("data:"):
            # Fetching remote URLs would turn the gateway into an SSRF proxy.
            raise ValidationError(
                f"messages[{index}].content[{position}] only supports base64 data URLs, "
                "not remote image URLs"
            )
        header, separator, payload = raw_url.partition(";base64,")
        if not separator or not payload:
            raise ValidationError(
                f"messages[{index}].content[{position}] is not a valid base64 data URL"
            )
        return ImagePart(mime_type=header[len("data:") :], base64_data=payload)

    def _parse_tool_result(self, raw: dict[str, Any], index: int) -> ToolResultPart:
        call_id = raw.get("tool_call_id") or raw.get("name")
        if not isinstance(call_id, str) or not call_id:
            raise ValidationError(f"messages[{index}] with role 'tool' requires a 'tool_call_id'")
        content = raw.get("content")
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = json.dumps(content)
        return ToolResultPart(
            call_id=call_id,
            function_name=raw.get("name") or call_id,
            content=content,
        )

    def _parse_tool_calls(self, raw: dict[str, Any], index: int) -> Iterable[ContentPart]:
        for call in raw.get("tool_calls") or []:
            if not isinstance(call, dict):
                raise ValidationError(f"messages[{index}].tool_calls entries must be objects")
            function = call.get("function") or {}
            arguments = function.get("arguments", "{}")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)
            yield ToolCallPart(
                call_id=str(call.get("id") or f"call_{index}"),
                function_name=str(function.get("name") or ""),
                arguments=arguments,
            )

    def _parse_parameters(self, data: dict[str, Any]) -> GenerationParameters:
        return GenerationParameters(
            temperature=data.get("temperature"),
            max_output_tokens=data.get("max_completion_tokens") or data.get("max_tokens"),
            top_p=data.get("top_p"),
        )

    def _parse_tools(self, raw: Any) -> Iterable[ToolDefinition]:
        if raw is None:
            return
        if not isinstance(raw, list):
            raise ValidationError("'tools' must be an array")
        for position, entry in enumerate(raw):
            if not isinstance(entry, dict) or entry.get("type") != "function":
                raise ValidationError(f"tools[{position}] must be a function definition")
            function = entry.get("function") or {}
            parameters = function.get("parameters")
            yield ToolDefinition(
                name=str(function.get("name") or ""),
                description=str(function.get("description") or ""),
                parameters=parameters if isinstance(parameters, dict) else None,
            )


def _usage_payload(usage: TokenUsage) -> dict[str, int]:
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


def _encode(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload).encode("utf-8")


def _sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()
