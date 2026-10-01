import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.domain.exceptions import ValidationError


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALLS = "tool_calls"
    ERROR = "error"


@dataclass(frozen=True)
class TextPart:
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValidationError("TextPart text must be a string")


@dataclass(frozen=True)
class ImagePart:
    mime_type: str
    base64_data: str

    def __post_init__(self) -> None:
        if not self.mime_type or not self.base64_data:
            raise ValidationError("ImagePart requires both mime_type and base64_data")


@dataclass(frozen=True)
class ToolCallPart:
    """A model-issued function call."""

    call_id: str
    function_name: str
    arguments: str

    def __post_init__(self) -> None:
        if not self.function_name:
            raise ValidationError("ToolCallPart requires function_name")
        if not isinstance(self.arguments, str):
            raise ValidationError("ToolCallPart arguments must be a JSON string")


@dataclass(frozen=True)
class ToolResultPart:
    """The caller's response to a ToolCallPart."""

    call_id: str
    function_name: str
    content: str

    def __post_init__(self) -> None:
        if not self.call_id:
            raise ValidationError("ToolResultPart requires call_id")
        if not isinstance(self.content, str):
            raise ValidationError("ToolResultPart content must be a string")


ContentPart = TextPart | ImagePart | ToolCallPart | ToolResultPart


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    parts: tuple[ContentPart, ...]

    def __post_init__(self) -> None:
        if not self.parts:
            raise ValidationError("ChatMessage requires at least one part")

    @classmethod
    def from_text(cls, role: Role, text: str) -> "ChatMessage":
        return cls(role=role, parts=(TextPart(text=text),))

    @property
    def text_content(self) -> str:
        """Concatenate text parts, keeping block boundaries intact."""
        return "\n".join(part.text for part in self.parts if isinstance(part, TextPart))


@dataclass(frozen=True)
class GenerationParameters:
    temperature: float | None = None
    max_output_tokens: int | None = None
    top_p: float | None = None

    def __post_init__(self) -> None:
        if self.temperature is not None:
            if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)):
                raise ValidationError("temperature must be a number")
            if not 0.0 <= float(self.temperature) <= 2.0:
                raise ValidationError("temperature must be between 0 and 2")
        if self.top_p is not None:
            if isinstance(self.top_p, bool) or not isinstance(self.top_p, (int, float)):
                raise ValidationError("top_p must be a number")
            if not 0.0 <= float(self.top_p) <= 1.0:
                raise ValidationError("top_p must be between 0 and 1")
        if self.max_output_tokens is not None:
            if isinstance(self.max_output_tokens, bool) or not isinstance(
                self.max_output_tokens, int
            ):
                raise ValidationError("max_output_tokens must be an integer")
            if self.max_output_tokens <= 0:
                raise ValidationError("max_output_tokens must be positive")


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str = ""
    parameters: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("ToolDefinition requires a name")


@dataclass(frozen=True)
class ChatRequest:
    model: str
    messages: tuple[ChatMessage, ...]
    stream: bool = False
    parameters: GenerationParameters = field(default_factory=GenerationParameters)
    tools: tuple[ToolDefinition, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValidationError("ChatRequest model must be a non-empty string")
        if not self.messages:
            raise ValidationError("ChatRequest messages must not be empty")


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    thinking_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            thinking_tokens=self.thinking_tokens + other.thinking_tokens,
        )


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    function_name: str
    arguments: str


@dataclass(frozen=True)
class StreamDelta:
    text: str = ""
    reasoning: str = ""
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class ChatResponse:
    id: str
    model: str
    content: str
    created_at: int
    finish_reason: str = "stop"
    usage: TokenUsage | None = None
    reasoning: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class QuotaInfo:
    remaining_fraction: float | None = None
    reset_time: str | None = None


@dataclass(frozen=True)
class ModelInfo:
    id: str
    display_name: str
    provider: str
    context_window: int = 1048576
    max_output_tokens: int = 65536
    supports_thinking: bool = False
    supports_tools: bool = False
    quota_info: QuotaInfo | None = None


@dataclass(frozen=True)
class AuthToken:
    access_token: str
    refresh_token: str | None
    expiry_time: float
    token_type: str = "Bearer"

    def __post_init__(self) -> None:
        if not self.access_token:
            raise ValidationError("AuthToken requires an access_token")

    def is_expired(self, buffer_seconds: float = 60.0) -> bool:
        return time.time() + buffer_seconds >= self.expiry_time

    def to_dict(self) -> dict[str, Any]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expiry_time": self.expiry_time,
            "token_type": self.token_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuthToken":
        if not isinstance(data, dict):
            raise ValidationError("Token payload must be a JSON object")
        access_token = data.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise ValidationError("Token payload is missing access_token")
        try:
            expiry = float(data.get("expiry_time", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValidationError("Token expiry_time must be numeric") from exc
        return cls(
            access_token=access_token,
            refresh_token=data.get("refresh_token"),
            expiry_time=expiry,
            token_type=data.get("token_type", "Bearer"),
        )
