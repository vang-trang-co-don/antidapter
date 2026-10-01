from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator

from core.domain.entities import (
    ChatRequest,
    ChatResponse,
    ModelInfo,
    StreamDelta,
)


class ProtocolTranslatorPort(ABC):
    """Translates between a wire protocol and domain entities.

    Owned by the inbound side so that a second protocol (Anthropic
    /v1/messages, for example) can be added without touching the domain or the
    HTTP transport.
    """

    @abstractmethod
    def parse_chat_request(self, body: bytes) -> ChatRequest:
        """Parse a wire payload into a domain request."""
        raise NotImplementedError

    @abstractmethod
    def serialize_chat_response(self, response: ChatResponse) -> bytes:
        """Render a completed turn as a wire payload."""
        raise NotImplementedError

    @abstractmethod
    def serialize_stream_chunk(
        self,
        chat_id: str,
        model: str,
        delta: StreamDelta,
        created_at: int,
    ) -> bytes:
        """Render one incremental delta as a Server-Sent Event."""
        raise NotImplementedError

    @abstractmethod
    def serialize_stream_done(self) -> bytes:
        """Render the end-of-stream sentinel."""
        raise NotImplementedError

    @abstractmethod
    def serialize_stream_error(self, message: str, error_type: str) -> bytes:
        """Render a failure that occurred after the stream was committed."""
        raise NotImplementedError

    @abstractmethod
    def serialize_models_list(self, models: Iterable[ModelInfo]) -> bytes:
        """Render a model catalog listing."""
        raise NotImplementedError

    @abstractmethod
    def serialize_model(self, model: ModelInfo) -> bytes:
        """Render a single model resource."""
        raise NotImplementedError

    @abstractmethod
    def serialize_model_details(self, models: Iterable[ModelInfo]) -> bytes:
        """Render the catalog with full capability metadata.

        The OpenAI /v1/models shape carries only id/created/owned_by, which is
        not enough for clients that need context windows, output limits and
        modality support. This richer shape is served from a separate route so
        the OpenAI contract stays untouched.
        """
        raise NotImplementedError

    @abstractmethod
    def serialize_error(self, message: str, error_type: str, status: int) -> bytes:
        """Render an error body in this protocol's error shape."""
        raise NotImplementedError

    @abstractmethod
    def wants_stream_usage(self, body: bytes) -> bool:
        """Whether the caller asked for a trailing usage-only chunk."""
        raise NotImplementedError


class ChatUseCase(ABC):
    """Inbound port for executing chat completions."""

    @abstractmethod
    def complete(self, request: ChatRequest) -> ChatResponse:
        """Execute a non-streaming chat turn."""
        raise NotImplementedError

    @abstractmethod
    def complete_stream(self, request: ChatRequest) -> Iterator[StreamDelta]:
        """Execute a streaming chat turn.

        Implementations MUST validate and authenticate eagerly, before
        returning, so that pre-flight failures can still be reported as HTTP
        status codes instead of corrupting an already-committed response.
        """
        raise NotImplementedError


class ModelCatalogUseCase(ABC):
    """Inbound port for querying available models."""

    @abstractmethod
    def list_models(self) -> tuple[ModelInfo, ...]:
        """Return the catalog of currently available models."""
        raise NotImplementedError

    @abstractmethod
    def get_model(self, model_id: str) -> ModelInfo:
        """Return a single model, raising ModelNotFoundError when absent."""
        raise NotImplementedError


class AuthUseCase(ABC):
    """Inbound port for managing authentication state."""

    @abstractmethod
    def ensure_authenticated(self) -> str:
        """Return a valid access token, refreshing or prompting as needed."""
        raise NotImplementedError

    @abstractmethod
    def login_interactive(self) -> None:
        """Trigger the interactive login flow."""
        raise NotImplementedError

    @abstractmethod
    def logout(self) -> None:
        """Discard any cached and persisted credentials."""
        raise NotImplementedError
