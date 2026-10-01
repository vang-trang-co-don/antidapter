from abc import ABC, abstractmethod
from collections.abc import Iterator

from core.domain.entities import AuthToken, ChatRequest, ModelInfo, StreamDelta


class UpstreamModelPort(ABC):
    """Outbound port for communicating with an upstream model API."""

    @abstractmethod
    def stream_generate(
        self,
        token: str,
        request: ChatRequest,
    ) -> Iterator[StreamDelta]:
        """Call the upstream endpoint and yield normalized deltas.

        Implementations MUST perform request dispatch eagerly enough that
        connection and authentication failures surface on the first ``next()``
        rather than being deferred indefinitely.
        """
        raise NotImplementedError

    @abstractmethod
    def fetch_models(self, token: str) -> tuple[ModelInfo, ...]:
        """Query the upstream catalog of available models."""
        raise NotImplementedError


class TokenSourcePort(ABC):
    """Read-only source of credentials.

    Modelled separately from TokenStoragePort so that read-only discovery
    mechanisms (an OS keyring we must not clobber, for example) do not have to
    implement no-op writes.
    """

    @abstractmethod
    def load(self) -> AuthToken | None:
        """Load a token if one is available here, otherwise None."""
        raise NotImplementedError


class TokenStoragePort(TokenSourcePort):
    """Outbound port for loading, saving and clearing authentication tokens."""

    @abstractmethod
    def save(self, token: AuthToken) -> None:
        """Persist the auth token durably."""
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Remove any persisted token. Must be safe to call when absent."""
        raise NotImplementedError


class OAuthProviderPort(ABC):
    """Outbound port for interacting with an identity provider."""

    @abstractmethod
    def start_interactive_flow(self) -> AuthToken:
        """Perform a loopback browser authorization and return the token."""
        raise NotImplementedError

    @abstractmethod
    def refresh_token(self, refresh_token: str) -> AuthToken:
        """Exchange a refresh token for a new access token."""
        raise NotImplementedError
