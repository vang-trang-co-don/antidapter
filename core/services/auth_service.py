import logging
import threading

from core.domain.entities import AuthToken
from core.domain.exceptions import (
    AuthenticationError,
    DomainException,
    TokenExpiredError,
)
from core.ports.inbound import AuthUseCase
from core.ports.outbound import OAuthProviderPort, TokenStoragePort

logger = logging.getLogger(__name__)


class AuthService(AuthUseCase):
    """Manages token state and coordinates refresh and interactive login.

    Concurrency model
    -----------------
    ``_lock`` guards the fast paths (cache probe, storage read, refresh) and is
    only ever held for cheap, bounded work. Interactive authorization opens a
    browser and can block for minutes, so it runs under a separate
    ``_login_lock`` and is *never* executed while ``_lock`` is held. This keeps
    a login from stalling every other caller, and double-checking the cache
    under ``_login_lock`` guarantees only one browser flow runs at a time.
    """

    def __init__(
        self,
        token_storage: TokenStoragePort,
        oauth_provider: OAuthProviderPort,
        allow_interactive: bool = True,
    ):
        self._token_storage = token_storage
        self._oauth_provider = oauth_provider
        self._allow_interactive = allow_interactive
        self._cached_token: AuthToken | None = None
        self._lock = threading.Lock()
        self._login_lock = threading.Lock()

    def ensure_authenticated(self) -> str:
        """Return a valid access token. Thread-safe."""
        cached = self._cached_token
        if cached is not None and not cached.is_expired():
            return cached.access_token

        with self._lock:
            cached = self._cached_token
            if cached is not None and not cached.is_expired():
                return cached.access_token

            token = self._token_storage.load()

            if token is not None and not token.is_expired():
                return self._adopt(token)

            if token is not None and token.refresh_token:
                try:
                    logger.info("Access token expired. Refreshing...")
                    return self._adopt(self._oauth_provider.refresh_token(token.refresh_token))
                except DomainException as exc:
                    # A failed refresh must not be terminal: fall through to a
                    # fresh authorization so one transient upstream blip cannot
                    # permanently brick the gateway.
                    logger.warning("Token refresh failed (%s); re-authorizing", exc.message)

        if not self._allow_interactive:
            raise TokenExpiredError(
                "No valid credentials and interactive login is disabled. "
                "Run `antidapter login` and restart the gateway."
            )
        return self._authorize_interactive()

    def login_interactive(self) -> None:
        """Force a fresh interactive authorization."""
        if not self._allow_interactive:
            raise AuthenticationError("Interactive login is disabled in this context")
        with self._login_lock:
            token = self._oauth_provider.start_interactive_flow()
            self._persist(token)
        logger.info("Interactive login completed successfully.")

    def logout(self) -> None:
        """Drop cached and persisted credentials."""
        with self._lock:
            self._cached_token = None
        self._token_storage.clear()
        logger.info("Credentials cleared.")

    def _authorize_interactive(self) -> str:
        """Run at most one browser flow at a time, without holding _lock."""
        with self._login_lock:
            # Another thread may have finished authorizing while we waited.
            cached = self._cached_token
            if cached is not None and not cached.is_expired():
                return cached.access_token

            logger.info("No valid credentials. Starting interactive login...")
            token = self._oauth_provider.start_interactive_flow()
            self._persist(token)
            return token.access_token

    def _adopt(self, token: AuthToken) -> str:
        self._persist(token)
        return token.access_token

    def _persist(self, token: AuthToken) -> None:
        self._token_storage.save(token)
        self._cached_token = token
