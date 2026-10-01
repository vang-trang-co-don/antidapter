import contextlib
import json
import logging
import secrets
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from config import OAuthConfig
from core.domain.entities import AuthToken
from core.domain.exceptions import AuthenticationError, UpstreamServiceError
from core.ports.outbound import OAuthProviderPort

logger = logging.getLogger(__name__)

_SUCCESS_PAGE = (
    b"<html><body><h1>Authentication successful</h1>"
    b"<p>You can close this tab and return to your terminal.</p></body></html>"
)
_FAILURE_PAGE = (
    b"<html><body><h1>Authentication failed</h1><p>Return to your terminal.</p></body></html>"
)
_AUTHORIZE_PROMPT = "\nOpen this URL in your browser to authorize Antidapter:\n\n    {url}\n\nWaiting for the login callback (up to the configured timeout)...\n"


class _CallbackResult:
    __slots__ = ("code", "error", "state")

    def __init__(self) -> None:
        self.code: str | None = None
        self.error: str | None = None
        self.state: str | None = None


class GoogleOAuthAdapter(OAuthProviderPort):
    """Google OAuth 2.0 loopback (installed app) flow with PKCE-less code exchange.

    The loopback listener is a plain HTTP server bound to an ephemeral port on
    the loopback interface. A random ``state`` is sent and verified on callback
    to prevent CSRF, and the wait is bounded by an explicit deadline so a
    cancelled consent screen fails fast instead of spinning forever.
    """

    def __init__(
        self,
        config: OAuthConfig,
        browser_opener: Callable[[str], bool] = webbrowser.open,
        clock: Callable[[], float] = time.monotonic,
        prompter: Callable[[str], None] | None = None,
        on_auth_url: Callable[[str], None] | None = None,
    ):
        self._config = config
        self._browser_opener = browser_opener
        self._clock = clock
        # Injected so the adapter never prints directly: the composition root
        # decides where the URL is shown, and tests can capture it.
        self._prompter = prompter or (lambda message: logger.info("%s", message))
        # Machine-readable hook for supervisors (the pi extension), which need
        # the bare URL rather than the human-formatted block.
        self._on_auth_url = on_auth_url or (lambda _url: None)

    def start_interactive_flow(self) -> AuthToken:
        state = secrets.token_urlsafe(32)
        result = _CallbackResult()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        redirect_uri = f"http://127.0.0.1:{port}{self._config.redirect_path}"
        auth_url = self._build_auth_url(redirect_uri, state)

        handler = self._build_handler(result, state)
        server = HTTPServer(("127.0.0.1", port), handler)
        server.timeout = 0.5

        logger.info("Waiting for OAuth callback on %s", redirect_uri)
        # Always show the URL: this is the fallback for headless hosts, remote
        # shells and browsers that fail to launch.
        self._on_auth_url(auth_url)
        self._prompter(_AUTHORIZE_PROMPT.format(url=auth_url))
        opened = False
        try:
            # webbrowser.open returns False on failure rather than raising, so
            # both outcomes have to be handled.
            opened = bool(self._browser_opener(auth_url))
        except Exception:
            opened = False
        if not opened:
            self._prompter("Could not open a browser automatically; open the URL above manually.")

        try:
            self._await_callback(server, result)
        finally:
            server.server_close()

        if result.error:
            raise AuthenticationError(f"Authorization was denied: {result.error}")
        if not result.code:
            raise AuthenticationError("Authorization failed: no code returned")
        return self._exchange_code(result.code, redirect_uri)

    def _build_auth_url(self, redirect_uri: str, state: str) -> str:
        params = {
            "client_id": self._config.client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(self._config.scopes),
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
        }
        return f"{self._config.auth_uri}?" + urllib.parse.urlencode(params)

    def _build_handler(
        self, result: _CallbackResult, expected_state: str
    ) -> type[BaseHTTPRequestHandler]:
        config = self._config

        class _CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path != config.redirect_path:
                    self.send_response(404)
                    self.end_headers()
                    return

                params = urllib.parse.parse_qs(parsed.query)

                if params.get("state", [""])[0] != expected_state:
                    self.send_response(400)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(_FAILURE_PAGE)
                    return

                error = params.get("error", [""])[0]
                code = params.get("code", [""])[0]
                if error:
                    result.error = error
                elif code:
                    result.code = code
                else:
                    result.error = "missing_code"

                self.send_response(200 if result.code else 400)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(_SUCCESS_PAGE if result.code else _FAILURE_PAGE)

            def log_message(self, fmt: str, *args: object) -> None:
                logger.debug("oauth callback: " + fmt, *args)

        return _CallbackHandler

    def _await_callback(self, server: HTTPServer, result: _CallbackResult) -> None:
        deadline = self._clock() + self._config.login_timeout
        while result.code is None and result.error is None:
            if self._clock() >= deadline:
                raise AuthenticationError(
                    f"Timed out after {self._config.login_timeout:.0f}s waiting for browser login"
                )
            # handle_request() returns after server.timeout with no request
            # handled, so this polls the deadline without busy-spinning.
            server.handle_request()

    def refresh_token(self, refresh_token: str) -> AuthToken:
        payload = self._token_request(
            {
                "client_id": self._config.client_id,
                "client_secret": self._config.client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            }
        )
        # Google omits refresh_token on refresh responses; keep the existing one.
        return self._token_from_payload(payload, fallback_refresh=refresh_token)

    def _exchange_code(self, code: str, redirect_uri: str) -> AuthToken:
        payload = self._token_request(
            {
                "client_id": self._config.client_id,
                "client_secret": self._config.client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri,
            }
        )
        return self._token_from_payload(payload)

    def _token_request(self, form: dict[str, str]) -> dict[str, Any]:
        data = urllib.parse.urlencode(form).encode("utf-8")
        req = urllib.request.Request(
            self._config.token_uri,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._config.request_timeout) as resp:
                payload: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
                return payload
        except urllib.error.HTTPError as exc:
            body = ""
            with contextlib.suppress(Exception):
                body = exc.read().decode("utf-8", errors="replace")
            with contextlib.suppress(Exception):
                exc.close()
            logger.error("OAuth token endpoint returned HTTP %s", exc.code)
            raise AuthenticationError(
                f"Token endpoint rejected the request (HTTP {exc.code}): {_summarize(body)}"
            ) from exc
        except urllib.error.URLError as exc:
            raise UpstreamServiceError(
                f"Could not reach the token endpoint: {exc.reason}",
                status_code=504,
                retryable=True,
            ) from exc
        except json.JSONDecodeError as exc:
            raise AuthenticationError("Token endpoint returned a malformed response") from exc

    def _token_from_payload(self, payload: dict, fallback_refresh: str | None = None) -> AuthToken:
        if "error" in payload:
            raise AuthenticationError(f"Authorization failed: {payload.get('error')}")
        try:
            access_token = payload["access_token"]
        except KeyError as exc:
            raise AuthenticationError("Token response contained no access_token") from exc
        try:
            expires_in = float(payload.get("expires_in", 3600))
        except (TypeError, ValueError):
            expires_in = 3600.0
        return AuthToken(
            access_token=access_token,
            refresh_token=payload.get("refresh_token") or fallback_refresh,
            expiry_time=time.time() + expires_in,
            token_type=payload.get("token_type", "Bearer"),
        )


def _summarize(body: str, limit: int = 200) -> str:
    flattened = " ".join(body.split())
    return flattened[:limit] + ("..." if len(flattened) > limit else "")
