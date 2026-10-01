"""Composition root.

The only module permitted to know about every concrete adapter and service.
Nothing here contains business logic; it exists purely to wire the object
graph and to decide which policies (interactive login, for example) apply in
which context.
"""

import sys
from collections.abc import Callable

from adapters.inbound.cli.cli_runner import CliRunner
from adapters.inbound.cli.events import SERVE_READY, EventSink, null_sink, stdout_sink
from adapters.inbound.http.openai_adapter import OpenAIProtocolTranslator
from adapters.inbound.http.server import create_http_server
from adapters.outbound.oauth.google_oauth import GoogleOAuthAdapter
from adapters.outbound.storage.composite_storage import ChainedTokenStorage
from adapters.outbound.storage.file_storage import FileTokenStorage
from adapters.outbound.storage.secret_service import SecretServiceCredentialSource
from adapters.outbound.upstream.google_cloudcode import GoogleCloudCodeAdapter
from config import AppConfig
from core.ports.inbound import (
    AuthUseCase,
    ChatUseCase,
    ModelCatalogUseCase,
    ProtocolTranslatorPort,
)
from core.ports.outbound import OAuthProviderPort, TokenSourcePort, TokenStoragePort
from core.services.auth_service import AuthService
from core.services.chat_service import ChatService
from core.services.model_catalog_service import ModelCatalogService


def console_prompter(message: str) -> None:
    """Show interactive prompts on stderr, keeping stdout free for piped output."""
    print(message, file=sys.stderr, flush=True)


def event_prompter(sink: EventSink) -> Callable[[str], None]:
    """Prompter for supervised runs: human text to stderr, a progress event out."""

    def prompt(message: str) -> None:
        first_line = message.strip().splitlines()
        summary = first_line[0] if first_line else ""
        if summary:
            sink.emit("progress", message=summary)
        print(message, file=sys.stderr, flush=True)

    return prompt


class Container:
    def __init__(self, config: AppConfig, events: EventSink | None = None):
        self.config = config
        self.events = events or null_sink()
        # When events are enabled, stdout carries NDJSON and must not be
        # polluted with human output; the banner goes to stderr instead.
        self._out = sys.stderr if self.events.enabled else sys.stdout
        self._prompter = event_prompter(self.events) if self.events.enabled else console_prompter

        # -- driven adapters ------------------------------------------------
        self.file_storage: TokenStoragePort = FileTokenStorage(config.storage)
        self.keyring_source: TokenSourcePort = SecretServiceCredentialSource(config.storage)
        self.token_storage: TokenStoragePort = ChainedTokenStorage(
            sink=self.file_storage,
            sources=(self.keyring_source,),
        )
        self.oauth_provider: OAuthProviderPort = GoogleOAuthAdapter(
            config.oauth,
            prompter=self._prompter,
            on_auth_url=lambda url: self.events.emit("auth_url", url=url),
        )
        self.upstream_model = GoogleCloudCodeAdapter(config.upstream)

        # -- use cases ------------------------------------------------------
        # Three auth policies, because "can this context open a browser?" is a
        # real distinction:
        #   cli_auth    - a human at a terminal, may prompt
        #   server_auth - long-running; follows configuration
        #   probe_auth  - never interactive, for supervised status checks
        self.cli_auth: AuthUseCase = AuthService(
            token_storage=self.token_storage,
            oauth_provider=self.oauth_provider,
            allow_interactive=True,
        )
        self.server_auth: AuthUseCase = AuthService(
            token_storage=self.token_storage,
            oauth_provider=self.oauth_provider,
            allow_interactive=config.server.allow_interactive_login,
        )
        self.probe_auth: AuthService = AuthService(
            token_storage=self.token_storage,
            oauth_provider=self.oauth_provider,
            allow_interactive=False,
        )

        self.chat_service: ChatUseCase = ChatService(
            auth_use_case=self.server_auth,
            upstream_model=self.upstream_model,
        )
        self.model_catalog_service: ModelCatalogUseCase = ModelCatalogService(
            auth_use_case=self.server_auth,
            upstream_model=self.upstream_model,
        )

        # -- driving adapters ----------------------------------------------
        self.translator: ProtocolTranslatorPort = OpenAIProtocolTranslator(config.protocol)
        self.cli_runner = CliRunner(
            config=config,
            auth_use_case=self.cli_auth,
            model_catalog_use_case=self.model_catalog_service,
            chat_use_case=self.chat_service,
            translator=self.translator,
            probe_auth=self.probe_auth,
            serve=self._serve,
            out=self._out,
            events=self.events,
        )

    def _serve(self, host: str, port: int, idle_timeout: float = 0.0) -> None:
        server = create_http_server(
            host=host,
            port=port,
            translator=self.translator,
            chat_use_case=self.chat_service,
            catalog_use_case=self.model_catalog_service,
            config=self.config.server,
            idle_timeout=idle_timeout,
        )
        bound_host = str(server.server_address[0])
        bound_port = int(server.server_address[1])
        # A supervisor binding port 0 needs to learn the real port before it
        # can address the gateway.
        self.events.emit(
            SERVE_READY,
            host=bound_host,
            port=bound_port,
            # `base_url` carries the /v1 prefix the OpenAI routes live under;
            # `health_url` is at the root, so supervisors must not guess.
            base_url=f"http://{bound_host}:{bound_port}/v1",
            health_url=f"http://{bound_host}:{bound_port}/health",
        )
        self._banner(bound_host, bound_port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            self._banner_shutdown()
        finally:
            server.server_close()

    def _banner(self, host: str, port: int) -> None:
        base = f"http://{host}:{port}"
        lines = [
            "\n" + "=" * 60,
            f"  Antidapter listening on {base}",
            f"  OpenAI base URL:  {base}/v1",
            f"  Health check:     {base}/health",
        ]
        if self.config.server.api_key:
            lines.append("  Inbound auth:     bearer token required")
        elif host not in ("127.0.0.1", "localhost"):
            lines.append("  WARNING: no ANTIDAPTER_API_KEY set on a non-loopback bind")
        lines.append("=" * 60 + "\n")
        for line in lines:
            print(line, file=self._out, flush=True)

    def _banner_shutdown(self) -> None:
        print("\nShutting down Antidapter.", file=self._out, flush=True)


__all__ = ["Container", "console_prompter", "event_prompter", "stdout_sink"]
