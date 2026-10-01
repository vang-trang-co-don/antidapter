"""Composition root.

The only module permitted to know about every concrete adapter and service.
Nothing here contains business logic; it exists purely to wire the object
graph and to decide which policies (interactive login, for example) apply in
which context.
"""

import sys

from adapters.inbound.cli.cli_runner import CliRunner
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


def _console_prompter(message: str) -> None:
    """Show interactive prompts on stderr, keeping stdout free for piped output."""
    print(message, file=sys.stderr, flush=True)


class Container:
    def __init__(self, config: AppConfig):
        self.config = config

        # -- driven adapters ------------------------------------------------
        self.file_storage: TokenStoragePort = FileTokenStorage(config.storage)
        self.keyring_source: TokenSourcePort = SecretServiceCredentialSource(config.storage)
        self.token_storage: TokenStoragePort = ChainedTokenStorage(
            sink=self.file_storage,
            sources=(self.keyring_source,),
        )
        self.oauth_provider: OAuthProviderPort = GoogleOAuthAdapter(
            config.oauth, prompter=_console_prompter
        )
        self.upstream_model = GoogleCloudCodeAdapter(config.upstream)

        # -- use cases ------------------------------------------------------
        # The CLI may open a browser; a long-running server must not, because
        # that would block a request thread for the length of the login flow.
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
            serve=self._serve,
        )

    def _serve(self, host: str, port: int) -> None:
        server = create_http_server(
            host=host,
            port=port,
            translator=self.translator,
            chat_use_case=self.chat_service,
            catalog_use_case=self.model_catalog_service,
            config=self.config.server,
        )
        self._banner(host, port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            self._banner_shutdown()
        finally:
            server.server_close()

    def _banner(self, host: str, port: int) -> None:
        base = f"http://{host}:{port}"
        print("\n" + "=" * 60)
        print(f"  Antidapter listening on {base}")
        print(f"  OpenAI base URL:  {base}/v1")
        print(f"  Health check:     {base}/health")
        if self.config.server.api_key:
            print("  Inbound auth:     bearer token required")
        elif host not in ("127.0.0.1", "localhost"):
            print("  WARNING: no ANTIDAPTER_API_KEY set on a non-loopback bind")
        print("=" * 60 + "\n")

    def _banner_shutdown(self) -> None:
        print("\nShutting down Antidapter.")
