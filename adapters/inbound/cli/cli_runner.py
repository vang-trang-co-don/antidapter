import argparse
import json
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

from config import AppConfig
from core.domain.exceptions import DomainException
from core.ports.inbound import (
    AuthUseCase,
    ChatUseCase,
    ModelCatalogUseCase,
    ProtocolTranslatorPort,
)

logger = logging.getLogger(__name__)

COMMANDS = ("serve", "login", "logout", "models", "quota", "model", "pi-config")


class CliRunner:
    """Driving adapter for the command line.

    Depends only on inbound ports, so every command is exercisable in tests
    without touching HTTP or the filesystem.
    """

    def __init__(
        self,
        config: AppConfig,
        auth_use_case: AuthUseCase,
        model_catalog_use_case: ModelCatalogUseCase,
        chat_use_case: ChatUseCase,
        translator: ProtocolTranslatorPort,
        serve: Callable[[str, int], None] | None = None,
        out: TextIO | None = None,
    ):
        self._config = config
        self._auth = auth_use_case
        self._catalog = model_catalog_use_case
        self._chat = chat_use_case
        self._translator = translator
        self._serve = serve
        self._out = out or sys.stdout

    def run(self, argv: Sequence[str]) -> int:
        args = self._build_parser().parse_args(list(argv))
        try:
            return self._dispatch(args)
        except DomainException as exc:
            self._print(f"error: {exc.message}")
            return 2
        except KeyboardInterrupt:
            self._print("\nInterrupted.")
            return 130

    def _build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="antidapter",
            description="Antidapter - OpenAI-compatible gateway for Google Antigravity models",
        )
        parser.add_argument(
            "command",
            nargs="?",
            default="serve",
            choices=COMMANDS,
            help="Command to run",
        )
        parser.add_argument("model_id", nargs="?", help="Model id, for the 'model' command")
        parser.add_argument(
            "--write",
            action="store_true",
            help="pi-config: merge the block into pi's models.json instead of printing it",
        )
        parser.add_argument(
            "--pi-models-file",
            default=None,
            help="pi-config: path to pi's models.json (default ~/.pi/agent/models.json)",
        )
        parser.add_argument("--host", default=self._config.server.host)
        parser.add_argument("--port", type=int, default=self._config.server.port)
        parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
        return parser

    def _dispatch(self, args: argparse.Namespace) -> int:
        handlers = {
            "serve": lambda: self._handle_serve(args.host, args.port),
            "login": self._handle_login,
            "logout": self._handle_logout,
            "models": self._handle_models,
            "quota": self._handle_quota,
            "model": lambda: self._handle_model(args.model_id),
            "pi-config": lambda: self._handle_pi_config(args),
        }
        return handlers[args.command]()

    def _handle_login(self) -> int:
        self._auth.login_interactive()
        self._print("[+] Logged in successfully.")
        return 0

    def _handle_logout(self) -> int:
        self._auth.logout()
        self._print("[+] Credentials cleared.")
        return 0

    def _handle_models(self) -> int:
        models = sorted(self._catalog.list_models(), key=lambda item: item.id)
        self._print(f"\nAvailable models ({len(models)}):")
        for model in models:
            flags = []
            if model.supports_thinking:
                flags.append("thinking")
            if model.supports_tools:
                flags.append("tools")
            suffix = f" [{', '.join(flags)}]" if flags else ""
            self._print(f"  - {model.id:32} {model.display_name}{suffix}")
        self._print("")
        return 0

    def _handle_model(self, model_id: str | None) -> int:
        if not model_id:
            self._print("error: `model` requires a model id")
            return 2
        model = self._catalog.get_model(model_id)
        self._print(self._translator.serialize_model(model).decode("utf-8"))
        return 0

    def _handle_quota(self) -> int:
        models = sorted(self._catalog.list_models(), key=lambda item: item.id)
        self._print("\nLive upstream quota:")
        self._print("-" * 72)
        for model in models:
            quota = model.quota_info
            if quota is None or quota.remaining_fraction is None:
                self._print(f"  {model.id:32} | unavailable")
                continue
            self._print(
                f"  {model.id:32} | {quota.remaining_fraction * 100:6.2f}% "
                f"| resets {quota.reset_time or 'unknown'}"
            )
        self._print("-" * 72)
        self._print("")
        return 0

    def _handle_pi_config(self, args: argparse.Namespace) -> int:
        """Emit (or install) the pi provider block for this gateway."""
        from adapters.inbound.pi.pi_provider import (
            DEFAULT_MODELS_PATH,
            DEFAULT_PROVIDER,
            PiProviderSettings,
            build_provider_block,
            merge_provider,
        )

        base_url = f"http://{self._config.server.host}:{self._config.server.port}/v1"
        settings = PiProviderSettings(
            provider=DEFAULT_PROVIDER,
            base_url=base_url,
            api_key=self._config.server.api_key or "none",
        )
        models = self._catalog.list_models()
        if not models:
            self._print("error: the upstream catalog is empty; is authentication working?")
            return 2

        block = build_provider_block(models, settings)

        if not args.write:
            self._print(json.dumps({settings.provider: block}, indent=2))
            self._print("")
            self._print("# Merge this into ~/.pi/agent/models.json, then run:")
            self._print(f"#   pi --provider {settings.provider} --model {models[0].id}")
            return 0

        target = (
            Path(args.pi_models_file).expanduser() if args.pi_models_file else DEFAULT_MODELS_PATH
        )
        changed, description = merge_provider(target, settings.provider, block)
        if not changed:
            self._print(f"error: {description}")
            return 2
        self._print(f"[+] {description}")
        self._print(
            f"    {len(models)} models; try: pi --provider {settings.provider} --list-models"
        )
        return 0

    def _handle_serve(self, host: str, port: int) -> int:
        # Authenticate before binding, so a missing credential fails fast at the
        # terminal instead of on the first request.
        self._auth.ensure_authenticated()
        if self._serve is None:
            self._print("error: no server factory configured")
            return 2
        self._serve(host, port)
        return 0

    def _print(self, message: str) -> None:
        print(message, file=self._out)
