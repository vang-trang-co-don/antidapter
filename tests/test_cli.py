import io
import unittest

from adapters.inbound.cli.cli_runner import CliRunner
from adapters.inbound.http.openai_adapter import OpenAIProtocolTranslator
from config import (
    AppConfig,
    OAuthConfig,
    ProtocolConfig,
    ServerConfig,
    StorageConfig,
    UpstreamConfig,
)
from core.domain.entities import ChatResponse, ModelInfo, QuotaInfo
from core.domain.exceptions import AuthenticationError, ModelNotFoundError
from core.ports.inbound import AuthUseCase, ChatUseCase, ModelCatalogUseCase


class FakeAuth(AuthUseCase):
    def __init__(self, error=None):
        self.error = error
        self.logged_in = 0
        self.logged_out = 0
        self.authenticated = 0

    def ensure_authenticated(self) -> str:
        self.authenticated += 1
        if self.error:
            raise self.error
        return "token"

    def login_interactive(self) -> None:
        self.logged_in += 1

    def logout(self) -> None:
        self.logged_out += 1


class FakeCatalog(ModelCatalogUseCase):
    def __init__(self, error=None):
        self.error = error
        self.models = (
            ModelInfo(id="zeta", display_name="Zeta", provider="google", supports_thinking=True),
            ModelInfo(
                id="alpha",
                display_name="Alpha",
                provider="google",
                quota_info=QuotaInfo(remaining_fraction=0.42, reset_time="soon"),
            ),
            ModelInfo(id="noquota", display_name="NoQuota", provider="google"),
        )

    def list_models(self):
        if self.error:
            raise self.error
        return self.models

    def get_model(self, model_id: str) -> ModelInfo:
        for model in self.models:
            if model.id == model_id:
                return model
        raise ModelNotFoundError(model_id)


class FakeChat(ChatUseCase):
    def complete(self, request) -> ChatResponse:
        return ChatResponse(id="i", model=request.model, content="x", created_at=1)

    def complete_stream(self, request):
        return iter(())


def build(auth=None, catalog=None, serve=None):
    config = AppConfig(
        oauth=OAuthConfig(client_id="i", client_secret="s"),
        upstream=UpstreamConfig(),
        server=ServerConfig(),
        storage=StorageConfig(config_dir="/tmp/antidapter-test"),
        protocol=ProtocolConfig(),
    )
    out = io.StringIO()
    runner = CliRunner(
        config=config,
        auth_use_case=auth or FakeAuth(),
        model_catalog_use_case=catalog or FakeCatalog(),
        chat_use_case=FakeChat(),
        translator=OpenAIProtocolTranslator(config.protocol),
        serve=serve,
        out=out,
    )
    return runner, out


class TestCommands(unittest.TestCase):
    def test_login(self):
        auth = FakeAuth()
        runner, out = build(auth)
        self.assertEqual(runner.run(["login"]), 0)
        self.assertEqual(auth.logged_in, 1)
        self.assertIn("Logged in", out.getvalue())

    def test_logout(self):
        auth = FakeAuth()
        runner, _out = build(auth)
        self.assertEqual(runner.run(["logout"]), 0)
        self.assertEqual(auth.logged_out, 1)

    def test_models_are_sorted(self):
        runner, out = build()
        self.assertEqual(runner.run(["models"]), 0)
        text = out.getvalue()
        self.assertLess(text.index("alpha"), text.index("zeta"))
        self.assertIn("3", text.split("(")[1])
        self.assertIn("thinking", text)

    def test_quota_reports_unavailable_gracefully(self):
        runner, out = build()
        self.assertEqual(runner.run(["quota"]), 0)
        text = out.getvalue()
        self.assertIn("42.00%", text)
        self.assertIn("unavailable", text)

    def test_model_lookup(self):
        runner, out = build()
        self.assertEqual(runner.run(["model", "alpha"]), 0)
        self.assertIn('"id": "alpha"', out.getvalue())

    def test_model_lookup_requires_an_id(self):
        runner, out = build()
        self.assertEqual(runner.run(["model"]), 2)
        self.assertIn("requires a model id", out.getvalue())

    def test_unknown_model_exits_nonzero(self):
        runner, out = build()
        self.assertEqual(runner.run(["model", "ghost"]), 2)
        self.assertIn("not found", out.getvalue())


class TestExitCodes(unittest.TestCase):
    def test_domain_error_exits_nonzero(self):
        from core.domain.exceptions import UpstreamServiceError

        runner, out = build(catalog=FakeCatalog(error=UpstreamServiceError("upstream down")))
        self.assertEqual(runner.run(["models"]), 2)
        self.assertIn("upstream down", out.getvalue())

    def test_auth_failure_exits_nonzero(self):
        runner, out = build(auth=FakeAuth(error=AuthenticationError("no creds")))
        self.assertEqual(runner.run(["serve", "--port", "0"]), 2)
        self.assertIn("no creds", out.getvalue())

    def test_unknown_command_is_rejected_by_argparse(self):
        runner, _out = build()
        with self.assertRaises(SystemExit) as ctx:
            runner.run(["teleport"])
        self.assertNotEqual(ctx.exception.code, 0)


class TestServe(unittest.TestCase):
    def test_authenticates_before_binding(self):
        started = {}

        def serve(host, port):
            started["args"] = (host, port)

        auth = FakeAuth()
        runner, _out = build(auth=auth, serve=serve)
        self.assertEqual(runner.run(["serve", "--port", "9999"]), 0)
        self.assertEqual(started["args"], ("127.0.0.1", 9999))
        self.assertEqual(auth.authenticated, 1)

    def test_does_not_serve_when_auth_fails(self):
        called = []
        runner, _out = build(
            auth=FakeAuth(error=AuthenticationError("expired")),
            serve=lambda h, p: called.append((h, p)),
        )
        self.assertEqual(runner.run(["serve"]), 2)
        self.assertEqual(called, [], "server started despite failed auth")

    def test_reports_missing_server_factory(self):
        runner, out = build(serve=None)
        self.assertEqual(runner.run(["serve"]), 2)
        self.assertIn("no server factory", out.getvalue())


if __name__ == "__main__":
    unittest.main()
