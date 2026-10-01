import io
import json
import os
import tempfile
import time
import unittest
import unittest.mock
from collections.abc import Callable
from pathlib import Path

from adapters.inbound.cli.cli_runner import CliRunner
from adapters.inbound.cli.events import (
    ERROR,
    STATUS,
    SUCCESS,
    EventSink,
    null_sink,
)
from adapters.inbound.http.openai_adapter import OpenAIProtocolTranslator
from config import (
    AppConfig,
    OAuthConfig,
    ProtocolConfig,
    ServerConfig,
    StorageConfig,
    UpstreamConfig,
    _load_candidates,
    load_dotenv,
)
from core.domain.entities import ChatResponse, ModelInfo
from core.domain.exceptions import AuthenticationError, TokenExpiredError
from core.ports.inbound import AuthUseCase, ChatUseCase, ModelCatalogUseCase


class FakeAuth(AuthUseCase):
    def __init__(self, error: Exception | None = None):
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
        if self.error:
            raise self.error

    def logout(self) -> None:
        self.logged_out += 1


class FakeCatalog(ModelCatalogUseCase):
    def list_models(self):
        return (ModelInfo(id="m", display_name="M", provider="google"),)

    def get_model(self, model_id: str) -> ModelInfo:
        return ModelInfo(id=model_id, display_name=model_id, provider="google")


class FakeChat(ChatUseCase):
    def complete(self, request) -> ChatResponse:
        return ChatResponse(id="i", model=request.model, content="x", created_at=1)

    def complete_stream(self, request):
        return iter(())


def build(
    auth: AuthUseCase | None = None,
    probe: AuthUseCase | None = None,
    serve=None,
    events: EventSink | None = None,
    allow_interactive: bool = True,
):
    config = AppConfig(
        oauth=OAuthConfig(client_id="i", client_secret="s"),
        upstream=UpstreamConfig(),
        server=ServerConfig(allow_interactive_login=allow_interactive),
        storage=StorageConfig(config_dir=tempfile.mkdtemp()),
        protocol=ProtocolConfig(),
    )
    out = io.StringIO()
    runner = CliRunner(
        config=config,
        auth_use_case=auth or FakeAuth(),
        model_catalog_use_case=FakeCatalog(),
        chat_use_case=FakeChat(),
        translator=OpenAIProtocolTranslator(config.protocol),
        probe_auth=probe,
        serve=serve,
        out=out,
        events=events or null_sink(),
    )
    return runner, out


class TestEventSink(unittest.TestCase):
    def test_writes_one_json_object_per_line(self):
        stream = io.StringIO()
        sink = EventSink(stream)
        sink.emit("auth_url", url="https://example.test/x")
        sink.emit("success")
        lines = stream.getvalue().strip().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(
            json.loads(lines[0]), {"event": "auth_url", "url": "https://example.test/x"}
        )
        self.assertEqual(json.loads(lines[1]), {"event": "success"})

    def test_null_sink_writes_nothing_and_is_disabled(self):
        sink = null_sink()
        self.assertFalse(sink.enabled)
        sink.emit("whatever", detail=1)  # must not raise

    def test_survives_a_broken_pipe(self):
        """A supervisor closing the pipe must not take the command down."""

        class Broken(io.StringIO):
            def write(self, _data):
                raise BrokenPipeError(32, "Broken pipe")

        sink = EventSink(Broken())
        sink.emit("auth_url", url="x")
        self.assertFalse(sink.enabled, "should have disabled itself after the failure")

    def test_reports_enabled_state(self):
        self.assertTrue(EventSink(io.StringIO()).enabled)


class TestJsonEventOutput(unittest.TestCase):
    def test_login_emits_success(self):
        stream = io.StringIO()
        auth = FakeAuth()
        runner, _out = build(auth=auth, events=EventSink(stream))
        self.assertEqual(runner.run(["login", "--json-events"]), 0)
        events = [json.loads(line) for line in stream.getvalue().strip().splitlines()]
        self.assertEqual(events[-1]["event"], SUCCESS)
        self.assertEqual(auth.logged_in, 1)

    def test_login_failure_emits_an_error_event(self):
        stream = io.StringIO()
        runner, _out = build(auth=FakeAuth(AuthenticationError("denied")), events=EventSink(stream))
        code = runner.run(["login", "--json-events"])
        self.assertEqual(code, 2)
        events = [json.loads(line) for line in stream.getvalue().strip().splitlines()]
        self.assertEqual(events[-1]["event"], ERROR)
        self.assertEqual(events[-1]["message"], "denied")

    def test_a_null_sink_keeps_stdout_free_of_events(self):
        """main.py only installs a sink when --json-events is passed; without
        one the command must not print anything machine-readable."""
        runner, out = build(events=null_sink())
        self.assertEqual(runner.run(["login"]), 0)
        self.assertNotIn('{"event"', out.getvalue())
        self.assertIn("Logged in successfully", out.getvalue())


class TestEnsureAuth(unittest.TestCase):
    def test_reports_valid_credentials_and_exits_zero(self):
        stream = io.StringIO()
        runner, _out = build(probe=FakeAuth(), events=EventSink(stream))
        self.assertEqual(runner.run(["ensure-auth", "--json-events"]), 0)
        event = json.loads(stream.getvalue().strip())
        self.assertEqual(event["event"], STATUS)
        self.assertTrue(event["authenticated"])

    def test_reports_missing_credentials_and_exits_nonzero(self):
        stream = io.StringIO()
        runner, _out = build(
            probe=FakeAuth(TokenExpiredError("no token")),
            events=EventSink(stream),
        )
        self.assertEqual(runner.run(["ensure-auth", "--json-events"]), 1)
        event = json.loads(stream.getvalue().strip())
        self.assertFalse(event["authenticated"])
        self.assertIn("no token", event["message"])

    def test_never_prompts_even_when_interactive_is_allowed(self):
        """A supervised status probe must never be able to open a browser."""
        probe = FakeAuth()
        runner, _out = build(auth=FakeAuth(), probe=probe, allow_interactive=True)
        runner.run(["ensure-auth", "--json-events"])
        self.assertEqual(probe.logged_in, 0, "probe must not trigger an interactive login")
        self.assertEqual(probe.authenticated, 1)

    def test_falls_back_to_the_primary_auth_when_no_probe_is_given(self):
        auth = FakeAuth()
        runner, _out = build(auth=auth)
        self.assertEqual(runner.run(["ensure-auth", "--json-events"]), 0)
        self.assertEqual(auth.authenticated, 1)


def _record_serve(started: list) -> Callable[[str, int, float], None]:
    def serve(host: str, port: int, idle_timeout: float = 0.0) -> None:
        started.append((host, port, idle_timeout))

    return serve


class TestServePreflightRule(unittest.TestCase):
    """The supervised gateway must be able to start without credentials.

    Regression guard: an unconditional preflight would make a background spawn
    either block on a browser login or exit, and a browser must only ever be
    opened by an explicit /login.
    """

    def test_preflights_when_interactive_login_is_allowed(self):
        started = []
        auth = FakeAuth()
        runner, _out = build(auth=auth, serve=_record_serve(started), allow_interactive=True)
        self.assertEqual(runner.run(["serve", "--port", "0"]), 0)
        self.assertEqual(auth.authenticated, 1, "a human at a terminal should be preflighted")
        self.assertEqual(len(started), 1)

    def test_skips_preflight_when_interactive_login_is_disabled(self):
        started = []
        auth = FakeAuth(error=TokenExpiredError("no credentials"))
        runner, _out = build(auth=auth, serve=_record_serve(started), allow_interactive=False)
        self.assertEqual(runner.run(["serve", "--port", "0"]), 0)
        self.assertEqual(auth.authenticated, 0, "must not attempt auth at all")
        self.assertEqual(len(started), 1, "server should still start")

    def test_serve_forwards_port_and_idle_timeout(self):
        started = []
        runner, _out = build(serve=_record_serve(started), allow_interactive=False)
        runner.run(["serve", "--port", "0", "--host", "127.0.0.1", "--idle-timeout", "900"])
        self.assertEqual(started, [("127.0.0.1", 0, 900.0)])

    def test_idle_timeout_defaults_to_disabled(self):
        started = []
        runner, _out = build(serve=_record_serve(started), allow_interactive=False)
        runner.run(["serve", "--port", "0"])
        self.assertEqual(started, [("127.0.0.1", 0, 0.0)])


class TestIdleWatchdog(unittest.TestCase):
    """The orphan backstop: a supervised gateway must retire on its own."""

    def test_fires_after_the_idle_window(self):
        from adapters.inbound.http.server import IdleWatchdog

        class FakeServer:
            def __init__(self):
                self.stopped = False

            def shutdown(self) -> None:
                self.stopped = True

        server = FakeServer()
        watchdog = IdleWatchdog(server, 0.2)
        watchdog.start()
        deadline = time.time() + 5
        while time.time() < deadline and not server.stopped:
            time.sleep(0.05)
        self.assertTrue(server.stopped, "watchdog never shut the server down")
        watchdog.stop()

    def test_activity_defers_shutdown(self):
        from adapters.inbound.http.server import IdleWatchdog

        class FakeServer:
            def __init__(self):
                self.stopped = False

            def shutdown(self) -> None:
                self.stopped = True

        server = FakeServer()
        watchdog = IdleWatchdog(server, 0.6)
        watchdog.start()
        # Keep touching it; it must not shut down while traffic continues.
        for _ in range(8):
            time.sleep(0.1)
            watchdog.touch()
        self.assertFalse(server.stopped, "shut down despite continued activity")
        watchdog.stop()

    def test_zero_timeout_never_starts(self):
        from adapters.inbound.http.server import IdleWatchdog

        class FakeServer:
            def shutdown(self) -> None:
                raise AssertionError("must not shut down when disabled")

        IdleWatchdog(FakeServer(), 0.0).start()
        time.sleep(0.2)

    def test_server_creates_one_when_asked(self):
        from adapters.inbound.http.server import _WatchdogServer

        self.assertTrue(hasattr(_WatchdogServer, "watchdog"))
        self.assertTrue(_WatchdogServer.daemon_threads)


class TestDotenvLookupOrder(unittest.TestCase):
    def setUp(self):
        self.saved = dict(os.environ)
        self.dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.saved)

    def _write(self, name: str, body: str) -> Path:
        target = self.dir / name
        target.write_text(body, encoding="utf-8")
        return target

    def test_explicit_path_is_exclusive(self):
        target = self._write("custom.env", "ANTIDAPTER_TEST_DOTENV=from-explicit\n")
        os.environ.pop("ANTIDAPTER_TEST_DOTENV", None)
        load_dotenv(target)
        self.assertEqual(os.environ["ANTIDAPTER_TEST_DOTENV"], "from-explicit")

    def test_missing_path_is_ignored(self):
        os.environ.pop("ANTIDAPTER_TEST_DOTENV", None)
        load_dotenv(self.dir / "does-not-exist.env")
        self.assertNotIn("ANTIDAPTER_TEST_DOTENV", os.environ)

    def test_home_config_dir_is_a_fallback(self):
        """A package clone has no .env, so the documented location must work."""
        fake_home = self.dir / "home"
        config_dir = fake_home / ".config" / "antidapter"
        config_dir.mkdir(parents=True)
        (config_dir / ".env").write_text(
            "ANTIDAPTER_TEST_DOTENV=from-config-dir\n", encoding="utf-8"
        )
        os.environ.pop("ANTIDAPTER_TEST_DOTENV", None)
        with unittest.mock.patch.dict(os.environ, {"HOME": str(fake_home)}):
            load_dotenv()
            # Read inside the patch: patch.dict restores os.environ on exit.
            self.assertEqual(os.environ.get("ANTIDAPTER_TEST_DOTENV"), "from-config-dir")

    def test_repo_env_does_not_shadow_the_config_dir(self):
        """All candidates merge, first-wins per key, so a partial .env is safe."""
        fake_home = self.dir / "home2"
        config_dir = fake_home / ".config" / "antidapter"
        config_dir.mkdir(parents=True)
        (config_dir / ".env").write_text(
            "ANTIDAPTER_TEST_SHARED=from-config-dir\nANTIDAPTER_TEST_ONLY_HOME=from-config-dir\n",
            encoding="utf-8",
        )
        repo_env = self._write("partial.env", "ANTIDAPTER_TEST_SHARED=from-partial\n")
        os.environ.pop("ANTIDAPTER_TEST_SHARED", None)
        os.environ.pop("ANTIDAPTER_TEST_ONLY_HOME", None)
        # The "repo" .env defines only one of the two keys.
        _load_candidates([repo_env, config_dir / ".env"])
        self.assertEqual(os.environ.get("ANTIDAPTER_TEST_SHARED"), "from-partial")
        self.assertEqual(os.environ.get("ANTIDAPTER_TEST_ONLY_HOME"), "from-config-dir")

    def test_real_environment_is_never_overridden(self):
        target = self._write("override.env", "ANTIDAPTER_TEST_DOTENV=from-file\n")
        os.environ["ANTIDAPTER_TEST_DOTENV"] = "from-env"
        load_dotenv(target)
        self.assertEqual(os.environ["ANTIDAPTER_TEST_DOTENV"], "from-env")


if __name__ == "__main__":
    unittest.main()
