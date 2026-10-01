import json
import socket
import threading
import time
import unittest
import urllib.error
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

from adapters.outbound.oauth.google_oauth import GoogleOAuthAdapter
from config import OAuthConfig
from core.domain.exceptions import AuthenticationError, UpstreamServiceError


def config(**overrides) -> OAuthConfig:
    defaults = {
        "client_id": "test-client",
        "client_secret": "test-secret",
        "login_timeout": 5.0,
        "request_timeout": 2.0,
    }
    defaults.update(overrides)
    return OAuthConfig(**defaults)


class FakeTokenEndpoint:
    """A local HTTP server standing in for the Google token endpoint."""

    def __init__(self, payload: dict, status: int = 200):
        self.payload = payload
        self.status = status
        self.requests: list[dict] = []
        handler = self._make_handler()
        self.server = HTTPServer(("127.0.0.1", 0), handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def token_uri(self) -> str:
        return f"http://127.0.0.1:{self.port}/token"

    def _make_handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                form = urllib.parse.parse_qs(self.rfile.read(length).decode())
                outer.requests.append({k: v[0] for k, v in form.items()})
                body = json.dumps(outer.payload).encode()
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt, *args) -> None:
                pass

        return Handler

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class TestAuthUrl(unittest.TestCase):
    def test_contains_required_oauth_parameters(self):
        adapter = GoogleOAuthAdapter(config())
        url = adapter._build_auth_url("http://127.0.0.1:9999/cb", "state-123")
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        self.assertEqual(query["client_id"], ["test-client"])
        self.assertEqual(query["response_type"], ["code"])
        self.assertEqual(query["access_type"], ["offline"])
        self.assertEqual(query["state"], ["state-123"])
        self.assertIn("aicode", query["scope"][0])
        self.assertNotIn("test-secret", url, "client secret must not appear in the URL")


class TestCallbackHandler(unittest.TestCase):
    """Exercises the loopback listener against real sockets."""

    def _run_flow(self, query: str, token_payload=None, token_status=200):
        endpoint = FakeTokenEndpoint(
            token_payload or {"access_token": "a", "refresh_token": "r", "expires_in": 3600},
            status=token_status,
        )
        opened: list[str] = []
        adapter = GoogleOAuthAdapter(
            config(token_uri=endpoint.token_uri),
            browser_opener=opened.append,
        )
        errors: list[Exception] = []

        def run() -> None:
            try:
                adapter.start_interactive_flow()
            except Exception as exc:  # captured for assertions
                errors.append(exc)
            finally:
                endpoint.close()

        thread = threading.Thread(target=run)
        thread.start()

        # The opener receives the URL; extract the loopback port and hit it.
        self.assertTrue(wait_for(lambda: bool(opened)), "browser was never opened")
        auth_url = opened[0]
        redirect = urllib.parse.parse_qs(urllib.parse.urlparse(auth_url).query)["redirect_uri"][0]
        parsed = urllib.parse.urlparse(redirect)
        state = urllib.parse.parse_qs(urllib.parse.urlparse(auth_url).query)["state"][0]

        status = None
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                with socket.create_connection(
                    ("127.0.0.1", int(parsed.netloc.split(":")[1])), timeout=2
                ) as s:
                    s.sendall(
                        f"GET {parsed.path}?{query.format(state=state)} HTTP/1.1\r\nHost: x\r\n\r\n".encode()
                    )
                    status = s.recv(64).split()[1].decode()
                break
            except OSError:
                time.sleep(0.05)
        thread.join(timeout=10)
        return status, errors, endpoint

    def test_successful_callback_exchanges_the_code(self):
        status, errors, endpoint = self._run_flow("code=abc123&state={state}")
        self.assertEqual(status, "200")
        self.assertEqual(errors, [])
        self.assertEqual(endpoint.requests[0]["grant_type"], "authorization_code")
        self.assertEqual(endpoint.requests[0]["code"], "abc123")
        self.assertEqual(endpoint.requests[0]["client_secret"], "test-secret")

    def test_mismatched_state_is_refused(self):
        """Regression: the callback used to accept any request."""
        status, errors, _ = self._run_flow("code=abc&state=attacker-state")
        self.assertEqual(status, "400")
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], AuthenticationError)

    def test_access_denied_fails_fast_instead_of_looping(self):
        """Regression: cancelling consent used to spin forever at 100% CPU."""
        status, errors, _ = self._run_flow("error=access_denied&state={state}")
        self.assertEqual(status, "400")
        self.assertEqual(len(errors), 1)
        self.assertIn("access_denied", errors[0].message)

    def test_missing_code_fails_fast(self):
        _status, errors, _ = self._run_flow("state={state}")
        self.assertEqual(len(errors), 1)
        self.assertIn("missing_code", errors[0].message)

    def test_request_to_another_path_is_ignored(self):
        """A stray request must not satisfy the flow or exchange a code."""
        endpoint = FakeTokenEndpoint({"access_token": "a"})
        opened: list[str] = []
        adapter = GoogleOAuthAdapter(
            config(token_uri=endpoint.token_uri, login_timeout=1.0),
            browser_opener=opened.append,
        )
        errors: list[Exception] = []

        def run() -> None:
            try:
                adapter.start_interactive_flow()
            except Exception as exc:
                errors.append(exc)
            finally:
                endpoint.close()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        self.assertTrue(wait_for(lambda: bool(opened)), "browser was never opened")
        redirect = urllib.parse.parse_qs(urllib.parse.urlparse(opened[0]).query)["redirect_uri"][0]
        port = int(urllib.parse.urlparse(redirect).netloc.split(":")[1])

        with socket.create_connection(("127.0.0.1", port), timeout=2) as sock:
            sock.sendall(b"GET /favicon.ico HTTP/1.1\r\nHost: x\r\n\r\n")
            sock.recv(64)
        thread.join(timeout=10)

        self.assertEqual(len(errors), 1, "stray request should not have completed the flow")
        self.assertIn("Timed out", errors[0].message)
        self.assertEqual(endpoint.requests, [], "a code was exchanged for a stray request")


class TestAuthorizationPrompt(unittest.TestCase):
    """Regression: the auth URL used to be swallowed instead of printed.

    webbrowser.open returns False on failure rather than raising, so a headless
    host would previously hang with no way to see the URL.
    """

    def test_url_is_prompted_even_when_the_browser_opens(self):
        endpoint = FakeTokenEndpoint({"access_token": "a", "expires_in": 1})
        messages: list[str] = []
        adapter = GoogleOAuthAdapter(
            config(token_uri=endpoint.token_uri, login_timeout=0.4),
            browser_opener=lambda _url: True,
            prompter=messages.append,
        )
        try:
            with self.assertRaises(AuthenticationError):
                adapter.start_interactive_flow()
        finally:
            endpoint.close()
        joined = "\n".join(messages)
        self.assertIn("accounts.google.com", joined)
        self.assertIn("state=", joined)

    def test_failure_to_open_is_reported(self):
        endpoint = FakeTokenEndpoint({"access_token": "a", "expires_in": 1})
        messages: list[str] = []
        adapter = GoogleOAuthAdapter(
            config(token_uri=endpoint.token_uri, login_timeout=0.4),
            browser_opener=lambda _url: False,
            prompter=messages.append,
        )
        try:
            with self.assertRaises(AuthenticationError):
                adapter.start_interactive_flow()
        finally:
            endpoint.close()
        self.assertIn("open the URL above manually", "\n".join(messages))

    def test_browser_exceptions_are_not_fatal(self):
        endpoint = FakeTokenEndpoint({"access_token": "a", "expires_in": 1})
        messages: list[str] = []

        def boom(_url):
            raise RuntimeError("no display")

        adapter = GoogleOAuthAdapter(
            config(token_uri=endpoint.token_uri, login_timeout=0.4),
            browser_opener=boom,
            prompter=messages.append,
        )
        try:
            with self.assertRaises(AuthenticationError):
                adapter.start_interactive_flow()
        finally:
            endpoint.close()
        self.assertIn("open the URL above manually", "\n".join(messages))


class TestRefresh(unittest.TestCase):
    def test_retains_refresh_token_when_omitted(self):
        """Regression: losing the refresh token forces a full re-login."""
        endpoint = FakeTokenEndpoint({"access_token": "new", "expires_in": 3600})
        try:
            adapter = GoogleOAuthAdapter(config(token_uri=endpoint.token_uri))
            token = adapter.refresh_token("original-refresh")
            self.assertEqual(token.access_token, "new")
            self.assertEqual(token.refresh_token, "original-refresh")
            self.assertGreater(token.expiry_time, time.time())
        finally:
            endpoint.close()

    def test_uses_provided_refresh_token_when_returned(self):
        endpoint = FakeTokenEndpoint(
            {"access_token": "new", "refresh_token": "rotated", "expires_in": 60}
        )
        try:
            adapter = GoogleOAuthAdapter(config(token_uri=endpoint.token_uri))
            self.assertEqual(adapter.refresh_token("old").refresh_token, "rotated")
        finally:
            endpoint.close()

    def test_http_error_is_translated(self):
        endpoint = FakeTokenEndpoint({"error": "invalid_grant"}, status=400)
        try:
            adapter = GoogleOAuthAdapter(config(token_uri=endpoint.token_uri))
            with self.assertRaises(AuthenticationError) as ctx:
                adapter.refresh_token("bad")
            self.assertIn("400", ctx.exception.message)
        finally:
            endpoint.close()

    def test_unreachable_endpoint_is_retryable(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            dead_port = probe.getsockname()[1]
        adapter = GoogleOAuthAdapter(config(token_uri=f"http://127.0.0.1:{dead_port}/token"))
        with self.assertRaises(UpstreamServiceError) as ctx:
            adapter.refresh_token("r")
        self.assertTrue(ctx.exception.retryable)

    def test_error_payload_is_surfaced(self):
        endpoint = FakeTokenEndpoint({"error": "invalid_scope"})
        try:
            adapter = GoogleOAuthAdapter(config(token_uri=endpoint.token_uri))
            with self.assertRaises(AuthenticationError) as ctx:
                adapter.refresh_token("r")
            self.assertIn("invalid_scope", ctx.exception.message)
        finally:
            endpoint.close()

    def test_missing_access_token_is_rejected(self):
        endpoint = FakeTokenEndpoint({"expires_in": 10})
        try:
            adapter = GoogleOAuthAdapter(config(token_uri=endpoint.token_uri))
            with self.assertRaises(AuthenticationError):
                adapter.refresh_token("r")
        finally:
            endpoint.close()

    def test_malformed_json_is_rejected(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                body = b"not json"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt, *args) -> None:
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            adapter = GoogleOAuthAdapter(
                config(token_uri=f"http://127.0.0.1:{server.server_address[1]}/token")
            )
            with self.assertRaises(AuthenticationError):
                adapter.refresh_token("r")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


class TestDeadline(unittest.TestCase):
    def test_timeout_raises_rather_than_waiting_forever(self):
        """Regression: server.timeout was never checked, so the loop spun."""
        from adapters.outbound.oauth.google_oauth import _CallbackResult

        adapter = GoogleOAuthAdapter(config(login_timeout=0.3))
        ticks = {"n": 0}

        def clock() -> float:
            ticks["n"] += 1
            return ticks["n"] * 0.1

        adapter._clock = clock
        server = HTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
        # start_interactive_flow sets this; set it here too so handle_request
        # polls instead of blocking forever.
        server.timeout = 0.05
        try:
            with self.assertRaises(AuthenticationError) as ctx:
                adapter._await_callback(server, _CallbackResult())
            self.assertIn("Timed out", ctx.exception.message)
        finally:
            server.server_close()


if __name__ == "__main__":
    unittest.main()
