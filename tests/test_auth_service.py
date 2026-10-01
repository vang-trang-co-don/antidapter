import time
import unittest

from core.domain.entities import AuthToken
from core.domain.exceptions import AuthenticationError, TokenExpiredError
from core.ports.outbound import OAuthProviderPort, TokenStoragePort
from core.services.auth_service import AuthService


class FakeStorage(TokenStoragePort):
    def __init__(self, token: AuthToken | None = None, fail_save: bool = False):
        self.stored = token
        self.saved: list[AuthToken] = []
        self.cleared = 0
        self.fail_save = fail_save

    def load(self):
        return self.stored

    def save(self, token: AuthToken) -> None:
        if self.fail_save:
            raise OSError("disk full")
        self.stored = token
        self.saved.append(token)

    def clear(self) -> None:
        self.cleared += 1
        self.stored = None


class FakeOAuth(OAuthProviderPort):
    def __init__(
        self, refresh_error: Exception | None = None, login_error: Exception | None = None
    ):
        self.interactive_calls = 0
        self.refresh_calls = 0
        self._refresh_error = refresh_error
        self._login_error = login_error

    def start_interactive_flow(self) -> AuthToken:
        self.interactive_calls += 1
        if self._login_error:
            raise self._login_error
        return AuthToken("interactive-access", "interactive-refresh", time.time() + 3600)

    def refresh_token(self, refresh_token: str) -> AuthToken:
        self.refresh_calls += 1
        if self._refresh_error:
            raise self._refresh_error
        return AuthToken(f"refreshed-{refresh_token}", refresh_token, time.time() + 3600)


def valid_token() -> AuthToken:
    return AuthToken("valid", "rf", time.time() + 1000)


def expired_token(refresh: str = "my-refresh") -> AuthToken:
    return AuthToken("expired", refresh, time.time() - 100)


class TestAuthServiceHappyPaths(unittest.TestCase):
    def test_uses_valid_stored_token_without_network(self):
        storage, oauth = FakeStorage(valid_token()), FakeOAuth()
        self.assertEqual(AuthService(storage, oauth).ensure_authenticated(), "valid")
        self.assertEqual((oauth.refresh_calls, oauth.interactive_calls), (0, 0))

    def test_logs_in_when_no_token_exists(self):
        storage, oauth = FakeStorage(None), FakeOAuth()
        service = AuthService(storage, oauth)
        self.assertEqual(service.ensure_authenticated(), "interactive-access")
        self.assertEqual(oauth.interactive_calls, 1)
        self.assertEqual(storage.stored.access_token, "interactive-access")

    def test_refreshes_expired_token(self):
        storage, oauth = FakeStorage(expired_token()), FakeOAuth()
        self.assertEqual(AuthService(storage, oauth).ensure_authenticated(), "refreshed-my-refresh")
        self.assertEqual((oauth.refresh_calls, oauth.interactive_calls), (1, 0))

    def test_caches_after_first_call(self):
        storage, oauth = FakeStorage(valid_token()), FakeOAuth()
        service = AuthService(storage, oauth)
        self.assertEqual(service.ensure_authenticated(), "valid")
        storage.stored = None  # cache must keep serving
        self.assertEqual(service.ensure_authenticated(), "valid")

    def test_relogin_when_expired_without_refresh_token(self):
        storage, oauth = FakeStorage(AuthToken("stale", None, time.time() - 5)), FakeOAuth()
        self.assertEqual(AuthService(storage, oauth).ensure_authenticated(), "interactive-access")
        self.assertEqual(oauth.interactive_calls, 1)


class TestAuthServiceResilience(unittest.TestCase):
    def test_failed_refresh_falls_back_to_interactive_login(self):
        """Regression: a transient refresh failure used to be terminal."""
        oauth = FakeOAuth(refresh_error=AuthenticationError("upstream 500"))
        service = AuthService(FakeStorage(expired_token()), oauth)
        self.assertEqual(service.ensure_authenticated(), "interactive-access")
        self.assertEqual((oauth.refresh_calls, oauth.interactive_calls), (1, 1))

    def test_failed_refresh_raises_when_interactive_disabled(self):
        """Regression: a server must not hang for the length of a browser flow."""
        oauth = FakeOAuth(refresh_error=AuthenticationError("invalid_grant"))
        service = AuthService(FakeStorage(expired_token()), oauth, allow_interactive=False)
        with self.assertRaises(TokenExpiredError):
            service.ensure_authenticated()
        self.assertEqual(oauth.interactive_calls, 0)

    def test_missing_credentials_raise_immediately_when_interactive_disabled(self):
        """A request thread must never block on a browser login."""
        oauth = FakeOAuth()
        service = AuthService(FakeStorage(None), oauth, allow_interactive=False)
        with self.assertRaises(TokenExpiredError):
            service.ensure_authenticated()
        self.assertEqual(oauth.interactive_calls, 0)

    def test_login_interactive_refused_when_disabled(self):
        service = AuthService(FakeStorage(), FakeOAuth(), allow_interactive=False)
        with self.assertRaises(AuthenticationError):
            service.login_interactive()


class TestAuthServiceConcurrency(unittest.TestCase):
    def test_login_lock_does_not_block_the_cached_fast_path(self):
        """Regression: one browser login used to stall every other caller."""
        import threading

        entered = threading.Event()
        release = threading.Event()

        class BlockingOAuth(FakeOAuth):
            def start_interactive_flow(self) -> AuthToken:
                entered.set()
                release.wait(5)
                return super().start_interactive_flow()

        service = AuthService(FakeStorage(None), BlockingOAuth())
        slow = threading.Thread(target=service.ensure_authenticated)
        slow.start()
        self.assertTrue(entered.wait(5), "interactive login never started")

        # A second caller must not be blocked by the in-flight browser flow.
        service._cached_token = valid_token()
        results: list[str] = []
        fast = threading.Thread(target=lambda: results.append(service.ensure_authenticated()))
        fast.start()
        fast.join(2)
        self.assertEqual(results, ["valid"], "fast path was blocked by the login lock")

        release.set()
        slow.join(5)

    def test_only_one_interactive_login_runs_at_a_time(self):
        import threading

        barrier = threading.Barrier(4)
        oauth = FakeOAuth()
        service = AuthService(FakeStorage(None), oauth)
        errors: list[Exception] = []

        def worker():
            try:
                barrier.wait(5)
                service.ensure_authenticated()
            except Exception as exc:  # pragma: no cover - surfaced via assert
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)

        self.assertEqual(errors, [])
        self.assertEqual(oauth.interactive_calls, 1, "duplicate browser flows")


class TestAuthServiceLogout(unittest.TestCase):
    def test_logout_clears_cache_and_storage(self):
        storage = FakeStorage(valid_token())
        service = AuthService(storage, FakeOAuth())
        service.ensure_authenticated()
        service.logout()
        self.assertIsNone(service._cached_token)
        self.assertIsNone(storage.stored)
        self.assertEqual(storage.cleared, 1)


if __name__ == "__main__":
    unittest.main()
