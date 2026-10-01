import json
import os
import stat
import tempfile
import unittest

from adapters.outbound.storage.composite_storage import ChainedTokenStorage
from adapters.outbound.storage.file_storage import FileTokenStorage
from adapters.outbound.storage.secret_service import SecretServiceCredentialSource
from config import StorageConfig
from core.domain.entities import AuthToken
from core.domain.exceptions import DomainException
from core.ports.outbound import TokenSourcePort, TokenStoragePort


def token(access: str = "access") -> AuthToken:
    return AuthToken(access, "refresh", 12345.0)


class TestFileTokenStorage(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.config = StorageConfig(config_dir=self.dir)
        self.storage = FileTokenStorage(self.config)

    def test_round_trip(self):
        self.storage.save(token())
        self.assertEqual(self.storage.load(), token())

    def test_load_returns_none_when_absent(self):
        self.assertIsNone(self.storage.load())

    def test_credentials_are_owner_only(self):
        """Regression: the file used to be created world-readable (0644)."""
        self.storage.save(token())
        mode = os.stat(self.config.file_path).st_mode
        self.assertEqual(stat.S_IMODE(mode), 0o600)
        self.assertFalse(mode & (stat.S_IRGRP | stat.S_IROTH))

    def test_config_dir_is_owner_only(self):
        nested = os.path.join(self.dir, "sub")
        FileTokenStorage(StorageConfig(config_dir=nested)).save(token())
        self.assertEqual(stat.S_IMODE(os.stat(nested).st_mode), 0o700)

    def test_save_is_atomic_and_leaves_no_temp_files(self):
        self.storage.save(token())
        self.storage.save(token("second"))
        leftovers = [n for n in os.listdir(self.dir) if n.startswith(".credentials-")]
        self.assertEqual(leftovers, [], "temp file was not cleaned up")
        self.assertEqual(self.storage.load().access_token, "second")

    def test_clear_removes_the_file(self):
        self.storage.save(token())
        self.storage.clear()
        self.assertFalse(os.path.exists(self.config.file_path))
        self.assertIsNone(self.storage.load())

    def test_clear_is_idempotent(self):
        self.storage.clear()
        self.storage.clear()

    def test_corrupt_file_is_ignored_not_fatal(self):
        with open(self.config.file_path, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        self.assertIsNone(self.storage.load())

    def test_file_missing_access_token_is_ignored(self):
        os.makedirs(self.dir, exist_ok=True)
        with open(self.config.file_path, "w", encoding="utf-8") as handle:
            json.dump({"refresh_token": "r"}, handle)
        self.assertIsNone(self.storage.load())

    def test_save_failure_raises_domain_error(self):
        blocked = os.path.join(self.dir, "afile")
        with open(blocked, "w", encoding="utf-8") as handle:
            handle.write("x")
        storage = FileTokenStorage(StorageConfig(config_dir=blocked, filename="nested/creds.json"))
        with self.assertRaises(DomainException):
            storage.save(token())


class NullSource(TokenSourcePort):
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error
        self.calls = 0

    def load(self):
        self.calls += 1
        if self._error:
            raise self._error
        return self._result


class OtherStorage(FileTokenStorage):
    pass


class TestChainedTokenStorage(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.sink = FileTokenStorage(StorageConfig(config_dir=self.dir))

    def test_reads_from_sink_first(self):
        self.sink.save(token("from-sink"))
        source = NullSource(token("from-source"))
        chained = ChainedTokenStorage(self.sink, (source,))
        self.assertEqual(chained.load().access_token, "from-sink")
        self.assertEqual(source.calls, 0, "fallback was consulted unnecessarily")

    def test_promotes_discovered_token_to_sink(self):
        source = NullSource(token("discovered"))
        chained = ChainedTokenStorage(self.sink, (source,))
        self.assertEqual(chained.load().access_token, "discovered")
        self.assertEqual(self.sink.load().access_token, "discovered")

    def test_falls_through_multiple_sources_in_order(self):
        empty = NullSource(None)
        found = NullSource(token("second"))
        chained = ChainedTokenStorage(self.sink, (empty, found))
        self.assertEqual(chained.load().access_token, "second")

    def test_returns_none_when_nothing_found(self):
        self.assertIsNone(ChainedTokenStorage(self.sink, (NullSource(None),)).load())

    def test_a_broken_source_does_not_prevent_discovery(self):
        broken = NullSource(error=RuntimeError("dbus died"))
        working = NullSource(token("ok"))
        chained = ChainedTokenStorage(self.sink, (broken, working))
        self.assertEqual(chained.load().access_token, "ok")

    def test_save_goes_to_sink_only(self):
        source = NullSource(token("read-only"))
        chained = ChainedTokenStorage(self.sink, (source,))
        chained.save(token("written"))
        self.assertEqual(self.sink.load().access_token, "written")

    def test_promotion_failure_does_not_lose_the_token(self):
        class BrokenSink(TokenStoragePort):
            def load(self):
                return None

            def save(self, token):
                raise OSError("read-only filesystem")

            def clear(self):
                pass

        chained = ChainedTokenStorage(BrokenSink(), (NullSource(token("x")),))
        self.assertEqual(chained.load().access_token, "x")

    def test_clear_clears_writable_stores_and_skips_read_only_ones(self):
        other_dir = tempfile.mkdtemp()
        other = FileTokenStorage(StorageConfig(config_dir=other_dir))
        other.save(token())
        chained = ChainedTokenStorage(self.sink, (NullSource(token("r")), other))
        self.sink.save(token())
        chained.clear()
        self.assertIsNone(self.sink.load())
        self.assertIsNone(other.load())


class TestSecretServiceCredentialSource(unittest.TestCase):
    """The keyring source is read-only and must degrade gracefully."""

    def setUp(self):
        self.config = StorageConfig(
            config_dir=tempfile.mkdtemp(),
            keyring_service="gemini",
            keyring_username="antigravity",
        )
        self.source = SecretServiceCredentialSource(self.config)

    def test_does_not_implement_writable_storage(self):
        """Regression: save() used to be a silent no-op `pass`."""
        self.assertIsInstance(self.source, TokenSourcePort)
        self.assertNotIsInstance(self.source, TokenStoragePort)

    def test_returns_none_when_gi_is_unavailable(self):
        self.assertIsNone(self.source.load())

    def test_maps_legacy_agy_payload_shape(self):
        parsed = self.source._to_token(
            {"token": {"access_token": "a", "refresh_token": "r", "expiry": 99.0}}
        )
        self.assertEqual(parsed.access_token, "a")
        self.assertEqual(parsed.expiry_time, 99.0)

    def test_accepts_flat_payload_shape(self):
        parsed = self.source._to_token({"access_token": "a", "expiry_time": 5.0})
        self.assertEqual(parsed.expiry_time, 5.0)

    def test_ignores_malformed_payloads(self):
        for payload in (None, [], {}, {"token": {}}, {"access_token": ""}):
            self.assertIsNone(self.source._to_token(payload), payload)


if __name__ == "__main__":
    unittest.main()
