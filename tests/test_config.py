import dataclasses
import os
import tempfile
import unittest
from pathlib import Path

from config import (
    AppConfig,
    OAuthConfig,
    ProtocolConfig,
    ServerConfig,
    StorageConfig,
    UpstreamConfig,
    load_dotenv,
)
from core.domain.exceptions import ConfigurationError

MINIMAL = {
    "ANTIDAPTER_CLIENT_ID": "id",
    "ANTIDAPTER_CLIENT_SECRET": "secret",
}


class TestSecretsAreNotHardcoded(unittest.TestCase):
    def test_no_credential_is_embedded_in_the_source(self):
        """Regression: a Google client_secret was committed to config.py."""
        source = Path(__file__).resolve().parent.parent / "config.py"
        text = source.read_text(encoding="utf-8")
        # Assembled from fragments so this test does not match itself.
        for marker in ("GOCS" + "PX", "apps." + "googleusercontent.com"):
            self.assertNotIn(marker, text, "an OAuth credential is hardcoded in config.py")

    def test_missing_credentials_raise_a_clear_error(self):
        with self.assertRaises(ConfigurationError) as ctx:
            OAuthConfig.from_env({})
        self.assertIn("ANTIDAPTER_CLIENT_ID", ctx.exception.message)


class TestOAuthConfig(unittest.TestCase):
    def test_reads_from_env(self):
        config = OAuthConfig.from_env(MINIMAL)
        self.assertEqual((config.client_id, config.client_secret), ("id", "secret"))

    def test_defaults_are_applied(self):
        config = OAuthConfig.from_env(MINIMAL)
        self.assertEqual(config.auth_uri, "https://accounts.google.com/o/oauth2/v2/auth")
        self.assertTrue(any("auth/aicode" in scope for scope in config.scopes))

    def test_timeout_is_configurable(self):
        config = OAuthConfig.from_env({**MINIMAL, "ANTIDAPTER_LOGIN_TIMEOUT": "30"})
        self.assertEqual(config.login_timeout, 30.0)

    def test_non_numeric_timeout_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            OAuthConfig.from_env({**MINIMAL, "ANTIDAPTER_LOGIN_TIMEOUT": "soon"})


class TestUpstreamConfig(unittest.TestCase):
    def test_defaults(self):
        config = UpstreamConfig.from_env({})
        self.assertEqual(config.base_url, "https://daily-cloudcode-pa.googleapis.com")
        self.assertEqual(config.max_retries, 3)
        self.assertGreater(config.request_timeout, 0)

    def test_trailing_slash_is_stripped(self):
        self.assertEqual(
            UpstreamConfig.from_env({"ANTIDAPTER_UPSTREAM_URL": "https://x.test/"}).base_url,
            "https://x.test",
        )

    def test_retry_knobs(self):
        config = UpstreamConfig.from_env(
            {
                "ANTIDAPTER_MAX_RETRIES": "5",
                "ANTIDAPTER_RETRY_BASE_DELAY": "0.25",
                "ANTIDAPTER_RETRY_MAX_DELAY": "2",
            }
        )
        self.assertEqual((config.max_retries, config.retry_base_delay), (5, 0.25))

    def test_dead_load_code_assist_path_is_gone(self):
        self.assertFalse(hasattr(UpstreamConfig, "load_code_assist_path"))


class TestServerConfig(unittest.TestCase):
    def test_defaults(self):
        config = ServerConfig.from_env({})
        self.assertEqual((config.host, config.port), ("127.0.0.1", 8080))
        self.assertIsNone(config.api_key)

    def test_api_key_is_read_from_env(self):
        self.assertEqual(ServerConfig.from_env({"ANTIDAPTER_API_KEY": "k"}).api_key, "k")

    def test_blank_api_key_means_disabled(self):
        self.assertIsNone(ServerConfig.from_env({"ANTIDAPTER_API_KEY": ""}).api_key)

    def test_invalid_port_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            ServerConfig(port=0)
        with self.assertRaises(ConfigurationError):
            ServerConfig(port=70000)

    def test_non_numeric_port_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            ServerConfig.from_env({"ANTIDAPTER_PORT": "http"})

    def test_interactive_login_toggle(self):
        self.assertFalse(
            ServerConfig.from_env(
                {"ANTIDAPTER_ALLOW_INTERACTIVE_LOGIN": "false"}
            ).allow_interactive_login
        )
        self.assertTrue(
            ServerConfig.from_env(
                {"ANTIDAPTER_ALLOW_INTERACTIVE_LOGIN": "1"}
            ).allow_interactive_login
        )

    def test_invalid_body_limit_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            ServerConfig(max_request_bytes=0)


class TestStorageConfig(unittest.TestCase):
    def test_file_path_is_joined(self):
        config = StorageConfig(config_dir="/tmp/x")
        self.assertEqual(config.file_path, "/tmp/x/credentials.json")

    def test_default_dir(self):
        self.assertTrue(StorageConfig.from_env({}).config_dir.endswith("antidapter"))


class TestProtocolConfig(unittest.TestCase):
    def test_default_model_is_defined_once(self):
        """Regression: the default model was duplicated in two modules."""
        self.assertEqual(ProtocolConfig().default_model, "gemini-3.6-flash-low")
        import adapters.inbound.http.openai_adapter as module

        source = Path(module.__file__).read_text(encoding="utf-8")
        self.assertNotIn("gemini-3.6-flash-low", source)

    def test_overridable(self):
        self.assertEqual(
            ProtocolConfig.from_env({"ANTIDAPTER_DEFAULT_MODEL": "m"}).default_model, "m"
        )


class TestAppConfig(unittest.TestCase):
    def test_builds_from_minimal_env(self):
        config = AppConfig.from_env(MINIMAL)
        self.assertIsInstance(config.oauth, OAuthConfig)
        self.assertIsInstance(config.upstream, UpstreamConfig)
        self.assertIsInstance(config.server, ServerConfig)
        self.assertIsInstance(config.storage, StorageConfig)
        self.assertIsInstance(config.protocol, ProtocolConfig)

    def test_is_immutable(self):
        config = AppConfig.from_env(MINIMAL)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            config.log_level = "DEBUG"  # type: ignore[misc]

    def test_config_dir_is_redirectable_for_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig.from_env({**MINIMAL, "ANTIDAPTER_CONFIG_DIR": tmp})
            self.assertEqual(config.storage.config_dir, tmp)


class TestLoadDotenv(unittest.TestCase):
    def test_reads_key_values_without_overriding_real_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(
                "# comment\nANTIDAPTER_TEST_A=1\nANTIDAPTER_TEST_B='quoted'\nBROKEN\n",
                encoding="utf-8",
            )
            os.environ["ANTIDAPTER_TEST_A"] = "preset"
            os.environ.pop("ANTIDAPTER_TEST_B", None)
            try:
                load_dotenv(path)
                self.assertEqual(os.environ["ANTIDAPTER_TEST_A"], "preset")
                self.assertEqual(os.environ["ANTIDAPTER_TEST_B"], "quoted")
            finally:
                os.environ.pop("ANTIDAPTER_TEST_A", None)
                os.environ.pop("ANTIDAPTER_TEST_B", None)

    def test_missing_file_is_ignored(self):
        load_dotenv(Path("/nonexistent/.env"))


if __name__ == "__main__":
    unittest.main()
