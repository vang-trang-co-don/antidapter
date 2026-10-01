import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from core.domain.exceptions import ConfigurationError

_TRUE = frozenset({"1", "true", "yes", "on"})


def load_dotenv(path: Path | None = None) -> None:
    """Populate os.environ from a .env file without overriding real env vars.

    Keeps secrets out of version control while avoiding a third-party
    dependency. Missing or unreadable files are ignored.
    """
    env_path = path or (Path(__file__).resolve().parent / ".env")
    try:
        content = env_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_str(env: Mapping[str, str], key: str, default: str | None = None) -> str:
    value = env.get(key, default)
    if value is None:
        raise ConfigurationError(f"Missing required environment variable: {key}")
    return value


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{key} must be an integer, got {raw!r}") from exc


def _env_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{key} must be a number, got {raw!r}") from exc


def _env_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in _TRUE


@dataclass(frozen=True)
class OAuthConfig:
    client_id: str
    client_secret: str
    auth_uri: str = "https://accounts.google.com/o/oauth2/v2/auth"
    token_uri: str = "https://oauth2.googleapis.com/token"
    redirect_path: str = "/oauth2callback"
    login_timeout: float = 180.0
    request_timeout: float = 30.0
    scopes: tuple[str, ...] = (
        "openid",
        "email",
        "profile",
        "https://www.googleapis.com/auth/aicode",
        "https://www.googleapis.com/auth/cloud-platform",
    )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "OAuthConfig":
        env = os.environ if env is None else env
        return cls(
            client_id=_env_str(env, "ANTIDAPTER_CLIENT_ID"),
            client_secret=_env_str(env, "ANTIDAPTER_CLIENT_SECRET"),
            auth_uri=_env_str(env, "ANTIDAPTER_AUTH_URI", cls.auth_uri),
            token_uri=_env_str(env, "ANTIDAPTER_TOKEN_URI", cls.token_uri),
            redirect_path=_env_str(env, "ANTIDAPTER_REDIRECT_PATH", cls.redirect_path),
            login_timeout=_env_float(env, "ANTIDAPTER_LOGIN_TIMEOUT", cls.login_timeout),
            request_timeout=_env_float(env, "ANTIDAPTER_OAUTH_TIMEOUT", cls.request_timeout),
        )


@dataclass(frozen=True)
class UpstreamConfig:
    base_url: str = "https://daily-cloudcode-pa.googleapis.com"
    project_id: str = "aicode-consumers"
    user_agent: str = "antigravity"
    stream_generate_path: str = "/v1internal:streamGenerateContent?alt=sse"
    fetch_models_path: str = "/v1internal:fetchAvailableModels"
    request_timeout: float = 300.0
    catalog_ttl: float = 300.0
    max_retries: int = 3
    retry_base_delay: float = 0.5
    retry_max_delay: float = 8.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "UpstreamConfig":
        env = os.environ if env is None else env
        return cls(
            base_url=_env_str(env, "ANTIDAPTER_UPSTREAM_URL", cls.base_url).rstrip("/"),
            project_id=_env_str(env, "ANTIDAPTER_PROJECT_ID", cls.project_id),
            user_agent=_env_str(env, "ANTIDAPTER_USER_AGENT", cls.user_agent),
            request_timeout=_env_float(env, "ANTIDAPTER_UPSTREAM_TIMEOUT", cls.request_timeout),
            catalog_ttl=_env_float(env, "ANTIDAPTER_CATALOG_TTL", cls.catalog_ttl),
            max_retries=_env_int(env, "ANTIDAPTER_MAX_RETRIES", cls.max_retries),
            retry_base_delay=_env_float(env, "ANTIDAPTER_RETRY_BASE_DELAY", cls.retry_base_delay),
            retry_max_delay=_env_float(env, "ANTIDAPTER_RETRY_MAX_DELAY", cls.retry_max_delay),
        )


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8080
    api_key: str | None = None
    max_request_bytes: int = 32 * 1024 * 1024
    allow_interactive_login: bool = True

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ServerConfig":
        env = os.environ if env is None else env
        api_key = env.get("ANTIDAPTER_API_KEY") or None
        return cls(
            host=_env_str(env, "ANTIDAPTER_HOST", cls.host),
            port=_env_int(env, "ANTIDAPTER_PORT", cls.port),
            api_key=api_key,
            max_request_bytes=_env_int(env, "ANTIDAPTER_MAX_REQUEST_BYTES", cls.max_request_bytes),
            allow_interactive_login=_env_bool(
                env, "ANTIDAPTER_ALLOW_INTERACTIVE_LOGIN", cls.allow_interactive_login
            ),
        )

    def __post_init__(self) -> None:
        if not 0 < self.port < 65536:
            raise ConfigurationError(f"ANTIDAPTER_PORT out of range: {self.port}")
        if self.max_request_bytes <= 0:
            raise ConfigurationError("ANTIDAPTER_MAX_REQUEST_BYTES must be positive")


@dataclass(frozen=True)
class StorageConfig:
    config_dir: str
    filename: str = "credentials.json"
    keyring_service: str = "gemini"
    keyring_username: str = "antigravity"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "StorageConfig":
        env = os.environ if env is None else env
        return cls(
            config_dir=_env_str(
                env,
                "ANTIDAPTER_CONFIG_DIR",
                os.path.expanduser("~/.config/antidapter"),
            ),
            filename=_env_str(env, "ANTIDAPTER_CREDENTIALS_FILE", cls.filename),
            keyring_service=_env_str(env, "ANTIDAPTER_KEYRING_SERVICE", cls.keyring_service),
            keyring_username=_env_str(env, "ANTIDAPTER_KEYRING_USERNAME", cls.keyring_username),
        )

    @property
    def file_path(self) -> str:
        return os.path.join(self.config_dir, self.filename)


@dataclass(frozen=True)
class ProtocolConfig:
    """Settings owned by the wire-protocol adapters."""

    default_model: str = "gemini-3.6-flash-low"
    emit_reasoning: bool = True

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ProtocolConfig":
        env = os.environ if env is None else env
        return cls(
            default_model=_env_str(env, "ANTIDAPTER_DEFAULT_MODEL", cls.default_model),
            emit_reasoning=_env_bool(env, "ANTIDAPTER_EMIT_REASONING", cls.emit_reasoning),
        )


@dataclass(frozen=True)
class AppConfig:
    oauth: OAuthConfig
    upstream: UpstreamConfig
    server: ServerConfig
    storage: StorageConfig
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "AppConfig":
        env = os.environ if env is None else env
        return cls(
            oauth=OAuthConfig.from_env(env),
            upstream=UpstreamConfig.from_env(env),
            server=ServerConfig.from_env(env),
            storage=StorageConfig.from_env(env),
            protocol=ProtocolConfig.from_env(env),
            log_level=_env_str(env, "ANTIDAPTER_LOG_LEVEL", cls.log_level).upper(),
        )
