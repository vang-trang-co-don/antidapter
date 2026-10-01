# AGENTS.md - Antidapter Development Guidelines

This document provides instructions, coding conventions, architectural boundaries, and workflows for autonomous AI coding agents collaborating on the **Antidapter** codebase.

---

## 1. Project Philosophy & Core Principles

Antidapter is an unopinionated model gateway that bridges Google Antigravity models directly to external agent harnesses (such as Pi, Claude Code, Aider, Cline).

When modifying this repository, agents MUST uphold these principles:

1. **Clean Architecture (Hexagonal / Ports & Adapters):**
   * Keep the **Core Domain** (`core/domain/`, `core/services/`) completely free from infrastructure details (no `urllib`, no `http.server`, no Google-specific JSON payload keys, no external frameworks).
   * All outbound calls MUST go through abstract ports (`core/ports/outbound.py`).
   * All inbound requests MUST enter through abstract use cases (`core/ports/inbound.py`).
2. **SOLID Principles:**
   * **Single Responsibility:** Keep handlers, services, and adapters small and focused.
   * **Open/Closed:** Add new adapters (e.g. Anthropic adapter, new storage backend) without modifying domain services.
   * **Dependency Inversion:** Depend on interfaces/ports, never on concrete implementations.
3. **Dependency Injection (DI):**
   * Never instantiate dependencies directly inside business services or handlers.
   * Wire all components inside `container.py` (Composition Root).
4. **No Hardcoding:**
   * Never hardcode client secrets, IDs, endpoints, or port numbers in business logic.
   * Define configuration options in `config.py` with environment variable overrides and sensible defaults.
5. **Zero External Dependencies:**
   * The codebase strictly uses the Python 3 standard library (`http.server`, `urllib`, `dataclasses`, `argparse`). Do not introduce third-party package dependencies (e.g. `requests`, `fastapi`, `pydantic`) without explicit user instruction.
6. **Explicit Validation & Error Handling:**
   * Raise custom domain exceptions from `core/domain/exceptions.py`.
   * Translate domain exceptions into standard HTTP error responses in the inbound HTTP adapters.

---

## 2. Directory Structure

```
antidapter/
├── config.py                       # Typed Dataclasses loaded via environment variables
├── container.py                    # Composition Root (Dependency Injection wiring)
├── main.py                         # Application CLI entrypoint
│
├── core/                           # Pure Domain Layer
│   ├── domain/
│   │   ├── entities.py             # Domain entities and value objects
│   │   └── exceptions.py           # Domain exceptions hierarchy
│   ├── ports/
│   │   ├── inbound.py              # Inbound Ports (Use Cases)
│   │   └── outbound.py             # Outbound Ports (Driven Interfaces)
│   └── services/
│       ├── chat_service.py         # Implements ChatUseCase
│       ├── model_catalog_service.py# Implements ModelCatalogUseCase
│       └── auth_service.py         # Implements AuthUseCase (thread-safe token manager)
│
├── adapters/                       # Adapters Layer
│   ├── inbound/                    # Driving Adapters
│   │   ├── cli/                    # CliRunner (serve, login, logout, models, quota, model)
│   │   └── http/
│   │       ├── server.py           # HTTP transport only: routing, auth, framing
│   │       └── openai_adapter.py   # OpenAIProtocolTranslator (implements the port)
│   └── outbound/                   # Driven Adapters
│       ├── oauth/                  # GoogleOAuthAdapter (loopback flow + refresh)
│       ├── storage/                # FileTokenStorage, SecretServiceCredentialSource,
│       │                           #   ChainedTokenStorage
│       └── upstream/               # GoogleCloudCodeAdapter (SSE + retry/backoff)
│
├── tests/                          # 225 mocked unit tests (0 live network calls)
├── pyproject.toml                  # packaging + ruff + mypy configuration
└── .env.example                    # template for required credentials
```

---

## 3. Testing & Verification Commands

Before proposing or completing any task, agents **MUST** execute the test suite and verify that all tests pass:

```bash
# 1. Run all unit tests (no network access)
python3 -m unittest discover tests

# 2. Lint and format
uvx ruff check .
uvx ruff format --check .

# 3. Strict type check
uvx --with types-setuptools mypy .

# 4. Live checks (these DO hit the network; run manually, never in CI unit runs)
python3 main.py models
python3 main.py quota
```

*Note: All unit tests in `tests/` MUST be isolated and mock the outbound ports. Do NOT make live network calls during unit test runs. `tests/test_layering.py` enforces the architectural rules below and will fail on a boundary violation.*

---

## 4. Coding Conventions

- **Typing:** Use strict Python 3.10+ type hints everywhere (`dataclass(frozen=True)`, `X | None`, `collections.abc` generics). `mypy` runs with `disallow_untyped_defs`.
- **Immutability:** Domain entities and configuration objects must be `frozen=True`. Return `tuple`, not `list`, from ports.
- **Imports:** Use absolute imports from root (`from core.domain.entities import ...`). Relative imports are forbidden and enforced by a test.
- **Logging:** Use `logging.getLogger(__name__)`. `print` is allowed only in the CLI's user-facing output and the startup banner.
- **Errors:** Raise `ValidationError` (never bare `ValueError`) from the domain, and `UpstreamServiceError` with an accurate `retryable` flag from outbound adapters. Never let `urllib` exceptions escape an adapter.
- **Secrets:** Never commit credentials. OAuth client id/secret come from the environment; `.env` is gitignored and a test enforces this.
- **Streaming:** `ChatUseCase.complete_stream` MUST authenticate eagerly (return an iterator, never be a generator function), so the HTTP layer can still send real status codes. Mid-stream failures are reported as in-band SSE `error` events, never as a second HTTP response.

---

## 5. Known Invariants Worth Preserving

These are covered by tests; breaking one is a regression, not a refactor:

- **SSE framing.** Streaming responses are delimited by connection close. Sending `Connection: keep-alive` without `Content-Length` or chunked encoding deadlocks the client against the server, because `http.server` then waits for another request.
- **Locking.** `AuthService` holds `_lock` only for cheap work. Interactive authorization runs under a separate `_login_lock` and is never nested inside `_lock`.
- **Read-only discovery.** The OS keyring source implements `TokenSourcePort`, not `TokenStoragePort`, so it is structurally incapable of clobbering another application's credentials.
- **No SSRF.** Remote `image_url` values are rejected; only base64 `data:` URLs are accepted.
