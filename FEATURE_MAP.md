# ANTIDAPTER FEATURE & CAPABILITY MAP

Agent-facing description of what actually exists in this repository. Every entry
below is covered by tests; anything not listed is not implemented.

```yaml
system:
  name: antidapter
  architecture: hexagonal_ports_and_adapters
  runtime: python_3_10_plus_stdlib_only
  composition_root: container.py:Container
  entrypoint: main.py:main
  tests: 198 unit tests, zero live network calls

invariants:
  layer_isolation: >
    core/ imports nothing from adapters/ and nothing from urllib, http.server,
    json or webbrowser. Verified by review and by mypy's module graph.
  dependency_inversion: >
    Services depend on core/ports ABCs only. container.py is the sole module
    that names concrete adapters.
  no_third_party_dependencies: >
    Standard library only. The single optional import is PyGObject (gi) for OS
    keyring discovery; it is imported lazily and its absence is not fatal.
  no_embedded_secrets: >
    OAuth credentials come from the environment only. tests/test_config.py
    asserts no credential literal exists in config.py.
  immutable_domain: >
    Domain entities and configuration objects are frozen dataclasses.
  eager_preflight: >
    complete_stream authenticates before returning its iterator, so the HTTP
    layer can still map pre-flight failures to real status codes.

features:

  # =========================================================================
  # INBOUND (DRIVING) ADAPTERS
  # =========================================================================
  - id: inbound.http.server
    status: implemented
    module: adapters/inbound/http/server.py:GatewayHandler
    responsibility: >
      HTTP transport only: routing, bearer auth, body limits, status mapping,
      SSE framing. Holds no protocol knowledge.
    routes:
      - { method: GET,  path: /health,               auth: false }
      - { method: GET,  path: /v1/models,            auth: true  }
      - { method: GET,  path: /v1/models/{model_id}, auth: true  }
      - { method: GET,  path: /v1/gateway/models,  auth: true,
        note: "capability-rich catalog for clients that need context windows" }
      - { method: POST, path: /v1/chat/completions,  auth: true  }
      - { method: GET,  path: /models,               auth: true  }
      - { method: POST, path: /chat/completions,     auth: true  }
    behaviours:
      - "Query strings are ignored when matching (e.g. /v1/models?limit=1)."
      - "Unknown path -> 404, wrong method -> 405, both as JSON error bodies."
      - "OPTIONS preflight returns 204 with CORS headers."
      - "Internal errors return a generic message; tracebacks go to the log only."
      - "A client that hangs up mid-response is logged as a normal disconnect,
         not reported as a 500."
      - "SSE responses are delimited by connection close. Claiming keep-alive
         would deadlock the client against the server, so Connection: close is
         sent deliberately."
    tests: tests/test_server.py

  - id: inbound.http.openai_translator
    status: implemented
    module: adapters/inbound/http/openai_adapter.py:OpenAIProtocolTranslator
    implements: core.ports.inbound.ProtocolTranslatorPort
    capabilities:
      - "Messages: system, developer, user, assistant, tool, function."
      - "Content: plain strings and text/image_url blocks."
      - "Tool definitions and tool_calls / tool_call_id round-tripping."
      - "tools, temperature, top_p, max_tokens / max_completion_tokens."
      - "stream_options.include_usage emits a trailing usage-only chunk with
         an empty choices array."
      - "reasoning_content emitted for thought parts."
      - "GET /v1/models and single-model serialization."
    deliberate_rejections:
      - "Unknown roles: 400, never silently coerced to `user`."
      - "Remote image_url values: 400, to avoid SSRF."
      - "Malformed base64 data URLs and unsupported block types: 400, never
         silently dropped."
    tests: tests/test_openai_adapter.py

  - id: inbound.cli
    status: implemented
    module: adapters/inbound/cli/cli_runner.py:CliRunner
    commands:
      - { name: serve,     args: [--host, --port], note: "authenticates before binding" }
      - { name: login,     note: "prints the auth URL, then forces a browser authorization" }
      - { name: logout,    note: "clears cached and persisted credentials" }
      - { name: models,    note: "sorted catalog with thinking/tools flags" }
      - { name: quota,     note: "live remaining fraction and reset time" }
      - { name: model,     args: [model_id] }
      - { name: pi-config, args: [--write, --pi-models-file],
        note: "renders the pi provider block from the live catalog" }
    exit_codes: { 0: success, 2: domain or configuration error, 130: interrupted }
    tests: tests/test_cli.py

  - id: inbound.pi_provider
    status: implemented
    module: adapters/inbound/pi/pi_provider.py
    purpose: >
      Pi keeps custom providers in ~/.pi/agent/models.json. Antidapter is
      exposed under its own provider name so pi's built-in providers are never
      shadowed. The block is generated from the live catalog rather than
      hand-maintained, so model lists cannot drift.
    merge_safety: >
      Other providers in the file are preserved, malformed files are refused
      rather than clobbered, and the write is atomic.
    tests: tests/test_pi_provider.py

  - id: inbound.pi_provider
    status: implemented
    module: adapters/inbound/pi/pi_provider.py
    purpose: >
      Renders the pi provider block from the live catalog, and merges it into
      ~/.pi/agent/models.json. Other providers are preserved, malformed files
      are refused rather than clobbered, and the write is atomic.
    tests: tests/test_pi_provider.py

  - id: integration.pi_agy_extension
    status: implemented
    module: pi/agy-provider/index.js
    language: JavaScript (pi extension, not part of the Python package)
    purpose: >
      Registers a first-class `agy` provider with pi via pi.registerProvider.
      The model list is fetched from GET /v1/gateway/models at module load, so
      `pi --list-models` (which exits before session_start fires) is correct,
      and re-probed on session_start to pick up a gateway started after pi.
    degradation: >
      With the gateway down the extension loads with a single seed model and
      warns, bounded by a 2s startup probe, so pi startup is never blocked.
    install: "pi install git:github.com/vang-trang-co-don/antidapter"
    packaging: >
      Published from this repository rather than a separate one, because the
      extension depends on the gateway's /v1/gateway/models endpoint. pi's git
      source syntax has no subdirectory support, so the repository root carries
      a package.json whose pi.extensions field points at the extension.

  - id: inbound.anthropic_messages
    status: not_implemented
    note: >
      Feasible without touching the domain or the transport: implement
      ProtocolTranslatorPort and add a route. Deliberately not built.

  # =========================================================================
  # CORE DOMAIN & USE CASES
  # =========================================================================
  - id: core.chat_service
    status: implemented
    module: core/services/chat_service.py:ChatService
    implements: core.ports.inbound.ChatUseCase
    depends_on: [core.ports.inbound.AuthUseCase, core.ports.outbound.UpstreamModelPort]
    contracts:
      complete: { in: ChatRequest, out: ChatResponse }
      complete_stream: { in: ChatRequest, out: Iterator[StreamDelta], note: "eager auth" }
    behaviours:
      - "Request validation is not duplicated: the frozen ChatRequest entity
         enforces its own invariants in __post_init__."
      - "Cumulative usage from the upstream takes the last reported value
         rather than being summed."
      - "usage is None when the upstream reports none, so no all-zero usage
         block is fabricated."
      - "finish_reason becomes tool_calls when tool calls were returned."
    tests: tests/test_chat_service.py

  - id: core.auth_service
    status: implemented
    module: core/services/auth_service.py:AuthService
    implements: core.ports.inbound.AuthUseCase
    depends_on: [core.ports.outbound.TokenStoragePort, core.ports.outbound.OAuthProviderPort]
    concurrency:
      - "_lock guards only cheap work (cache probe, storage read, refresh)."
      - "_login_lock guards interactive authorization and is never held
         together with _lock, so one browser flow cannot stall other callers."
      - "The cache is re-checked under _login_lock, so exactly one browser
         flow runs even when many threads arrive at once."
    policy:
      - "A failed refresh falls back to re-authorization rather than failing
         terminally; a transient upstream blip cannot brick the gateway."
      - "allow_interactive=False makes missing or expired credentials raise
         immediately, so a request thread never blocks for a browser flow."
    tests: tests/test_auth_service.py

  - id: core.model_catalog_service
    status: implemented
    module: core/services/model_catalog_service.py:ModelCatalogService
    implements: core.ports.inbound.ModelCatalogUseCase
    returns: tuple[ModelInfo, ...]
    raises: ModelNotFoundError

  - id: core.domain
    status: implemented
    module: core/domain/{entities,exceptions}.py
    entities:
      - Role, FinishReason
      - TextPart, ImagePart, ToolCallPart, ToolResultPart
      - ChatMessage, GenerationParameters, ToolDefinition, ChatRequest
      - TokenUsage, ToolCall, StreamDelta, ChatResponse
      - QuotaInfo, ModelInfo, AuthToken
    invariants:
      - "All validation raises ValidationError, which also subclasses
         ValueError for compatibility."
      - "GenerationParameters rejects non-numeric, non-finite and
         out-of-range values instead of forwarding them upstream."
      - "ChatMessage.text_content joins text blocks with a newline, preserving
         paragraph boundaries."
      - "AuthToken.to_dict / from_dict centralize token serialization and
         accept the legacy `expiry` key."
    tests: tests/test_domain.py

  # =========================================================================
  # OUTBOUND (DRIVEN) ADAPTERS
  # =========================================================================
  - id: outbound.google_cloudcode
    status: implemented
    module: adapters/outbound/upstream/google_cloudcode.py:GoogleCloudCodeAdapter
    implements: core.ports.outbound.UpstreamModelPort
    endpoints:
      stream_generate: "POST {base}/v1internal:streamGenerateContent?alt=sse"
      fetch_models:   "POST {base}/v1internal:fetchAvailableModels"
    resilience:
      - "Every request has a bounded timeout (ANTIDAPTER_UPSTREAM_TIMEOUT)."
      - "429/408/425/5xx and transport errors are retried with exponential
         backoff bounded by ANTIDAPTER_RETRY_MAX_DELAY."
      - "Streaming is retried only until the first chunk is committed; once
         deltas are handed to the caller the stream is never replayed."
      - "The model catalog is cached for ANTIDAPTER_CATALOG_TTL seconds, so
         startup probes and model pickers do not each cost a round trip to
         Google. A failed read is never cached."
      - "All urllib failures are translated to domain exceptions, so callers
         never depend on the transport."
    sse:
      - "Understands `data:` payloads, ignores comments, `event:`, `id:` and
         `retry:` lines."
      - "Terminates cleanly on the `data: [DONE]` sentinel."
      - "Malformed chunks are logged and skipped, not fatal."
    mapping:
      request: "ChatRequest -> {project, model, request: {contents, systemInstruction,
                generationConfig, tools}}; assistant -> model, tool -> functionResponse,
                tool_calls -> functionCall."
      response: "candidates.parts -> text / reasoning (thought:true) / tool calls;
                 finishReason -> OpenAI stop reason; usageMetadata -> TokenUsage."
    tests: tests/test_upstream_adapter.py

  - id: outbound.google_oauth
    status: implemented
    module: adapters/outbound/oauth/google_oauth.py:GoogleOAuthAdapter
    implements: core.ports.outbound.OAuthProviderPort
    flow:
      - "Loopback listener on an ephemeral 127.0.0.1 port."
      - "A random `state` is sent and verified on callback (CSRF protection)."
      - "The wait is bounded by an explicit deadline, so cancelling the consent
         screen fails fast instead of spinning forever."
      - "error=access_denied is surfaced as AuthenticationError."
      - "A missing browser opener is non-fatal; the URL is logged."
    notes:
      - "Refresh responses omit refresh_token, so the existing one is retained."
      - "HTTP error bodies are read and closed to avoid leaking connections."
    tests: covered indirectly via AuthService doubles; the callback handler is
           exercised by tests/test_auth_service.py's single-flight cases.

  - id: outbound.file_storage
    status: implemented
    module: adapters/outbound/storage/file_storage.py:FileTokenStorage
    implements: core.ports.outbound.TokenStoragePort
    guarantees:
      - "Credentials are written 0600 inside a 0700 directory."
      - "Writes are atomic (temp file + rename), so a crash cannot leave
         truncated JSON."
      - "Unreadable or corrupt files are ignored, not fatal."
      - "Failures raise DomainException rather than a bare OSError."
    tests: tests/test_storage.py

  - id: outbound.secret_service
    status: implemented_read_only
    module: adapters/outbound/storage/secret_service.py:SecretServiceCredentialSource
    implements: core.ports.outbound.TokenSourcePort   # deliberately NOT TokenStoragePort
    rationale: >
      The keyring entries belong to another application. Implementing only the
      read-only source port means there is no way to clobber them; anything
      found is promoted to Antidapter's own storage by the chaining adapter.
    notes:
      - "PyGObject is imported lazily; absence degrades to 'no credentials'."
      - "Understands both the nested agy shape (token.expiry) and a flat shape."
    tests: tests/test_storage.py

  - id: outbound.chained_storage
    status: implemented
    module: adapters/outbound/storage/composite_storage.py:ChainedTokenStorage
    strategy: >
      Reads the sink first, then each source in order. A token found in a
      fallback is promoted to the sink. A source that raises is logged and
      skipped rather than breaking discovery. clear() touches every writable
      store.
    tests: tests/test_storage.py
```

---

## Known limitations

- No Anthropic `/v1/messages` endpoint.
- `streamGenerateContent` is SSE only; there is no non-streaming upstream mode
  (the gateway synthesizes one by consuming the stream).
- Multimodal input supports images only; audio and file parts are rejected
  with a 400 rather than silently ignored.
- Thinking/reasoning output is passed through verbatim; there is no
  reasoning-budget control.
- Retry backoff is deterministic (no jitter), so concurrent clients can
  synchronize on retries.
- Pi's built-in `openai` provider ignores `OPENAI_BASE_URL`, so harnesses that
  can only be pointed at a provider *name* (rather than a configurable base
  URL) cannot be served without shadowing that provider. The `antidapter`
  provider is the supported path.
- The upstream `inputTokenLimit` / `maxOutputTokens` field names are assumed;
  they are mapped defensively with defaults if absent.
