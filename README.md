# Antidapter

An OpenAI-compatible gateway that exposes Google Antigravity (Cloud Code) models
to any agent harness — Pi, Claude Code, Cline, Aider, and anything else that
speaks the OpenAI Chat Completions API.

No `agy` binary required. No IDE required. No injected system prompts.

---

## Features

- **OpenAI-compatible** `/v1/chat/completions` with SSE streaming and
  `stream_options.include_usage`, plus `/v1/models` and `/v1/models/{id}`.
- **Tool calling** — `tools`, assistant `tool_calls` and `role: "tool"` results
  are round-tripped to the upstream `functionCall` / `functionResponse` parts.
- **Vision** — base64 `data:` image URLs.
- **Reasoning** — `thought` parts are surfaced as `reasoning_content` rather
  than being concatenated into the visible answer.
- **Credential discovery** — reuses an existing `agy` login from the OS keyring
  when present, otherwise maintains its own `0600` credentials file.
- **Resilience** — bounded request timeouts, exponential backoff with jitter-free
  retry on `429`/`5xx`, and fail-fast credential handling.
- **Zero third-party Python dependencies** — standard library only.

---

## Setup

### 1. Credentials

Antidapter needs a Google OAuth 2.0 **Desktop / Installed app** client with the
Cloud Code (`aicode`) scope enabled. These are **not** an API key.

```bash
cp .env.example .env
$EDITOR .env      # set ANTIDAPTER_CLIENT_ID and ANTIDAPTER_CLIENT_SECRET
```

`.env` is gitignored. The app refuses to start without these values rather than
falling back to a bundled default.

### 2. Authenticate

```bash
python3 main.py login
```

This opens a browser, waits up to 180 seconds, and stores the resulting refresh
token in `~/.config/antidapter/credentials.json` with `0600` permissions.

If you have already logged in with `agy`, Antidapter will discover and reuse
those credentials automatically — `login` is then optional.

### 3. Inspect

```bash
python3 main.py models     # catalog with capability flags
python3 main.py quota      # live remaining quota per model
python3 main.py model gemini-3.6-flash-low
```

### 4. Serve

```bash
python3 main.py serve --port 8080
```

---

## Connecting Pi (`pi`)

Two integration paths. Both avoid the `openai` provider, so your built-in
providers are never shadowed.

### Recommended: the `agy` provider extension

A pi extension in `pi/agy-provider/` registers a provider named `agy` whose model
list is read from the running gateway at startup. The catalog changes whenever
Google rotates models, so a static entry would go stale silently — this one
always matches `python3 main.py models`.

```bash
pi install ./pi/agy-provider     # global; add -l for project-local

pi --provider agy --model gemini-3.6-flash-low
pi --provider agy --list-models
```

If the gateway is not running, the extension falls back to a single seed model
and warns, rather than failing to load. Configure with `ANTIDAPTER_BASE_URL`
(default `http://127.0.0.1:8080/v1`) and `ANTIDAPTER_API_KEY`.

### Alternative: a static `models.json` entry

```bash
python3 main.py pi-config            # print the provider block
python3 main.py pi-config --write    # merge into ~/.pi/agent/models.json

pi --provider antidapter --model gemini-3.6-flash-low
```

Re-run `--write` whenever the catalog changes. If you enable
`ANTIDAPTER_API_KEY`, re-run it so the key is baked in.

### Do not use `--provider openai`

- Pi's built-in `openai` provider points at `platform.openai.com` and **ignores
  `OPENAI_BASE_URL` entirely**, so this route does not work — it authenticates
  against the real OpenAI API and returns `401`.
- Redirecting the `openai` slot would shadow that provider for every other
  session on the machine.

---

## Connecting other harnesses

Anything that accepts a custom OpenAI base URL works:

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-3.6-flash-low","messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```

A harness that only understands the `openai` provider name (rather than a
configurable base URL) is the one case this cannot serve without shadowing that
provider.

---

## Security

The gateway holds a Google access token on your behalf, so treat the port as
privileged:

- It binds to `127.0.0.1` by default. **If you bind to `0.0.0.0`, set
  `ANTIDAPTER_API_KEY`** — callers must then send
  `Authorization: Bearer <key>`. The server prints a warning if you do not.
- CORS is permissive (`*`) for browser-based clients. It is not an access
  control; the API key is.
- Request bodies are capped at `ANTIDAPTER_MAX_REQUEST_BYTES` (32 MiB default).
- Remote `image_url` values are rejected: fetching them would make the gateway
  an SSRF proxy. Only base64 `data:` URLs are accepted.

---

## Configuration

Every setting is environment-driven. See `.env.example` for the full list.

| Variable | Default | Purpose |
| --- | --- | --- |
| `ANTIDAPTER_CLIENT_ID` | *required* | OAuth client id |
| `ANTIDAPTER_CLIENT_SECRET` | *required* | OAuth client secret |
| `ANTIDAPTER_HOST` / `ANTIDAPTER_PORT` | `127.0.0.1` / `8080` | Bind address |
| `ANTIDAPTER_API_KEY` | *(unset)* | Require a bearer token inbound |
| `ANTIDAPTER_ALLOW_INTERACTIVE_LOGIN` | `true` | Set `false` so requests never block on a browser login |
| `ANTIDAPTER_UPSTREAM_TIMEOUT` | `300` | Per-request upstream timeout (s) |
| `ANTIDAPTER_MAX_RETRIES` | `3` | Retries for `429`/`5xx` before failing |
| `ANTIDAPTER_RETRY_BASE_DELAY` / `_MAX_DELAY` | `0.5` / `8` | Exponential backoff bounds (s) |
| `ANTIDAPTER_MAX_REQUEST_BYTES` | `33554432` | Inbound body cap |
| `ANTIDAPTER_CONFIG_DIR` | `~/.config/antidapter` | Credential location |
| `ANTIDAPTER_DEFAULT_MODEL` | `gemini-3.6-flash-low` | Used when the request omits `model` |
| `ANTIDAPTER_LOG_LEVEL` | `INFO` | Root log level |

---

## Development

```bash
python3 -m unittest discover tests   # 198 tests, no network
uvx ruff check . && uvx ruff format --check .
uvx --with types-setuptools mypy .
```

The test suite is fully mocked: no test performs a live network call.

See `FEATURE_MAP.md` for the module-by-module capability map and
`AGENTS.md` for the architectural rules this codebase is held to.

---

## Architecture

```
main.py            CLI entrypoint: loads .env, configures logging
container.py       composition root — the only module that knows every adapter
config.py          frozen dataclasses loaded from the environment
core/
  domain/          entities + exception hierarchy (no I/O, no framework)
  ports/           inbound and outbound ABCs
  services/        use-case implementations
adapters/
  inbound/         HTTP transport, OpenAI protocol translator, CLI
  outbound/        Google Cloud Code, Google OAuth, file + keyring storage
```

Dependencies point inward only. `core/` imports nothing from `adapters/`, no
`urllib`, and no `http.server`; all of that is injected through ports and wired
in `container.py`.
