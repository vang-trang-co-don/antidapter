/**
 * pi extension: registers an `agy` provider backed by Antidapter.
 *
 * You should never have to run a server by hand. The flow is:
 *
 *   pi  ->  /login  ->  "Sign in with Google (Antigravity)"  ->  browser
 *       ->  /model  ->  pick an agy model
 *
 * This extension owns the whole lifecycle:
 *
 *   - `/login` runs the bundled gateway's own `login` command, relays the
 *     authorization URL to pi (which displays it and opens a browser), and
 *     waits for the loopback callback to complete.
 *   - The gateway is spawned lazily on an ephemeral port, so two pi sessions
 *     (or a manually started gateway) can never collide.
 *   - The model list is read from the running gateway, so it always matches
 *     what upstream currently offers. Google rotates models; a static list
 *     would go stale silently.
 *   - The child is killed on `session_shutdown`.
 *
 * Design notes:
 *
 * - The Python engine is located relative to this file, not via a config path,
 *   so the install keeps working wherever pi clones the package.
 * - The child is always started with interactive login DISABLED. That is what
 *   guarantees a browser can only ever be opened by `/login`, never
 *   spontaneously by a background process. `antidapter serve` honours this by
 *   skipping its startup credential check.
 * - pi's credential store keeps only a marker plus the token expiry. The actual
 *   refresh token stays in the gateway's own 0600 credentials file, so there is
 *   exactly one copy of a long-lived secret.
 *
 * Configuration (all optional):
 *   ANTIDAPTER_PYTHON          python interpreter (default: python3)
 *   ANTIDAPTER_HOME            override the gateway checkout
 *   ANTIDAPTER_BASE_URL        use an already-running gateway instead of spawning
 *   ANTIDAPTER_API_KEY         forwarded to the child; also the provider api key
 */

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const PROVIDER_ID = "agy";
const PROVIDER_NAME = "Antigravity (Google)";
const LOGIN_LABEL = "Sign in with Google (Antigravity)";

const DEFAULT_PYTHON = "python3";
const PROBE_TIMEOUT_MS = 1_500;
const HEALTH_TIMEOUT_MS = 20_000;
const LOGIN_TIMEOUT_MS = 300_000;
const GATEWAY_SHUTDOWN_GRACE_MS = 1_500;
/**
 * Orphan safety net: if pi is SIGKILLed the child cannot be signalled, so the
 * gateway retires itself after a long idle period. Generous, because a real
 * session can sit idle while the user thinks.
 */
const ORPHAN_IDLE_TIMEOUT_S = 900;

/** Shown until the gateway reports a real catalog. */
const SEED_MODELS = [
  {
    id: "gemini-3.6-flash-low",
    name: "Gemini 3.6 Flash (Low)",
    reasoning: true,
    input: ["text", "image"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 1_048_576,
    maxTokens: 65_536,
  },
];

// ---------------------------------------------------------------------------
// Locating and running the gateway
// ---------------------------------------------------------------------------

/** The checkout root: this file lives at <root>/pi/agy-provider/index.js. */
function gatewayRoot() {
  if (process.env.ANTIDAPTER_HOME) {
    return path.resolve(process.env.ANTIDAPTER_HOME);
  }
  return path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
}

function pythonExecutable() {
  return process.env.ANTIDAPTER_PYTHON || DEFAULT_PYTHON;
}

function childEnv() {
  return {
    ...process.env,
    // A supervised gateway must never be the thing that opens a browser.
    ANTIDAPTER_ALLOW_INTERACTIVE_LOGIN: "false",
  };
}

/**
 * Run a gateway subcommand, yielding one parsed object per NDJSON line.
 * stderr is inherited so logs land in pi's output rather than vanishing.
 */
function runGateway(args, { onEvent, signal, timeoutMs } = {}) {
  const root = gatewayRoot();
  if (!existsSync(path.join(root, "main.py"))) {
    throw new Error(
      `Antidapter gateway not found at ${root}. ` +
        `Set ANTIDAPTER_HOME to a checkout, or reinstall the package.`
    );
  }
  const child = spawn(pythonExecutable(), [path.join(root, "main.py"), ...args, "--json-events"], {
    cwd: root,
    env: childEnv(),
    stdio: ["ignore", "pipe", "inherit"],
  });

  let buffer = "";
  let timer = null;
  const onAbort = () => child.kill("SIGTERM");
  const cleanup = () => {
    if (timer) clearTimeout(timer);
    if (signal) signal.removeEventListener("abort", onAbort);
  };

  if (signal) signal.addEventListener("abort", onAbort, { once: true });
  if (timeoutMs) {
    timer = setTimeout(() => child.kill("SIGTERM"), timeoutMs);
  }

  child.stdout.setEncoding("utf8");
  child.stdout.on("data", (chunk) => {
    buffer += chunk;
    let index;
    while ((index = buffer.indexOf("\n")) !== -1) {
      const line = buffer.slice(0, index).trim();
      buffer = buffer.slice(index + 1);
      if (!line) continue;
      try {
        onEvent?.(JSON.parse(line));
      } catch {
        // Non-JSON on stdout would be a gateway bug; ignore rather than crash pi.
      }
    }
  });

  return new Promise((resolve, reject) => {
    child.on("error", (error) => {
      cleanup();
      reject(
        error.code === "ENOENT"
          ? new Error(
              `Could not run '${pythonExecutable()}'. Install Python 3.10+ or set ANTIDAPTER_PYTHON.`
            )
          : error
      );
    });
    child.on("close", (code) => {
      cleanup();
      resolve(code ?? 0);
    });
  });
}

function isAuthenticated() {
  return new Promise((resolve) => {
    runGateway(["ensure-auth"])
      .then((code) => resolve(code === 0))
      .catch(() => resolve(false));
  });
}

/** `/health` lives at the server root, not under the /v1 route prefix. */
function healthUrlFor(url) {
  return url.replace(/\/v1$/, "") + "/health";
}

async function isHealthy(url) {
  try {
    const response = await fetch(healthUrl ?? healthUrlFor(url), {
      signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
    });
    return response.ok;
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Gateway supervision
// ---------------------------------------------------------------------------

let child = null;
let baseUrl = null;
let healthUrl = null;

async function ensureGateway() {
  if (baseUrl) return baseUrl;

  const external = process.env.ANTIDAPTER_BASE_URL;
  if (external) {
    baseUrl = external.replace(/\/+$/, "");
    healthUrl = healthUrlFor(baseUrl);
    return baseUrl;
  }

  if (child) return await waitForHealth();

  const started = new Promise((resolve, reject) => {
    const root = gatewayRoot();
    if (!existsSync(path.join(root, "main.py"))) {
      reject(
        new Error(
          `Antidapter gateway not found at ${root}. Set ANTIDAPTER_HOME to a checkout.`
        )
      );
      return;
    }
    child = spawn(
      pythonExecutable(),
      [
        path.join(root, "main.py"),
        "serve",
        "--port",
        "0",
        "--idle-timeout",
        String(ORPHAN_IDLE_TIMEOUT_S),
        "--json-events",
      ],
      {
        cwd: root,
        env: childEnv(),
        stdio: ["ignore", "pipe", "inherit"],
        // Own process group, so a hard kill of pi does not leave orphans we
        // cannot signal.
        detached: false,
      }
    );
    child.on("error", (error) => {
      child = null;
      reject(
        error.code === "ENOENT"
          ? new Error(
              `Could not run '${pythonExecutable()}'. Install Python 3.10+ or set ANTIDAPTER_PYTHON.`
            )
          : error
      );
    });
    child.on("close", () => {
      child = null;
      baseUrl = null;
    });

    let buffer = "";
    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk) => {
      buffer += chunk;
      let index;
      while ((index = buffer.indexOf("\n")) !== -1) {
        const line = buffer.slice(0, index).trim();
        buffer = buffer.slice(index + 1);
        if (!line) continue;
        let event;
        try {
          event = JSON.parse(line);
        } catch {
          continue;
        }
        if (event.event === "serve_ready") {
          baseUrl = String(event.base_url).replace(/\/+$/, "");
          // The gateway tells us where /health is; it is not under /v1.
          healthUrl = event.health_url ? String(event.health_url) : healthUrlFor(baseUrl);
          resolve(baseUrl);
        } else if (event.event === "error") {
          reject(new Error(event.message || "gateway failed to start"));
        }
      }
    });

    setTimeout(() => {
      if (!baseUrl) {
        stopGateway();
        reject(new Error("Gateway did not report a listening port in time"));
      }
    }, HEALTH_TIMEOUT_MS).unref?.();
  });

  baseUrl = await started;
  return await waitForHealth();
}

async function waitForHealth() {
  const deadline = Date.now() + HEALTH_TIMEOUT_MS;
  while (Date.now() < deadline) {
    if (await isHealthy(baseUrl)) return baseUrl;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`Gateway at ${baseUrl} never became healthy`);
}

function killChildNow() {
  if (!child) return;
  child.kill("SIGTERM");
  child = null;
  baseUrl = null;
  healthUrl = null;
}

function stopGateway() {
  if (!child) return;
  const dying = child;
  child = null;
  baseUrl = null;
  healthUrl = null;
  dying.kill("SIGTERM");
  // Escalate if it ignores the polite request.
  setTimeout(() => {
    if (dying.exitCode === null) dying.kill("SIGKILL");
  }, GATEWAY_SHUTDOWN_GRACE_MS).unref?.();
}

// ---------------------------------------------------------------------------
// Model catalog
// ---------------------------------------------------------------------------

function toPiModel(entry, url) {
  const contextWindow = Number.isSafeInteger(entry.contextWindow)
    ? entry.contextWindow
    : 1_048_576;
  const maxTokens = Number.isSafeInteger(entry.maxTokens) ? entry.maxTokens : 65_536;
  return {
    id: entry.id,
    name: entry.name || entry.id,
    api: "openai-completions",
    provider: PROVIDER_ID,
    baseUrl: url,
    reasoning: entry.reasoning === true,
    input: Array.isArray(entry.input) ? entry.input : ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow,
    maxTokens,
    compat: {
      supportsStore: false,
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
      supportsUsageInStreaming: true,
      supportsStrictMode: false,
      maxTokensField: "max_tokens",
    },
  };
}

async function fetchModels(url) {
  const response = await fetch(`${url}/gateway/models`, {
    headers: {
      Authorization: `Bearer ${process.env.ANTIDAPTER_API_KEY || "none"}`,
      Accept: "application/json",
    },
    signal: AbortSignal.timeout(10_000),
  });
  if (!response.ok) {
    throw new Error(`catalog request failed: HTTP ${response.status}`);
  }
  const payload = await response.json();
  if (!Array.isArray(payload?.models)) {
    throw new Error("unexpected catalog shape");
  }
  return payload.models.map((entry) => toPiModel(entry, url));
}

// ---------------------------------------------------------------------------
// Extension
// ---------------------------------------------------------------------------

/**
 * Start the gateway and read its catalog while this module is imported.
 *
 * This has to happen at module scope, not in a `session_start` handler:
 * `pi --list-models` prints and exits without ever firing `session_start`, and
 * `pi --provider agy` resolves its provider before any session exists. Doing
 * it here is what makes the model list correct everywhere.
 */
async function boot() {
  try {
    const url = await ensureGateway();
    const models = await fetchModels(url);
    return { url, models: models.length > 0 ? models : null };
  } catch (error) {
    return { url: null, models: null, error: error instanceof Error ? error : new Error(String(error)) };
  }
}

const startup = await boot();

export default function agyExtension(pi) {
  let url = startup.url;

  const register = (models) => {
    pi.registerProvider(PROVIDER_ID, {
      name: PROVIDER_NAME,
      baseUrl: url || "http://127.0.0.1:9/v1",
      apiKey: process.env.ANTIDAPTER_API_KEY || "none",
      api: "openai-completions",
      models,
      oauth: {
        name: PROVIDER_NAME,
        loginLabel: LOGIN_LABEL,
        isSubscription: false,

        /** Browser authorization, driven by the gateway's own login command. */
        login: async (callbacks) => {
          let authUrl = null;
          let failure = null;

          const code = await runGateway(["login"], {
            signal: callbacks.signal,
            timeoutMs: LOGIN_TIMEOUT_MS,
            onEvent: (event) => {
              if (event.event === "auth_url") {
                authUrl = event.url;
                // pi renders the URL and opens a browser for us.
                callbacks.onAuth({
                  url: event.url,
                  instructions: "Complete the Google sign-in, then return to pi.",
                });
              } else if (event.event === "progress") {
                callbacks.onProgress?.(event.message);
              } else if (event.event === "error") {
                failure = event.message;
              }
            },
          });

          if (failure) throw new Error(failure);
          if (code !== 0 || !authUrl) {
            throw new Error("Authorization did not complete");
          }

          // The gateway re-reads its credentials on demand, so the new token is
          // picked up without a restart. Re-register the catalog now so /model
          // offers the real models instead of the seed.
          try {
            url = await ensureGateway();
            const models = await fetchModels(url);
            if (models.length > 0) register(models);
          } catch {
            // Login succeeded; a catalog refresh can retry on next start.
          }

          return {
            // Marker only: the real refresh token lives in the gateway's own
            // 0600 credentials file, so there is a single copy of it.
            refresh: "antidapter",
            access: "antidapter",
            expires: Math.floor(Date.now() / 1000) + 3600,
          };
        },

        /**
         * The gateway refreshes tokens on its own, transparently. This exists so
         * pi's expiry view stays honest without duplicating the secret.
         */
        refreshToken: async (credential) => {
          const ok = await isAuthenticated();
          if (!ok) {
            throw new Error("Antidapter credentials are no longer valid; run /login again");
          }
          return { ...credential, expires: Math.floor(Date.now() / 1000) + 3600 };
        },

        getApiKey: () => process.env.ANTIDAPTER_API_KEY || "local",
      },
    });
  };

  register(startup.models ?? SEED_MODELS);

  pi.on("session_start", async (_event, ctx) => {
    // Retry: the load-time attempt may have run before the user started the
    // gateway themselves, or failed transiently.
    try {
      url = await ensureGateway();
      const models = await fetchModels(url);
      if (models.length > 0) {
        register(models);
        if (ctx?.model?.provider === PROVIDER_ID) {
          ctx?.ui?.notify?.(`${PROVIDER_ID}: ${models.length} models from Antidapter`, "info");
        }
      }
    } catch (error) {
      if (ctx?.model?.provider === PROVIDER_ID) {
        ctx?.ui?.notify?.(`${PROVIDER_ID}: ${error.message}`, "warning");
      }
    }
  });

  pi.on("session_shutdown", () => {
    stopGateway();
  });

  // `session_shutdown` does not fire on every exit path (a non-interactive
  // `-p` run, a signal, an uncaught throw), and a leaked Python process would
  // hold a port forever. These are the backstops; child.kill() is synchronous,
  // so it is safe in an `exit` handler.
  process.once("exit", () => killChildNow());
  for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"]) {
    process.once(signal, () => {
      killChildNow();
      process.exit(130);
    });
  }
}
