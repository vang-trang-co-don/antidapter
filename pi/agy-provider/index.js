/**
 * pi extension: registers an `agy` provider backed by Antidapter.
 *
 * Why an extension rather than a models.json entry: Antidapter's catalog
 * changes whenever Google rotates models (it currently exposes 27, and that
 * number moves). A static provider entry goes stale silently. This extension
 * registers the provider immediately, then replaces its model list with the
 * live catalog from GET /v1/gateway/models once the session starts.
 *
 * Configuration (all optional):
 *   ANTIDAPTER_BASE_URL   default http://127.0.0.1:8080/v1
 *   ANTIDAPTER_API_KEY    only needed when the gateway sets ANTIDAPTER_API_KEY
 *
 * Enable globally:  pi install ./pi/agy-provider
 * Enable per-run:   pi --extension ./pi/agy-provider
 */

const PROVIDER_ID = "agy";
const PROVIDER_NAME = "Antigravity (Antidapter)";
const DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1";

/**
 * Bound on the startup catalog probe. The gateway is on loopback, so a healthy
 * setup answers in well under 100ms; this only governs how long a *down*
 * gateway stalls the first render.
 */
const STARTUP_TIMEOUT_MS = 2_000;
const REFRESH_TIMEOUT_MS = 5_000;

/** Placeholder so the provider exists before the live catalog is fetched. */
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

function baseUrl() {
  const raw = process.env.ANTIDAPTER_BASE_URL || DEFAULT_BASE_URL;
  return raw.replace(/\/+$/, "");
}

function apiKey() {
  // The gateway binds to loopback and is unauthenticated by default, but pi
  // still requires a non-empty key to build a request.
  return process.env.ANTIDAPTER_API_KEY || "none";
}

function toPiModel(entry, url) {
  const contextWindow = Number.isSafeInteger(entry.contextWindow) ? entry.contextWindow : 1_048_576;
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

async function fetchCatalog(timeoutMs) {
  const url = baseUrl();
  const response = await fetch(`${url}/gateway/models`, {
    headers: { Authorization: `Bearer ${apiKey()}`, Accept: "application/json" },
    signal: AbortSignal.timeout(timeoutMs ?? REFRESH_TIMEOUT_MS),
  });
  if (!response.ok) {
    throw new Error(`Antidapter returned HTTP ${response.status} from ${url}/gateway/models`);
  }
  const payload = await response.json();
  if (!Array.isArray(payload?.models)) {
    throw new Error(`Unexpected catalog shape from ${url}/gateway/models`);
  }
  return payload.models.map((entry) => toPiModel(entry, url));
}

function register(pi, models) {
  pi.registerProvider(PROVIDER_ID, {
    name: PROVIDER_NAME,
    baseUrl: baseUrl(),
    apiKey: apiKey(),
    api: "openai-completions",
    models,
  });
}

/**
 * Probe the live catalog while this module is being imported.
 *
 * `pi --list-models` prints and exits without ever firing `session_start`, so a
 * catalog fetched only from that hook would leave the model list showing the
 * single seed entry. Fetching here makes `--list-models` correct too. A short
 * timeout keeps a stopped gateway from stalling startup, and any failure falls
 * back to the seed rather than breaking the extension.
 */
const startupCatalog = await fetchCatalog(STARTUP_TIMEOUT_MS).catch(() => []);
const initialModels = startupCatalog.length > 0 ? startupCatalog : SEED_MODELS;

export default function agyExtension(pi) {
  register(pi, initialModels);

  // Re-probe on session start so a gateway that came up after pi launched is
  // picked up without a restart.
  pi.on("session_start", async (_event, ctx) => {
    try {
      const models = await fetchCatalog();
      if (models.length > 0) {
        register(pi, models);
        ctx?.ui?.notify?.(`${PROVIDER_ID}: ${models.length} models from Antidapter`, "info");
      }
    } catch {
      ctx?.ui?.notify?.(
        `${PROVIDER_ID}: could not reach Antidapter at ${baseUrl()} (run: antidapter serve)`,
        "warning"
      );
    }
  });
}
