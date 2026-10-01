"""pi (pi.dev) provider configuration generator.

pi keeps custom providers in ``~/.pi/agent/models.json`` under ``providers``.
Each entry needs a base URL, an API dialect and an explicit model list, so the
block has to be kept in step with whatever the upstream catalog currently
returns. Hand-editing that JSON is how model lists silently drift, so this
module renders the block from the live catalog and can merge it back safely.

This is a driving adapter: it owns the shape of a *client's* configuration
file, not the domain and not the wire protocol.
"""

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.domain.entities import ModelInfo

DEFAULT_PROVIDER = "antidapter"
DEFAULT_MODELS_PATH = Path("~/.pi/agent/models.json").expanduser()
OPENAI_COMPLETIONS = "openai-completions"


@dataclass(frozen=True)
class PiProviderSettings:
    provider: str = DEFAULT_PROVIDER
    base_url: str = "http://127.0.0.1:8080/v1"
    api_key: str = "none"
    context_window: int = 1_048_576
    max_output_tokens: int = 65_536
    include_images: bool = True


def build_provider_block(
    models: Iterable[ModelInfo],
    settings: PiProviderSettings,
) -> dict[str, Any]:
    """Render a pi provider entry for the given catalog."""
    return {
        "baseUrl": settings.base_url,
        "api": OPENAI_COMPLETIONS,
        "apiKey": settings.api_key,
        "compat": {
            "supportsStore": False,
            "supportsDeveloperRole": False,
        },
        "models": [_render_model(model, settings) for model in models],
    }


def _render_model(model: ModelInfo, settings: PiProviderSettings) -> dict[str, Any]:
    accepted: list[str] = ["text"]
    if settings.include_images:
        accepted.append("image")
    return {
        "id": model.id,
        "name": model.display_name or model.id,
        "input": accepted,
        "contextWindow": model.context_window or settings.context_window,
        "maxTokens": model.max_output_tokens or settings.max_output_tokens,
    }


def merge_provider(
    models_path: Path,
    provider: str,
    block: Mapping[str, Any],
) -> tuple[bool, str]:
    """Insert or replace `provider` in pi's models.json.

    Returns (changed, description). The file is rewritten atomically so an
    interrupted run cannot corrupt pi's configuration.
    """
    document: dict[str, Any] = {}
    if models_path.exists():
        try:
            loaded = json.loads(models_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"could not read {models_path}: {exc}"
        if not isinstance(loaded, dict):
            return False, f"{models_path} is not a JSON object"
        document = loaded

    providers = document.setdefault("providers", {})
    if not isinstance(providers, dict):
        return False, f"{models_path} has a non-object 'providers' key"

    action = "replaced" if provider in providers else "added"
    providers[provider] = dict(block)

    models_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2) + "\n"
    # Write to a sibling temp file and rename, so an interrupted run can never
    # leave pi with a half-written configuration.
    descriptor, temp_path = tempfile.mkstemp(dir=models_path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(temp_path, models_path)
    except BaseException:
        _silent_unlink(temp_path)
        raise
    return True, f"{action} provider '{provider}' in {models_path}"


def _silent_unlink(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
