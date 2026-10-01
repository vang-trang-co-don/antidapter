import json
import os
import tempfile
import unittest
from pathlib import Path

from adapters.inbound.pi.pi_provider import (
    PiProviderSettings,
    build_provider_block,
    merge_provider,
)
from core.domain.entities import ModelInfo, QuotaInfo

SETTINGS = PiProviderSettings(provider="antidapter", base_url="http://127.0.0.1:8080/v1")

MODELS = (
    ModelInfo(
        id="gemini-3.6-flash-low",
        display_name="Gemini 3.6 Flash (Low)",
        provider="google",
        context_window=1_048_576,
        max_output_tokens=65_536,
        supports_thinking=True,
    ),
    ModelInfo(
        id="claude-sonnet-4-6",
        display_name="Claude Sonnet 4.6 (Thinking)",
        provider="google",
        context_window=200_000,
        max_output_tokens=64_000,
    ),
)


class TestProviderBlock(unittest.TestCase):
    def test_shape_matches_pi_expectations(self):
        block = build_provider_block(MODELS, SETTINGS)
        self.assertEqual(block["baseUrl"], "http://127.0.0.1:8080/v1")
        self.assertEqual(block["api"], "openai-completions")
        self.assertIn("compat", block)
        self.assertEqual(len(block["models"]), 2)

    def test_model_fields(self):
        entry = build_provider_block(MODELS, SETTINGS)["models"][0]
        self.assertEqual(entry["id"], "gemini-3.6-flash-low")
        self.assertEqual(entry["name"], "Gemini 3.6 Flash (Low)")
        self.assertEqual(entry["contextWindow"], 1_048_576)
        self.assertEqual(entry["maxTokens"], 65_536)
        self.assertEqual(entry["input"], ["text", "image"])

    def test_uses_per_model_capabilities(self):
        entries = build_provider_block(MODELS, SETTINGS)["models"]
        self.assertEqual(entries[1]["contextWindow"], 200_000)
        self.assertEqual(entries[1]["maxTokens"], 64_000)

    def test_falls_back_to_defaults_for_zero_values(self):
        bare = ModelInfo(id="m", display_name="", provider="google")
        entry = build_provider_block((bare,), SETTINGS)["models"][0]
        self.assertEqual(entry["name"], "m")
        self.assertEqual(entry["contextWindow"], SETTINGS.context_window)

    def test_images_can_be_disabled(self):
        settings = PiProviderSettings(include_images=False)
        entry = build_provider_block(MODELS, settings)["models"][0]
        self.assertEqual(entry["input"], ["text"])

    def test_api_key_is_carried_through_when_auth_is_enabled(self):
        settings = PiProviderSettings(api_key="s3cret")
        self.assertEqual(build_provider_block(MODELS, settings)["apiKey"], "s3cret")

    def test_quota_info_is_not_leaked_into_client_config(self):
        model = ModelInfo(
            id="m", display_name="M", provider="g", quota_info=QuotaInfo(0.5, "later")
        )
        self.assertNotIn("quotaInfo", json.dumps(build_provider_block((model,), SETTINGS)))


class TestMergeIntoPiConfig(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.path = self.dir / "models.json"

    def test_creates_the_file_when_absent(self):
        changed, description = merge_provider(self.path, "antidapter", {"baseUrl": "x"})
        self.assertTrue(changed)
        self.assertIn("added", description)
        self.assertEqual(
            json.loads(self.path.read_text())["providers"]["antidapter"]["baseUrl"], "x"
        )

    def test_preserves_other_providers(self):
        self.path.write_text(
            json.dumps({"providers": {"openai": {"baseUrl": "https://api.openai.com"}}})
        )
        merge_provider(self.path, "antidapter", {"baseUrl": "x"})
        providers = json.loads(self.path.read_text())["providers"]
        self.assertIn("openai", providers, "existing providers must survive")
        self.assertIn("antidapter", providers)

    def test_replaces_an_existing_provider(self):
        self.path.write_text(
            json.dumps({"providers": {"antidapter": {"baseUrl": "stale", "models": []}}})
        )
        _changed, description = merge_provider(self.path, "antidapter", {"baseUrl": "fresh"})
        self.assertIn("replaced", description)
        block = json.loads(self.path.read_text())["providers"]["antidapter"]
        self.assertEqual(block["baseUrl"], "fresh")

    def test_refuses_malformed_json_instead_of_clobbering_it(self):
        self.path.write_text("{ not json")
        changed, description = merge_provider(self.path, "antidapter", {"baseUrl": "x"})
        self.assertFalse(changed)
        self.assertIn("could not read", description)
        self.assertEqual(self.path.read_text(), "{ not json", "file must be left untouched")

    def test_refuses_a_non_object_providers_key(self):
        self.path.write_text(json.dumps({"providers": []}))
        changed, description = merge_provider(self.path, "antidapter", {"baseUrl": "x"})
        self.assertFalse(changed)
        self.assertIn("non-object", description)

    def test_writes_are_atomic_and_leave_no_temp_files(self):
        merge_provider(self.path, "antidapter", {"baseUrl": "x"})
        leftovers = [n for n in os.listdir(self.dir) if n.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    def test_creates_parent_directories(self):
        nested = self.dir / "a" / "b" / "models.json"
        changed, _ = merge_provider(nested, "antidapter", {"baseUrl": "x"})
        self.assertTrue(changed)
        self.assertTrue(nested.exists())

    def test_end_to_end_round_trip(self):
        block = build_provider_block(MODELS, SETTINGS)
        merge_provider(self.path, "antidapter", block)
        document = json.loads(self.path.read_text())
        stored = document["providers"]["antidapter"]
        self.assertEqual(stored["api"], "openai-completions")
        self.assertEqual(
            [m["id"] for m in stored["models"]], ["gemini-3.6-flash-low", "claude-sonnet-4-6"]
        )


if __name__ == "__main__":
    unittest.main()
