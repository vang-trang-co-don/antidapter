import json
import unittest

from adapters.inbound.http.openai_adapter import OpenAIProtocolTranslator
from config import ProtocolConfig
from core.domain.entities import (
    ChatResponse,
    ImagePart,
    ModelInfo,
    QuotaInfo,
    Role,
    StreamDelta,
    TokenUsage,
    ToolCall,
    ToolCallPart,
    ToolResultPart,
)
from core.domain.exceptions import ValidationError


def parse(payload: dict, **config_kwargs):
    translator = OpenAIProtocolTranslator(ProtocolConfig(**config_kwargs))
    return translator.parse_chat_request(json.dumps(payload).encode("utf-8"))


class TestRequestParsing(unittest.TestCase):
    def test_simple_string_message(self):
        request = parse(
            {
                "model": "custom",
                "messages": [{"role": "user", "content": "Hello!"}],
                "temperature": 0.5,
            }
        )
        self.assertEqual(request.model, "custom")
        self.assertEqual(request.messages[0].role, Role.USER)
        self.assertEqual(request.messages[0].parts[0].text, "Hello!")
        self.assertEqual(request.parameters.temperature, 0.5)

    def test_missing_model_uses_configured_default(self):
        request = parse({"messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(request.model, "gemini-3.6-flash-low")

    def test_blank_model_uses_configured_default(self):
        request = parse({"model": "   ", "messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(request.model, "gemini-3.6-flash-low")

    def test_content_blocks(self):
        request = parse(
            {
                "model": "m",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Part 1"},
                            {"type": "text", "text": " Part 2"},
                        ],
                    }
                ],
            }
        )
        self.assertEqual(len(request.messages[0].parts), 2)

    def test_invalid_json(self):
        translator = OpenAIProtocolTranslator()
        with self.assertRaises(ValidationError):
            translator.parse_chat_request(b"not json")

    def test_non_object_body(self):
        with self.assertRaises(ValidationError):
            parse([1, 2, 3])

    def test_missing_messages(self):
        with self.assertRaises(ValidationError):
            parse({"model": "m"})

    def test_empty_messages(self):
        with self.assertRaises(ValidationError):
            parse({"model": "m", "messages": []})

    def test_prefers_max_completion_tokens(self):
        request = parse(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "max_tokens": 10,
                "max_completion_tokens": 99,
            }
        )
        self.assertEqual(request.parameters.max_output_tokens, 99)

    def test_rejects_bad_parameter_type(self):
        with self.assertRaises(ValidationError):
            parse(
                {
                    "model": "m",
                    "messages": [{"role": "user", "content": "x"}],
                    "temperature": "hot",
                }
            )


class TestRoleHandling(unittest.TestCase):
    def test_developer_role_maps_to_system(self):
        request = parse({"model": "m", "messages": [{"role": "developer", "content": "x"}]})
        self.assertEqual(request.messages[0].role, Role.SYSTEM)

    def test_unknown_role_is_rejected_not_coerced(self):
        """Regression: unknown roles used to be silently relabelled as `user`."""
        for role in ("wizard", "moderator"):
            with self.assertRaises(ValidationError) as ctx:
                parse({"model": "m", "messages": [{"role": role, "content": "x"}]})
            self.assertIn(role, ctx.exception.message)

    def test_tool_result_is_preserved(self):
        request = parse(
            {
                "model": "m",
                "messages": [
                    {"role": "user", "content": "weather?"},
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "name": "get_weather",
                        "content": "12C",
                    },
                ],
            }
        )
        self.assertEqual(request.messages[1].role, Role.TOOL)
        part = request.messages[1].parts[0]
        self.assertIsInstance(part, ToolResultPart)
        self.assertEqual(part.call_id, "call_1")
        self.assertEqual(part.content, "12C")

    def test_assistant_tool_calls_are_preserved(self):
        request = parse(
            {
                "model": "m",
                "messages": [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "f", "arguments": '{"a":1}'},
                            }
                        ],
                    }
                ],
            }
        )
        part = request.messages[0].parts[0]
        self.assertIsInstance(part, ToolCallPart)
        self.assertEqual(part.function_name, "f")

    def test_tool_message_requires_call_id(self):
        with self.assertRaises(ValidationError):
            parse({"model": "m", "messages": [{"role": "tool", "content": "x"}]})


class TestImageParsing(unittest.TestCase):
    def test_data_url(self):
        request = parse(
            {
                "model": "m",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,QUJD"},
                            }
                        ],
                    }
                ],
            }
        )
        part = request.messages[0].parts[0]
        self.assertIsInstance(part, ImagePart)
        self.assertEqual((part.mime_type, part.base64_data), ("image/png", "QUJD"))

    def test_remote_url_is_rejected(self):
        """Fetching remote URLs would turn the gateway into an SSRF proxy."""
        with self.assertRaises(ValidationError) as ctx:
            parse(
                {
                    "model": "m",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": "http://x/y.png"}}
                            ],
                        }
                    ],
                }
            )
        self.assertIn("data URL", ctx.exception.message)

    def test_malformed_data_url_is_rejected_not_swallowed(self):
        with self.assertRaises(ValidationError):
            parse(
                {
                    "model": "m",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": "data:image/png"}}
                            ],
                        }
                    ],
                }
            )

    def test_unsupported_block_type_is_rejected(self):
        with self.assertRaises(ValidationError):
            parse(
                {
                    "model": "m",
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "input_audio", "input_audio": {}}],
                        }
                    ],
                }
            )


class TestToolDefinitionParsing(unittest.TestCase):
    def test_parses_function_tools(self):
        request = parse(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            }
        )
        self.assertEqual(request.tools[0].name, "get_weather")
        self.assertEqual(request.tools[0].parameters, {"type": "object"})

    def test_rejects_non_function_tool(self):
        with self.assertRaises(ValidationError):
            parse(
                {
                    "model": "m",
                    "messages": [{"role": "user", "content": "x"}],
                    "tools": [{"type": "retrieval"}],
                }
            )


class TestToolSchemaNormalization(unittest.TestCase):
    """The upstream requires JSON Schema draft 2020-12 and rejects anything else.

    Forwarding a client's dialect unchanged produced an opaque 502 that the
    client retried repeatedly, so schemas are rewritten at the inbound edge.
    """

    def _params(self, parameters):
        request = parse(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "tools": [
                    {"type": "function", "function": {"name": "f", "parameters": parameters}}
                ],
            }
        )
        return request.tools[0].parameters

    def test_missing_root_type_is_supplied(self):
        """Anthropic-style input_schema omits it, and the upstream rejects that."""
        self.assertEqual(self._params({"properties": {"a": {"type": "string"}}})["type"], "object")

    def test_existing_root_type_is_preserved(self):
        self.assertEqual(self._params({"type": "object"})["type"], "object")

    def test_schema_key_is_stripped(self):
        self.assertNotIn(
            "$schema", self._params({"$schema": "http://json-schema.org/draft-07/schema#"})
        )

    def test_definitions_is_renamed_to_defs(self):
        schema = self._params({"definitions": {"A": {"type": "string"}}})
        self.assertIn("$defs", schema)
        self.assertNotIn("definitions", schema)

    def test_draft4_boolean_exclusives_are_dropped(self):
        schema = self._params({"properties": {"n": {"minimum": 1, "exclusiveMinimum": True}}})
        self.assertNotIn("exclusiveMinimum", schema["properties"]["n"])
        self.assertEqual(schema["properties"]["n"]["minimum"], 1)

    def test_numeric_exclusives_survive(self):
        schema = self._params({"properties": {"n": {"exclusiveMinimum": 3}}})
        self.assertEqual(schema["properties"]["n"]["exclusiveMinimum"], 3)

    def test_tuple_items_become_prefix_items(self):
        schema = self._params({"properties": {"t": {"items": [{"type": "string"}]}}})
        self.assertIn("prefixItems", schema["properties"]["t"])
        self.assertNotIn("items", schema["properties"]["t"])

    def test_nested_schemas_are_rewritten_recursively(self):
        schema = self._params(
            {"properties": {"o": {"type": "object", "definitions": {"X": {"type": "string"}}}}}
        )
        self.assertIn("$defs", schema["properties"]["o"])

    def test_none_schema_stays_none(self):
        request = parse(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "tools": [{"type": "function", "function": {"name": "f"}}],
            }
        )
        self.assertIsNone(request.tools[0].parameters)

    def test_non_object_schema_is_a_400_naming_the_tool(self):
        with self.assertRaises(ValidationError) as ctx:
            self._params(["not", "a", "schema"])
        self.assertIn("'f'", ctx.exception.message)

    def test_non_object_properties_is_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            self._params({"properties": "nope"})
        self.assertIn("'f'", ctx.exception.message)


class TestSerialization(unittest.TestCase):
    def setUp(self):
        self.translator = OpenAIProtocolTranslator()

    def test_chat_response(self):
        payload = json.loads(
            self.translator.serialize_chat_response(
                ChatResponse(
                    id="chatcmpl-1",
                    model="m",
                    content="hi",
                    created_at=1000,
                    usage=TokenUsage(1, 2, 3),
                )
            ).decode()
        )
        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(payload["choices"][0]["message"]["content"], "hi")
        self.assertEqual(payload["usage"]["total_tokens"], 3)

    def test_response_omits_usage_when_absent(self):
        payload = json.loads(
            self.translator.serialize_chat_response(
                ChatResponse(id="i", model="m", content="", created_at=1)
            ).decode()
        )
        self.assertNotIn("usage", payload)

    def test_response_serializes_tool_calls(self):
        payload = json.loads(
            self.translator.serialize_chat_response(
                ChatResponse(
                    id="i",
                    model="m",
                    content="",
                    created_at=1,
                    tool_calls=(ToolCall("c1", "f", "{}"),),
                )
            ).decode()
        )
        call = payload["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual((call["id"], call["type"]), ("c1", "function"))

    def test_response_serializes_reasoning(self):
        payload = json.loads(
            self.translator.serialize_chat_response(
                ChatResponse(id="i", model="m", content="a", created_at=1, reasoning="why")
            ).decode()
        )
        self.assertEqual(payload["choices"][0]["message"]["reasoning_content"], "why")

    def test_stream_chunk(self):
        raw = self.translator.serialize_stream_chunk(
            "id-1", "m-1", StreamDelta(text="chunk"), 1000
        ).decode()
        self.assertTrue(raw.startswith("data: {"))
        parsed = json.loads(raw[5:].strip())
        self.assertEqual(parsed["choices"][0]["delta"]["content"], "chunk")
        self.assertEqual(parsed["object"], "chat.completion.chunk")

    def test_usage_only_delta_becomes_a_bare_usage_chunk(self):
        """OpenAI's include_usage chunk carries an empty choices array."""
        raw = self.translator.serialize_stream_chunk(
            "i", "m", StreamDelta(usage=TokenUsage(1, 1, 2)), 5
        ).decode()
        parsed = json.loads(raw[5:].strip())
        self.assertEqual(parsed["choices"], [])
        self.assertEqual(parsed["usage"]["total_tokens"], 2)

    def test_finish_only_delta_still_emits_a_choice(self):
        raw = self.translator.serialize_stream_chunk(
            "i", "m", StreamDelta(finish_reason="stop"), 5
        ).decode()
        parsed = json.loads(raw[5:].strip())
        self.assertEqual(len(parsed["choices"]), 1)
        self.assertEqual(parsed["choices"][0]["finish_reason"], "stop")
        self.assertEqual(parsed["choices"][0]["delta"], {})

    def test_stream_tool_call_chunk(self):
        raw = self.translator.serialize_stream_chunk(
            "i", "m", StreamDelta(tool_calls=(ToolCall("c1", "f", "{}"),)), 5
        ).decode()
        parsed = json.loads(raw[5:].strip())
        self.assertEqual(parsed["choices"][0]["delta"]["tool_calls"][0]["index"], 0)

    def test_done_sentinel(self):
        self.assertEqual(self.translator.serialize_stream_done(), b"data: [DONE]\n\n")

    def test_stream_error_is_an_sse_event(self):
        raw = self.translator.serialize_stream_error("boom", "upstream_service_error")
        self.assertTrue(raw.startswith(b"data: {"))
        self.assertEqual(json.loads(raw[5:].strip())["error"]["type"], "upstream_service_error")

    def test_models_list_and_single(self):
        models = (ModelInfo(id="a", display_name="A", provider="google"),)
        listing = json.loads(self.translator.serialize_models_list(models).decode())
        self.assertEqual(listing["object"], "list")
        self.assertEqual(listing["data"][0]["id"], "a")
        single = json.loads(self.translator.serialize_model(models[0]).decode())
        self.assertEqual(single["id"], "a")

    def test_model_details_include_capability_metadata(self):
        """The richer catalog shape clients need; /v1/models stays OpenAI-pure."""
        models = (
            ModelInfo(
                id="gemini-x",
                display_name="Gemini X",
                provider="google",
                context_window=1_048_576,
                max_output_tokens=65_536,
                supports_thinking=True,
                supports_tools=True,
                quota_info=QuotaInfo(remaining_fraction=0.5, reset_time="later"),
            ),
        )
        payload = json.loads(self.translator.serialize_model_details(models).decode())
        entry = payload["models"][0]
        self.assertEqual(entry["id"], "gemini-x")
        self.assertEqual(entry["contextWindow"], 1_048_576)
        self.assertEqual(entry["maxTokens"], 65_536)
        self.assertTrue(entry["reasoning"])
        self.assertTrue(entry["supportsTools"])
        self.assertEqual(entry["input"], ["text", "image"])
        self.assertEqual(entry["quota"]["remainingFraction"], 0.5)

    def test_model_details_omits_quota_when_unknown(self):
        payload = json.loads(
            self.translator.serialize_model_details(
                (ModelInfo(id="m", display_name="M", provider="g"),)
            ).decode()
        )
        self.assertIsNone(payload["models"][0]["quota"])

    def test_openai_model_listing_has_no_capability_fields(self):
        """Adding a separate route means /v1/models must stay spec-shaped."""
        raw = self.translator.serialize_models_list(
            (ModelInfo(id="m", display_name="M", provider="g"),)
        ).decode()
        for field in ("contextWindow", "reasoning", "maxTokens", "quota"):
            self.assertNotIn(field, raw)

    def test_error_shape(self):
        payload = json.loads(self.translator.serialize_error("bad", "validation_error", 400))
        self.assertEqual(payload["error"]["code"], 400)
        self.assertEqual(payload["error"]["type"], "validation_error")

    def test_wants_stream_usage(self):
        body = json.dumps({"stream_options": {"include_usage": True}}).encode()
        self.assertTrue(self.translator.wants_stream_usage(body))
        self.assertFalse(self.translator.wants_stream_usage(b"not json"))


if __name__ == "__main__":
    unittest.main()
