import time
import unittest

from core.domain.entities import (
    AuthToken,
    ChatMessage,
    ChatRequest,
    GenerationParameters,
    ImagePart,
    QuotaInfo,
    Role,
    TextPart,
    ToolCall,
    ToolCallPart,
    ToolDefinition,
    ToolResultPart,
)
from core.domain.exceptions import ValidationError


class TestContentParts(unittest.TestCase):
    def test_text_part_rejects_non_string(self):
        with self.assertRaises(ValidationError):
            TextPart(text=123)  # type: ignore[arg-type]

    def test_validation_error_is_also_a_value_error(self):
        """Domain validation stays catchable as ValueError for compatibility."""
        with self.assertRaises(ValueError):
            TextPart(text=123)  # type: ignore[arg-type]

    def test_image_part_requires_both_fields(self):
        with self.assertRaises(ValidationError):
            ImagePart(mime_type="", base64_data="abc")
        with self.assertRaises(ValidationError):
            ImagePart(mime_type="image/png", base64_data="")


class TestChatMessage(unittest.TestCase):
    def test_from_text(self):
        message = ChatMessage.from_text(role=Role.USER, text="hello")
        self.assertEqual(message.role, Role.USER)
        self.assertEqual(message.text_content, "hello")

    def test_text_blocks_keep_their_boundaries(self):
        """Regression: blocks used to be concatenated with no separator."""
        message = ChatMessage(
            role=Role.USER,
            parts=(TextPart("Para one."), TextPart("Para two.")),
        )
        self.assertEqual(message.text_content, "Para one.\nPara two.")

    def test_empty_parts_rejected(self):
        with self.assertRaises(ValidationError):
            ChatMessage(role=Role.USER, parts=())


class TestGenerationParameters(unittest.TestCase):
    def test_accepts_valid_values(self):
        params = GenerationParameters(temperature=0.7, top_p=0.9, max_output_tokens=64)
        self.assertEqual(params.temperature, 0.7)

    def test_rejects_non_numeric_temperature(self):
        """Regression: 'hot' used to be forwarded verbatim to the upstream."""
        with self.assertRaises(ValidationError):
            GenerationParameters(temperature="hot")  # type: ignore[arg-type]

    def test_rejects_out_of_range_values(self):
        with self.assertRaises(ValidationError):
            GenerationParameters(temperature=2.5)
        with self.assertRaises(ValidationError):
            GenerationParameters(top_p=1.5)
        with self.assertRaises(ValidationError):
            GenerationParameters(max_output_tokens=0)

    def test_rejects_bool_masquerading_as_number(self):
        with self.assertRaises(ValidationError):
            GenerationParameters(temperature=True)  # type: ignore[arg-type]


class TestChatRequest(unittest.TestCase):
    def test_valid(self):
        request = ChatRequest(
            model="gemini-3.6-flash-low",
            messages=(ChatMessage.from_text(Role.USER, "hello"),),
            stream=True,
            parameters=GenerationParameters(temperature=0.7),
        )
        self.assertTrue(request.stream)

    def test_rejects_blank_model(self):
        with self.assertRaises(ValidationError):
            ChatRequest(model="   ", messages=(ChatMessage.from_text(Role.USER, "hi"),))

    def test_rejects_empty_messages(self):
        with self.assertRaises(ValidationError):
            ChatRequest(model="m", messages=())


class TestToolEntities(unittest.TestCase):
    def test_tool_call_part_requires_name(self):
        with self.assertRaises(ValidationError):
            ToolCallPart(call_id="c1", function_name="", arguments="{}")

    def test_tool_result_part_requires_call_id(self):
        with self.assertRaises(ValidationError):
            ToolResultPart(call_id="", function_name="f", content="ok")

    def test_tool_definition_requires_name(self):
        with self.assertRaises(ValidationError):
            ToolDefinition(name="")


class TestAuthToken(unittest.TestCase):
    def test_expiry(self):
        now = time.time()
        self.assertTrue(AuthToken("t", "r", now - 10).is_expired())
        self.assertFalse(AuthToken("t", "r", now + 3600).is_expired())

    def test_round_trips_through_dict(self):
        token = AuthToken("access", "refresh", time.time() + 10, "Bearer")
        self.assertEqual(AuthToken.from_dict(token.to_dict()), token)

    def test_from_dict_rejects_non_numeric_expiry(self):
        with self.assertRaises(ValidationError):
            AuthToken.from_dict({"access_token": "a", "expiry_time": "soon"})

    def test_requires_access_token(self):
        with self.assertRaises(ValidationError):
            AuthToken.from_dict({"refresh_token": "r"})


class TestTokenUsage(unittest.TestCase):
    def test_addition(self):
        from core.domain.entities import TokenUsage

        total = TokenUsage(1, 2, 3) + TokenUsage(10, 20, 30)
        self.assertEqual((total.prompt_tokens, total.total_tokens), (11, 33))


class TestModelInfo(unittest.TestCase):
    def test_quota_is_optional(self):
        from core.domain.entities import ModelInfo

        model = ModelInfo(id="m", display_name="M", provider="google")
        self.assertIsNone(model.quota_info)
        self.assertIsNotNone(
            ModelInfo(id="m", display_name="M", provider="g", quota_info=QuotaInfo(0.5, "t"))
        )


class TestToolCall(unittest.TestCase):
    def test_tool_call_is_hashable_value(self):
        call = ToolCall(call_id="c1", function_name="f", arguments="{}")
        self.assertEqual(call, ToolCall("c1", "f", "{}"))


if __name__ == "__main__":
    unittest.main()
