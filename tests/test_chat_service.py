import unittest
from collections.abc import Iterator

from core.domain.entities import (
    ChatMessage,
    ChatRequest,
    ModelInfo,
    Role,
    StreamDelta,
    TokenUsage,
    ToolCall,
)
from core.domain.exceptions import ModelNotFoundError
from core.ports.inbound import AuthUseCase
from core.ports.outbound import UpstreamModelPort
from core.services.chat_service import ChatService, new_chat_id
from core.services.model_catalog_service import ModelCatalogService


class FakeAuth(AuthUseCase):
    def __init__(self, token: str = "test-token"):
        self.token = token
        self.calls = 0

    def ensure_authenticated(self) -> str:
        self.calls += 1
        return self.token

    def login_interactive(self) -> None:
        pass

    def logout(self) -> None:
        pass


class ScriptedUpstream(UpstreamModelPort):
    def __init__(self, deltas: tuple[StreamDelta, ...] = ()):
        self.deltas = deltas
        self.tokens: list[str] = []
        self.requests: list[ChatRequest] = []

    def stream_generate(self, token: str, request: ChatRequest) -> Iterator[StreamDelta]:
        self.tokens.append(token)
        self.requests.append(request)
        return iter(self.deltas)

    def fetch_models(self, token: str) -> tuple[ModelInfo, ...]:
        self.tokens.append(token)
        return (
            ModelInfo(id="b-model", display_name="B", provider="google"),
            ModelInfo(id="a-model", display_name="A", provider="google"),
        )


def request(stream: bool = False) -> ChatRequest:
    return ChatRequest(
        model="test-model",
        messages=(ChatMessage.from_text(Role.USER, "Hi"),),
        stream=stream,
    )


class TestChatServiceComplete(unittest.TestCase):
    def setUp(self):
        self.upstream = ScriptedUpstream(
            (
                StreamDelta(text="Hello "),
                StreamDelta(text="world"),
                StreamDelta(
                    finish_reason="stop",
                    usage=TokenUsage(5, 2, 7),
                ),
            )
        )
        self.auth = FakeAuth()
        self.service = ChatService(self.auth, self.upstream)

    def test_aggregates_content_and_usage(self):
        response = self.service.complete(request())
        self.assertEqual(response.content, "Hello world")
        self.assertEqual(response.model, "test-model")
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.usage.total_tokens, 7)
        self.assertTrue(response.id.startswith("chatcmpl-"))
        self.assertEqual(self.upstream.tokens, ["test-token"])

    def test_reports_no_usage_when_upstream_sends_none(self):
        """Regression: an all-zero usage block used to be reported as real."""
        upstream = ScriptedUpstream((StreamDelta(text="hi"),))
        response = ChatService(FakeAuth(), upstream).complete(request())
        self.assertIsNone(response.usage)

    def test_keeps_last_cumulative_usage_without_double_counting(self):
        upstream = ScriptedUpstream(
            (
                StreamDelta(usage=TokenUsage(5, 1, 6)),
                StreamDelta(usage=TokenUsage(5, 4, 9)),
            )
        )
        response = ChatService(FakeAuth(), upstream).complete(request())
        self.assertEqual(response.usage.total_tokens, 9)

    def test_collects_reasoning_and_tool_calls(self):
        upstream = ScriptedUpstream(
            (
                StreamDelta(reasoning="thinking..."),
                StreamDelta(tool_calls=(ToolCall("c1", "get_weather", '{"city":"Oslo"}'),)),
                StreamDelta(finish_reason="stop"),
            )
        )
        response = ChatService(FakeAuth(), upstream).complete(request())
        self.assertEqual(response.reasoning, "thinking...")
        self.assertEqual(response.tool_calls[0].function_name, "get_weather")
        self.assertEqual(response.finish_reason, "tool_calls")

    def test_does_not_authenticate_lazily(self):
        """complete_stream must authenticate before returning, not on next()."""
        service = ChatService(self.auth, ScriptedUpstream())
        iterator = service.complete_stream(request(stream=True))
        self.assertEqual(self.auth.calls, 1, "authentication was deferred")
        list(iterator)

    def test_propagates_auth_failure_before_returning(self):
        from core.domain.exceptions import AuthenticationError

        class FailingAuth(FakeAuth):
            def ensure_authenticated(self) -> str:
                raise AuthenticationError("nope")

        service = ChatService(FailingAuth(), ScriptedUpstream())
        with self.assertRaises(AuthenticationError) as ctx:
            service.complete_stream(request(stream=True))
        self.assertEqual(ctx.exception.code, "authentication_error")


class TestModelCatalogService(unittest.TestCase):
    def setUp(self):
        self.service = ModelCatalogService(FakeAuth(), ScriptedUpstream())

    def test_returns_immutable_tuple(self):
        models = self.service.list_models()
        self.assertIsInstance(models, tuple)
        self.assertEqual({model.id for model in models}, {"a-model", "b-model"})

    def test_get_model_finds_by_id(self):
        self.assertEqual(self.service.get_model("a-model").display_name, "A")

    def test_get_model_raises_for_unknown(self):
        with self.assertRaises(ModelNotFoundError) as ctx:
            self.service.get_model("nope")
        self.assertEqual(ctx.exception.model_id, "nope")
        self.assertEqual(ctx.exception.code, "model_not_found")


class TestChatId(unittest.TestCase):
    def test_ids_are_unique_and_prefixed(self):
        self.assertNotEqual(new_chat_id(), new_chat_id())
        self.assertTrue(new_chat_id().startswith("chatcmpl-"))


if __name__ == "__main__":
    unittest.main()
