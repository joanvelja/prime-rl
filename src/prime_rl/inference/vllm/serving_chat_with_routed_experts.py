import base64
from collections.abc import AsyncGenerator, AsyncIterator

import numpy as np
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest, ChatCompletionResponse
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, RequestResponseMetadata
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.reasoning import ReasoningParser

logger = init_logger(__name__)


class _RoutedExpertsCapture:
    def __init__(self, generator: AsyncGenerator[RequestOutput, None]):
        self._generator = generator
        self.routed_experts: dict[int, dict] = {}

    def _encode_routed_experts(self, arr: np.ndarray) -> dict:
        return {
            "data": base64.b85encode(arr.tobytes()).decode("ascii"),
            "shape": list(arr.shape),
        }

    async def __aiter__(self):
        async for request_output in self._generator:
            for output in request_output.outputs:
                if output.routed_experts is not None:
                    self.routed_experts[output.index] = self._encode_routed_experts(output.routed_experts)
            yield request_output

    def post_process(self, response: ChatCompletionResponse):
        for choice in response.choices:
            if choice.index in self.routed_experts:
                choice.routed_experts = self.routed_experts[choice.index]


class OpenAIServingChatWithRoutedExperts(OpenAIServingChat):
    """OpenAI chat serving wrapper that preserves routed expert metadata."""

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation,
        tokenizer,
        request_metadata: RequestResponseMetadata,
        reasoning_parser: ReasoningParser | None = None,
    ) -> ErrorResponse | ChatCompletionResponse:
        if self.model_config.enable_return_routed_experts:
            capture = _RoutedExpertsCapture(result_generator)
            result_generator = capture
        else:
            capture = None

        response = await super().chat_completion_full_generator(
            request,
            result_generator,
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
            reasoning_parser,
        )

        if capture and isinstance(response, ChatCompletionResponse):
            capture.post_process(response)

        return response
