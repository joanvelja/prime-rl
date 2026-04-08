import base64
from collections.abc import AsyncGenerator, AsyncIterator

import numpy as np
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest, ChatCompletionResponse
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.logger import init_logger
from vllm.outputs import RequestOutput

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


class OpenAIServingChatWithTokens(OpenAIServingChat):
    """OpenAI-compatible generate API that allows token-in and routed experts capture."""

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
        # We need to override the full_generator to be able to capture the routed experts
        # By default, VLLM does not save the routed experts into ChatCompletionResponse.choices, so we need to capture them manually
        # How this works:
        # 1. We create a custom generator that encapsulates the original result_generator in self._generator
        # 2. We override it's __aiter__ method to also capture the routed experts as an extra field in ChatCompletionResponse.choices
        # 3. We override the full_generator method to use the custom generator instead of the original one if expert routing is enabled
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
