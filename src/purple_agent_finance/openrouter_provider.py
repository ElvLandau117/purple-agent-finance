"""
OpenRouter LLM Provider

Wraps the OpenAI SDK to use OpenRouter's API endpoint, enabling access to
multiple LLM providers (OpenAI, Anthropic, Google, Meta, etc.) through a
single API key.

Usage:
    Set OPENROUTER_API_KEY environment variable
    Use model names like: openrouter/openai/gpt-4o, openrouter/anthropic/claude-sonnet-4
"""

import json
import os
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageToolCall

from model_library.base import (
    LLMConfig,
    QueryResult,
    QueryResultMetadata,
    QueryResultCost,
    TextInput,
    RawResponse,
    ToolCall,
    ToolResult,
    ToolDefinition,
    InputItem,
)

from .logger import get_logger

logger = get_logger(__name__)


class OpenRouterLLM:
    """
    LLM implementation that uses OpenRouter API via the OpenAI SDK.

    OpenRouter provides access to various LLM providers through a unified API.
    This class wraps the OpenAI SDK and redirects requests to OpenRouter's endpoint.

    This is a duck-typed class that implements the minimal interface required
    by finance_agent.py (query method, _registry_key, and logger attributes).
    """

    def __init__(self, model_name: str, config: LLMConfig):
        """
        Initialize the OpenRouter LLM provider.

        Args:
            model_name: The model identifier (e.g., "openai/gpt-4o", "anthropic/claude-sonnet-4")
            config: LLM configuration (temperature, max_output_tokens, etc.)
        """
        self._registry_key = f"openrouter/{model_name}"
        self._model_name = model_name
        self._config = config
        self.logger = logger

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is required for OpenRouter models"
            )

        self._client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    def _convert_messages(
        self, input_items: list[InputItem]
    ) -> list[dict[str, Any]]:
        """
        Convert model_library input items to OpenAI message format.

        Args:
            input_items: List of InputItem (TextInput, RawResponse, ToolResult)

        Returns:
            List of messages in OpenAI format
        """
        messages = []

        for item in input_items:
            if isinstance(item, TextInput):
                messages.append({"role": "user", "content": item.text})

            elif isinstance(item, RawResponse):
                # RawResponse contains the raw assistant message in its response attribute
                messages.append(item.response)

            elif isinstance(item, ToolResult):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.tool_call.id,
                        "content": str(item.result),
                    }
                )

        return messages

    def _convert_tools(
        self, tools: list[ToolDefinition] | None
    ) -> list[dict[str, Any]] | None:
        """
        Convert model_library tool definitions to OpenAI format.

        Args:
            tools: List of ToolDefinition objects

        Returns:
            List of tools in OpenAI format, or None if no tools
        """
        if not tools:
            return None

        openai_tools = []
        for tool in tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.body.description,
                    "parameters": {
                        "type": "object",
                        "properties": tool.body.properties,
                        "required": tool.body.required,
                    },
                },
            }
            openai_tools.append(openai_tool)

        return openai_tools

    def _parse_tool_calls(
        self, tool_calls: list[ChatCompletionMessageToolCall] | None
    ) -> list[ToolCall]:
        """
        Parse OpenAI tool calls into model_library ToolCall objects.

        Args:
            tool_calls: List of tool calls from OpenAI response

        Returns:
            List of ToolCall objects
        """
        if not tool_calls:
            return []

        parsed_calls = []
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = tc.function.arguments

            parsed_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    args=args,
                )
            )

        return parsed_calls

    async def query(
        self,
        input: str | list[InputItem],
        tools: list[ToolDefinition] | None = None,
    ) -> QueryResult:
        """
        Query the OpenRouter API.

        Args:
            input: Either a string prompt or list of InputItem for multi-turn conversations
            tools: Optional list of tool definitions

        Returns:
            QueryResult with the model's response
        """
        # Convert input to messages
        if isinstance(input, str):
            messages = [{"role": "user", "content": input}]
            history: list[InputItem] = [TextInput(text=input)]
        else:
            messages = self._convert_messages(input)
            history = list(input)

        # Convert tools
        openai_tools = self._convert_tools(tools)

        # Build request parameters
        request_params: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
        }

        if self._config.temperature is not None:
            request_params["temperature"] = self._config.temperature

        if self._config.max_tokens is not None:
            request_params["max_tokens"] = self._config.max_tokens

        if openai_tools:
            request_params["tools"] = openai_tools

        # Log request
        self.logger.info(
            f"\033[1;35m[OPENROUTER]\033[0m Querying model: {self._model_name}"
        )

        # Make the API call
        response = await self._client.chat.completions.create(**request_params)

        # Extract response data
        choice = response.choices[0]
        message = choice.message
        output_text = message.content or ""
        tool_calls = self._parse_tool_calls(message.tool_calls)

        # Build raw response for history
        raw_message: dict[str, Any] = {
            "role": "assistant",
            "content": output_text,
        }
        if message.tool_calls:
            raw_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        raw_response = RawResponse(response=raw_message)
        history.append(raw_response)

        # Build metadata
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        # Estimate cost (OpenRouter provides this in headers, but we use estimates here)
        # Actual costs vary by model; these are rough estimates
        cost_per_1k_input = 0.005  # Default estimate
        cost_per_1k_output = 0.015  # Default estimate

        input_cost = (prompt_tokens / 1000) * cost_per_1k_input
        output_cost = (completion_tokens / 1000) * cost_per_1k_output

        metadata = QueryResultMetadata(
            in_tokens=prompt_tokens,
            out_tokens=completion_tokens,
            cost=QueryResultCost(
                input=input_cost,
                output=output_cost,
            ),
        )

        self.logger.info(
            f"\033[1;35m[OPENROUTER]\033[0m Response received: "
            f"{completion_tokens} tokens, {len(tool_calls)} tool calls"
        )

        return QueryResult(
            output_text=output_text,
            tool_calls=tool_calls,
            history=history,
            metadata=metadata,
            reasoning=None,
        )
