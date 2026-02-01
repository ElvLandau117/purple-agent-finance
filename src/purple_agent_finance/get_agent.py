from .finance_agent import Agent
from .tools import EDGARSearch, GoogleWebSearch, ParseHtmlPage, RetrieveInformation
from .openrouter_provider import OpenRouterLLM
from model_library.registry_utils import get_registry_model
from model_library.base import LLMConfig

from dataclasses import dataclass
from typing import List

OPENROUTER_PREFIX = "openrouter/"


@dataclass
class Parameters:
    model_name: str
    max_turns: int
    tools: List[str]
    llm_config: LLMConfig


def get_agent(parameters: Parameters) -> Agent:
    """Helper method to instantiate an agent with the given parameters"""
    available_tools = {
        "google_web_search": GoogleWebSearch,
        "retrieve_information": RetrieveInformation,
        "parse_html_page": ParseHtmlPage,
        "edgar_search": EDGARSearch,
    }

    selected_tools = {}
    for tool in parameters.tools:
        if tool not in available_tools:
            raise Exception(
                f"Tool {tool} not found in tools. Available tools: {available_tools.keys()}"
            )
        selected_tools[tool] = available_tools[tool]()

    # Detect OpenRouter models and use appropriate provider
    if parameters.model_name.startswith(OPENROUTER_PREFIX):
        # Extract the model ID after "openrouter/" prefix
        # e.g., "openrouter/openai/gpt-4o" -> "openai/gpt-4o"
        model_id = parameters.model_name[len(OPENROUTER_PREFIX):]
        model = OpenRouterLLM(model_id, parameters.llm_config)
    else:
        model = get_registry_model(parameters.model_name, parameters.llm_config)

    agent = Agent(tools=selected_tools, llm=model, max_turns=parameters.max_turns)

    return agent
