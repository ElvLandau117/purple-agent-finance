import os

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from model_library.base import LLMConfig

from purple_agent_finance.get_agent import Parameters, get_agent

from messenger import Messenger


class Agent:
    def __init__(self):
        self.messenger = Messenger()
        self._finance_agent = None
        self._finance_agent_error: str | None = None

    def _ensure_finance_agent(self):
        if self._finance_agent is not None or self._finance_agent_error is not None:
            return

        model_name = os.environ.get("FINANCE_MODEL", "openai/gpt-4o-mini")
        max_turns = int(os.environ.get("FINANCE_MAX_TURNS", "8"))

        tools_raw = os.environ.get(
            "FINANCE_TOOLS",
            "google_web_search,retrieve_information,parse_html_page,edgar_search",
        )
        tools = [tool.strip() for tool in tools_raw.split(",") if tool.strip()]

        max_output_tokens = int(os.environ.get("FINANCE_MAX_OUTPUT_TOKENS", "2048"))
        temperature = float(os.environ.get("FINANCE_TEMPERATURE", "0"))

        try:
            self._finance_agent = get_agent(
                Parameters(
                    model_name=model_name,
                    max_turns=max_turns,
                    tools=tools,
                    llm_config=LLMConfig(
                        max_output_tokens=max_output_tokens,
                        temperature=temperature,
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001 - runtime fallback
            self._finance_agent_error = str(exc)

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Implement your agent logic here.

        Args:
            message: The incoming message
            updater: Report progress (update_status) and results (add_artifact)

        Use self.messenger.talk_to_agent(message, url) to call other agents.
        """
        input_text = get_message_text(message)
        await updater.update_status(TaskState.working, new_agent_text_message("Thinking..."))

        self._ensure_finance_agent()
        if self._finance_agent is None:
            answer_text = (
                "Finance agent is running in fallback mode (missing/invalid model credentials). "
                f"Details: {self._finance_agent_error}"
            )
        else:
            answer_text, _ = await self._finance_agent.run(input_text)

        await updater.add_artifact(parts=[Part(root=TextPart(text=answer_text))], name="Answer")
