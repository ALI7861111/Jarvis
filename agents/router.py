from autogen_agentchat.agents import AssistantAgent
from autogen_core.model_context import BufferedChatCompletionContext

from model_client import get_model_client

_SYSTEM_MESSAGE = """
You are Jarvis's router. Decide whether the latest user message can be
handled as plain conversation, or needs the operations team (research,
coding, or multi-step planning).

Reply CHAT for greetings, small talk, opinions, explanations, or anything
answerable from general knowledge without tools.

Reply OPERATION when the task requires:
- web or arXiv research / current or external information
- reading, writing, or running code, files, or shell commands
- breaking a task down into a multi-step plan

Reply with exactly one word, CHAT or OPERATION -- nothing else, no
punctuation, no explanation.
"""


def build_router() -> AssistantAgent:
    return AssistantAgent(
        name="router",
        model_client=get_model_client(),
        model_context=BufferedChatCompletionContext(buffer_size=10),
        description="Classifies a task as CHAT or OPERATION.",
        system_message=_SYSTEM_MESSAGE,
    )
