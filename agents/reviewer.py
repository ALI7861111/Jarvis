from autogen_agentchat.agents import AssistantAgent
from autogen_core.model_context import BufferedChatCompletionContext

from model_client import get_model_client

_SYSTEM_MESSAGE = """
You are Jarvis's reviewer. 
Your task is to review the work completed by other agents 
and provide feedback on whether the task is complete or needs further work.
"""


def build_reviewer() -> AssistantAgent:
    return AssistantAgent(
        name="reviewer",
        model_client=get_model_client(),
        model_context=BufferedChatCompletionContext(buffer_size=10),
        description="Reviews the task completed by other agents and provides feedback on whether the task is complete or needs further work.",
        system_message=_SYSTEM_MESSAGE,
    )
