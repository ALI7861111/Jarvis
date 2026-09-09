from autogen_agentchat.agents import AssistantAgent
from autogen_core.model_context import BufferedChatCompletionContext

from model_client import get_model_client


def build_chat_agent() -> AssistantAgent:
    return AssistantAgent(
        name="llm_chat",
        model_client=get_model_client(),
        model_context=BufferedChatCompletionContext(buffer_size=40),
        description="Jarvis's conversational voice for greetings, small talk, and general "
        "questions that don't need research, coding, or a step-by-step plan.",
        system_message="""
        You are Jarvis, a helpful conversational assistant. Answer directly
        and naturally -- you are handling this turn alone, with no other
        agents involved, so there is no need to hand off or announce
        completion.
        """,
    )
