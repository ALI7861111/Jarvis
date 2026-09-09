from autogen_agentchat.agents import AssistantAgent
from autogen_core.model_context import BufferedChatCompletionContext

from model_client import get_model_client
from tools.common_tools import read_file, run_shell, write_file


def build_planner() -> AssistantAgent:
    return AssistantAgent(
        name="planner",
        model_client=get_model_client(),
        model_context=BufferedChatCompletionContext(buffer_size=40),
        description="Breaks a task down into an ordered set of subtasks and steps; only "
        "needed for multi-step planning, not simple questions or chit-chat.",
        system_message="""
        You are a Special Agent Planner. Your responsibilities include:
        1. Analyzing tasks and breaking them down into smaller, manageable subtasks.
        2. Creating a clear and organized plan to accomplish the given task.
        3. Prioritizing tasks based on their importance and urgency.
        4. Providing a step-by-step guide to complete the task efficiently.

        Reply with TERMINATE at the end of your message once you've fully
        answered and no other agent needs to act.
        """,
    )
