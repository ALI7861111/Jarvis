from autogen_agentchat.agents import AssistantAgent
from autogen_core.model_context import BufferedChatCompletionContext

from model_client import get_model_client
from tools.common_tools import read_file, run_shell, write_file


def build_coder() -> AssistantAgent:
    return AssistantAgent(
        name="coder",
        model_client=get_model_client(),
        tools=[read_file, write_file, run_shell],
        reflect_on_tool_use=True,
        model_context=BufferedChatCompletionContext(buffer_size=40),
        description="Reads/writes code files and runs shell commands; only needed for "
        "programming tasks or answering questions about code.",
        system_message="""You are a Special Agent Coder.
        Your task is to read and write programming files,and run shell commands.
        you can also answer questions about code and programming concepts.
        You can read write programming files, run shell commands, and ask questions about code and programming requirements.

        Reply with TERMINATE at the end of your message once you've fully
        answered and no other agent needs to act.
        """,
    )
