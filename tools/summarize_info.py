from autogen_agentchat.agents import AssistantAgent

from model_client import get_model_client
from tools.common_tools import read_file

_SYSTEM_MESSAGE = """
You are an information summarization assistant.

- If the input looks like a file path rather than content, use the
  read_file tool to load it first.
- Summarize the content accurately and concisely.
- Extract important facts, key points, and relevant details.
- If a specific question is provided, answer it using only the source
  content.
- Do not invent information that is not present in the source.
- If the requested information is not present, clearly say so.
"""


async def information_summarizer(source: str, question: str = "") -> str:
    """Summarize text or a file, or answer a specific question about it.

    Args:
        source: Either the raw text to summarize, or a path to a file
            whose contents should be read and summarized.
        question: Optional specific question to answer using only the
            source content, instead of a general summary.
    """
    agent = AssistantAgent(
        name="summarizer",
        model_client=get_model_client(),
        tools=[read_file],
        reflect_on_tool_use=True,
        system_message=_SYSTEM_MESSAGE,
    )
    task = f"Question: {question}\n\nSource:\n{source}" if question else f"Summarize the following:\n\n{source}"
    result = await agent.run(task=task)
    return result.messages[-1].content
