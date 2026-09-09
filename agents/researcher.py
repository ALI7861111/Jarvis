from autogen_agentchat.agents import AssistantAgent
from autogen_core.model_context import BufferedChatCompletionContext

from model_client import get_model_client
from tools.common_tools import read_file
from tools.arxiv_search import arxiv_search, arxiv_search_and_download
from tools.web_search import web_search
from tools.summarize_info import information_summarizer

def build_researcher() -> AssistantAgent:
    return AssistantAgent(
        name="researcher",
        model_client=get_model_client(),
        tools=[web_search, arxiv_search_and_download, arxiv_search, read_file, information_summarizer],
        model_context=BufferedChatCompletionContext(buffer_size=40),
        description="Looks up facts, current events, and academic papers by searching the web "
        "or arXiv; only needed when the task requires external/up-to-date information.",
        system_message="""
        You are the Special Agent Researcher. Use web_search for general questions, current events,
        or anything not specifically about academic papers (e.g. dates, facts, news, how-tos).
        Use arxiv_search / arxiv_search_and_download only when the user specifically wants
        academic/research papers from arXiv. Call summarization tools to summarize results
        or look for specific information in them.

        When a paper needs to be read or summarized after arxiv_search_and_download,
        pass the exact "local_path" string from that tool's result to
        information_summarizer or read_file -- never invent or guess a file path.
        If "local_path" is null, say the PDF could not be downloaded instead of
        fabricating a path.

        Reply with TERMINATE at the end of your message once you've fully
        answered and no other agent needs to act.
        """,
    )


