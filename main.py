import getpass
import urllib.error
import urllib.request

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv
from core.chat import chat_loop
from tools.common_tools import read_file, write_file, run_shell
from agents.planner import plan
load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are Jarvis, a helpful assistant."),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)


def main():
    input_text = input("""
                        Please Select Model:
                        1) Anthropic (Claude)      
                        2) OpenAI (ChatGPT)
                        3) Ollama (LLaMA)
                        """)

    if input_text == "1":
        llm = ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL"), api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    if input_text == "2":
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL"), api_key=os.getenv("OPENAI_API_KEY")
        )
    if input_text == "3":
        llm = ChatOllama(model=os.getenv("OLLAMA_MODEL"), host=os.getenv("OLLAMA_HOST"))

    chat_loop(llm, tools=[read_file, write_file, run_shell, plan])


if __name__ == "__main__":
    main()
