import getpass
import urllib.error
import urllib.request

from autogen_core.models import ModelFamily
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient

import config

# The installed autogen-ext model registry may not know newer Claude model
# names, so this is passed explicitly rather than relying on lookup by name.
ANTHROPIC_MODEL_INFO = {
    "vision": True,
    "function_calling": True,
    "json_output": True,
    "family": ModelFamily.CLAUDE_3_5_SONNET,
    "structured_output": False,
    "multiple_system_messages": False,
}


def _ollama_running() -> bool:
    try:
        urllib.request.urlopen(f"{config.OLLAMA_HOST}/api/tags", timeout=0.5)
        return True
    except (urllib.error.URLError, OSError):
        return False


def _build_anthropic(api_key: str) -> AnthropicChatCompletionClient:
    return AnthropicChatCompletionClient(
        model=config.MODEL,
        api_key=api_key,
        model_info=ANTHROPIC_MODEL_INFO,
    )


def _build_openai(api_key: str) -> OpenAIChatCompletionClient:
    return OpenAIChatCompletionClient(model=config.OPENAI_MODEL, api_key=api_key)


def _prompt_for_provider() -> OpenAIChatCompletionClient | AnthropicChatCompletionClient:
    print("No local Ollama server detected and no ANTHROPIC_API_KEY / OPENAI_API_KEY found.")
    print("  1) Anthropic (Claude)")
    print("  2) OpenAI (ChatGPT)")
    choice = input("Choose a provider [1/2]: ").strip()
    if choice == "2":
        key = getpass.getpass("Enter your OpenAI API key: ").strip()
        return _build_openai(key)
    key = getpass.getpass("Enter your Anthropic API key: ").strip()
    return _build_anthropic(key)


def get_model_client():
    if _ollama_running():
        return OllamaChatCompletionClient(model=config.OLLAMA_MODEL, host=config.OLLAMA_HOST)

    if config.ANTHROPIC_API_KEY:
        return _build_anthropic(config.ANTHROPIC_API_KEY)

    if config.OPENAI_API_KEY:
        return _build_openai(config.OPENAI_API_KEY)

    return _prompt_for_provider()
