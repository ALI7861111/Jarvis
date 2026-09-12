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


def _build_ollama() -> OllamaChatCompletionClient:
    return OllamaChatCompletionClient(
        model=config.OLLAMA_MODEL, host=config.OLLAMA_HOST
    )


def _prompt_for_new_key():
    print(
        "No local Ollama server detected and no ANTHROPIC_API_KEY / OPENAI_API_KEY found."
    )
    print("  1) Anthropic (Claude)")
    print("  2) OpenAI (ChatGPT)")
    choice = input("Choose a provider [1/2]: ").strip()
    if choice == "2":
        key = getpass.getpass("Enter your OpenAI API key: ").strip()
        return _build_openai(key), "OpenAI (ChatGPT)", config.OPENAI_MODEL
    key = getpass.getpass("Enter your Anthropic API key: ").strip()
    return _build_anthropic(key), "Anthropic (Claude)", config.MODEL


def _choose_provider():
    # Each option is (label, model_name, builder) -- builder is only called
    # for the option the user actually picks (or the sole one available).
    options = []
    if _ollama_running():
        options.append(("Ollama (local)", config.OLLAMA_MODEL, _build_ollama))
    if config.ANTHROPIC_API_KEY:
        options.append(
            (
                "Anthropic (Claude)",
                config.MODEL,
                lambda: _build_anthropic(config.ANTHROPIC_API_KEY),
            )
        )
    if config.OPENAI_API_KEY:
        options.append(
            (
                "OpenAI (ChatGPT)",
                config.OPENAI_MODEL,
                lambda: _build_openai(config.OPENAI_API_KEY),
            )
        )

    if not options:
        return _prompt_for_new_key()

    if len(options) == 1:
        label, model_name, builder = options[0]
        return builder(), label, model_name

    print("Multiple model providers are available:")
    for i, (label, model_name, _) in enumerate(options, start=1):
        print(f"  {i}) {label} -- model: {model_name}")
    choice = input(f"Choose a provider [1-{len(options)}]: ").strip()
    try:
        index = int(choice) - 1
        if not 0 <= index < len(options):
            raise ValueError
    except ValueError:
        print("Invalid choice, defaulting to option 1.")
        index = 0
    label, model_name, builder = options[index]
    return builder(), label, model_name


# Resolved once per run -- every agent calls get_model_client() to build
# itself, and they should all share the same provider/model and instance
# rather than re-checking Ollama or re-prompting the user repeatedly.
_cached_client = None


def get_model_client():
    global _cached_client
    if _cached_client is None:
        client, label, model_name = _choose_provider()
        print(f"[Jarvis] Using {label} -- model: {model_name}")
        _cached_client = client
    return _cached_client
