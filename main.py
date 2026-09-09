import argparse
import asyncio
import re

from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console

from agents.chat import build_chat_agent
from agents.coder import build_coder
from agents.planner import build_planner
from agents.researcher import build_researcher
from agents.router import build_router
from model_client import get_model_client


selector_prompt = """You are the operations team Manager. The following roles are available:
{roles}

Read the conversation below, then select the next role from {participants} to
respond. Reply with the role's name exactly as given above -- nothing else.

Choose researcher for:
- web research
- finding sources
- comparing information
- current information

Choose coder for:
- writing code
- debugging
- software implementation
- programming questions

Choose planner for:
- project planning
- task decomposition
- roadmaps
- multi-step plans

Do not select multiple agents.

{history}

Read the above conversation. Then select the next role from {participants} to
play. Only return the role.
"""


def build_operations_team() -> SelectorGroupChat:
    researcher = build_researcher()
    coder = build_coder()
    planner = build_planner()

    return SelectorGroupChat(
        [researcher, coder, planner],
        model_client=get_model_client(),
        selector_prompt=selector_prompt,
        # TERMINATE ends a turn as soon as an agent says it's done; the
        # message cap is a hard safety net in case the model never says it,
        # so one user message can't hang the session forever.
        termination_condition=TextMentionTermination("TERMINATE") | MaxMessageTermination(8),
    )


def _final_reply_text(messages) -> str:
    # The message that satisfies TextMentionTermination is sometimes just
    # the bare word "TERMINATE" with no real content -- fall back to the
    # last message that has substance once that marker is stripped out.
    for msg in reversed(messages):
        cleaned = re.sub(r"\bTERMINATE\b", "", str(msg.content), flags=re.IGNORECASE).strip()
        if cleaned:
            return cleaned
    return "(no response)"


async def _run_turn(runner, context: list[TextMessage], verbose: bool) -> TaskResult:
    if verbose:
        return await Console(runner.run_stream(task=context))

    result: TaskResult | None = None
    async for message in runner.run_stream(task=context):
        if isinstance(message, TaskResult):
            result = message
    assert result is not None
    return result


async def _classify(router, context: list[TextMessage]) -> str:
    # The router is a plain, tool-less agent making a single one-word call --
    # cheap compared to spinning up the operations team, so it's fine to run
    # on every turn even for chit-chat.
    result = await router.run(task=context)
    verdict = str(result.messages[-1].content).strip().upper()
    return "OPERATION" if "OPERATION" in verdict else "CHAT"


async def main(verbose: bool) -> None:
    router = build_router()
    chat_agent = build_chat_agent()
    operations_team = build_operations_team()
    history: dict[int, dict[str, str]] = {}
    try:
        while True:
            task = input('Please provide a input (or "exit" to quit): ')
            if task.strip().lower() == "exit":
                break
            context = [
                message
                for turn in history.values()
                for message in (
                    TextMessage(content=turn["input"], source="user"),
                    TextMessage(content=turn["response"], source="team"),
                )
            ]
            context.append(TextMessage(content=task, source="user"))

            try:
                route = await _classify(router, context)
                runner = operations_team if route == "OPERATION" else chat_agent
                response = await _run_turn(runner, context, verbose)
            except Exception as e:
                # A single failed turn (a tool error, a flaky model call, ...)
                # shouldn't kill the whole session and its conversation memory.
                print(f"\nTurn failed: {e}")
                continue
            reply = _final_reply_text(response.messages)
            history[len(history)] = {"input": task, "response": reply}
            print(f"\nResponse: {reply}")
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show intermediate agent messages/events instead of just the final response.",
    )
    args = parser.parse_args()
    asyncio.run(main(args.verbose))
