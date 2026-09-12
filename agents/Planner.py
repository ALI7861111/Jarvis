import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

PLANNER_SYSTEM_PROMPT = (
    "You are a planning agent. Break down a complex task into a short, "
    "ordered list of simpler subtasks that can each be accomplished on "
    "their own. Reply with ONLY a JSON array of strings -- no prose, no "
    "markdown fences."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", PLANNER_SYSTEM_PROMPT),
        ("human", "{task}"),
    ]
)


@tool
def plan(llm, task: str) -> list[str]:
    """This tool breaks a complex task down into a list of simpler subtasks.

    Args:
        llm: A LangChain chat model instance (must implement invoke).
        task: The complex task to break down.

    Returns:
        An ordered list of subtask descriptions.
    """
    chain = prompt | llm
    response = chain.invoke({"task": task})
    content = response.content.strip()

    try:
        steps = json.loads(content)
        if isinstance(steps, list) and all(isinstance(s, str) for s in steps):
            return steps
    except json.JSONDecodeError:
        pass

    # Fallback if the model didn't return valid JSON: one step per non-empty line.
    return [
        line.strip("-*0123456789. \t") for line in content.splitlines() if line.strip()
    ]
