from langchain.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are Jarvis, a helpful AI assistant."
            "You have many special tools/agents at your disposal to"
            "help you answer questions and complete tasks."
            "You should look at the tools/agents available to you and use them when appropriate.",
        ),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)


def chat_loop(llm, tools=None):
    """
    This function runs a chat loop with the given language model.
    It prompts the user for input,
    sends the input to the model, and prints the model's response.
    The loop continues until the user types 'exit'.

    input:
        llm: A language model instance that has an 'invoke' method to process messages.
    """
    print("Enter 'exit' to quit the chat.")

    llm = llm.bind_tools(tools) if tools else llm
    tools_by_name = {t.name: t for t in tools} if tools else {}

    history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        messages = prompt.invoke({"input": user_input, "history": history}).to_messages()

        while True:
            response = llm.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                break

            for tool_call in response.tool_calls:
                print(f"Tool call detected: {tool_call['name']} with args {tool_call['args']}")
                tool = tools_by_name[tool_call["name"]]
                tool_message = tool.invoke(tool_call)
                messages.append(tool_message)

        print(f"AI: {response.content}")

        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=response.content))
