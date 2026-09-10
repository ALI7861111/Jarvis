from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import config

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Jarvis, a helpful assistant."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

llm = ChatAnthropic(model=config.MODEL, api_key=config.ANTHROPIC_API_KEY or None)
chain = prompt | llm


def main():
    history = []
    print("Jarvis chatbot. Type 'exit' to quit.")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break

        response = chain.invoke({"input": user_input, "history": history})
        print(f"Jarvis: {response.content}")

        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=response.content))


if __name__ == "__main__":
    main()
