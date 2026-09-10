

from langchain.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 

prompt = ChatPromptTemplate.from_messages([ 
    ("system", "You are Jarvis, a helpful AI assistant."), 
    MessagesPlaceholder("history"),
    ("human", "{input}") ])

def chat_loop(llm):
    """
    This function runs a chat loop with the given language model. It prompts the user for input, 
    sends the input to the model, and prints the model's response.
    The loop continues until the user types 'exit'.
    
    input: 
        llm: A language model instance that has an 'invoke' method to process messages.
    """
    print("Enter 'exit' to quit the chat.")
    while True:

        history = []

        user_input = input("You: ")

        messages = prompt.invoke({
            "input": user_input,
            "history": history
        })

        if user_input.lower() == "exit":
            break
        response = llm.invoke(messages)
        print(f"AI: {response.content}")

        history.append( HumanMessage(content=user_input) ) 
        history.append( AIMessage(content=response.content) )