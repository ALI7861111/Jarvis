

from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 

prompt = ChatPromptTemplate.from_messages([ 
    ("system", "You are Jarvis, a helpful AI assistant."
                "You have many special tools/agents at your disposal to"
                "help you answer questions and complete tasks."
                "You can use the provided tools for complex tasks."), 
    MessagesPlaceholder("history"),
    ("human", "{input}") ])

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

        if response.tool_calls:
            tool_name = response.tool_invocation.name
            tool_input = response.tool_invocation.input
            print(f"Tool '{tool_name}' invoked with input: {tool_input}")
            # Here you would call the actual tool function and get the result
            # For now, we just simulate a tool response
            tool_response = f"Simulated response from tool '{tool_name}'"
            print(f"Tool response: {tool_response}")
            # You might want to add the tool response to the history as well
            history.append(ToolMessage(tool_name=tool_name, content=tool_response))
            break
        print(f"AI: {response.content}")

        history.append( HumanMessage(content=user_input) ) 
        history.append( AIMessage(content=response.content) )