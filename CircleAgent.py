from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import Tool
# importting function from the some py files
from startup_chain import get_startup_data_chain
from legal_chain import get_legal_data_chain
from helper import get_llm, get_embeddings_model


llm=get_llm()
embedding=get_embeddings_model()

startup_data_rag_chain=get_startup_data_chain(llm,embedding)
legal_data_rag_chain=get_legal_data_chain(llm,embedding)


# Set Up ReAct Agent with Document Store Retriever
# Load the ReAct Docstore Prompt
react_docstore_prompt = hub.pull("hwchase17/react")


def invoke_startup_data_rag_chain(input, **kwargs):
    return startup_data_rag_chain.invoke({"input": input, "chat_history": kwargs.get("chat_history", [])})

def invoke_legal_data_rag_chain(input, **kwargs):
    return legal_data_rag_chain.invoke({"input": input, "chat_history": kwargs.get("chat_history", [])})

tools = [
    Tool(
        name="Startup Data Answer Question",
        func=invoke_startup_data_rag_chain,
        description="useful for when you need to answer questions about the context",
    ),
    Tool(
        name="Legal Data Answer Question",
        func=invoke_legal_data_rag_chain,
        description="useful for when you need to answer questions about the context",
    )
]


# Create the ReAct Agent with document store retriever
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_docstore_prompt,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, handle_parsing_errors=True, verbose=True,
)




chat_history = []
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    response = agent_executor.invoke({"input": query, "chat_history": chat_history})
    print(f"AI: {response['output']}")

    # Update history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response["output"]))
