import streamlit as st
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import Tool
from startup_chain import get_startup_data_chain
from legal_chain import get_legal_data_chain
from helper import get_llm, get_embeddings_model

import sys
import os

try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # Continue with system sqlite3 (may not work if version is too old)
    pass

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
def main():
    st.set_page_config(
        page_title="LangChain Agent Assistant",
        page_icon="🤖",
        layout="wide",
    )

    # Custom CSS for better styling - with dark mode support
    st.markdown("""
    <style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: row;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #f0f2f6;
        color: #262730; /* Dark text for light background */
    }
    .chat-message.bot {
        background-color: #e6f3f7;
        color: #262730; /* Dark text for light background */
    }
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .chat-message.user {
            background-color: #1E1E1E;
            color: #FFFFFF; /* Light text for dark background */
        }
        .chat-message.bot {
            background-color: #2D2D2D;
            color: #FFFFFF; /* Light text for dark background */
        }
    }
    .chat-message .avatar {
        width: 45px;
        height: 45px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
    .chat-message .message {
        flex-grow: 1;
    }
    .stTextInput {
        padding-bottom: 1rem;
    }
    .stMarkdown, .stText {
        padding-top: 0 !important;
    }
    .main-header {
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: #1e3a8a;
    }
    /* Dark mode support for header */
    @media (prefers-color-scheme: dark) {
        .main-header {
            background-color: #2C3E50;
            color: #FFFFFF;
        }
    }
    .main-header h1 {
        margin: 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # App header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 LangChain Agent Assistant</h1>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.title("About")
        st.markdown("""
        This is an AI assistant powered by LangChain that can access:
        - **Startup Data** - Information about startups and business
        - **Legal Data** - Legal information and documents
        
        The agent uses a ReAct framework to reason through complex questions
        and access the appropriate knowledge when needed.
        """)
        
        st.divider()
        
        if st.button("Clear Conversation", key="clear", type="primary"):
            st.session_state.chat_history = []
            st.session_state.messages = []
            st.rerun()

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize messages for the UI
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize LLM, embeddings, and chains if not already in session state
    if "agent_executor" not in st.session_state:
        with st.spinner("Initializing AI models and knowledge bases..."):
            llm = get_llm()
            embedding = get_embeddings_model()
            
            startup_data_rag_chain = get_startup_data_chain(llm, embedding)
            legal_data_rag_chain = get_legal_data_chain(llm, embedding)
            
            # Define tool functions
            def invoke_startup_data_rag_chain(input, **kwargs):
                return startup_data_rag_chain.invoke({"input": input, "chat_history": kwargs.get("chat_history", [])})

            def invoke_legal_data_rag_chain(input, **kwargs):
                return legal_data_rag_chain.invoke({"input": input, "chat_history": kwargs.get("chat_history", [])})

            tools = [
                Tool(
                    name="Startup Data Answer Question",
                    func=invoke_startup_data_rag_chain,
                    description="useful for when you need to answer questions about startups, business data, or company information",
                ),
                Tool(
                    name="Legal Data Answer Question",
                    func=invoke_legal_data_rag_chain,
                    description="useful for when you need to answer questions about legal information, regulations, or legal documents",
                )
            ]

            # Create the ReAct Agent
            react_docstore_prompt = hub.pull("hwchase17/react")
            agent = create_react_agent(
                llm=llm,
                tools=tools,
                prompt=react_docstore_prompt,
            )

            st.session_state.agent_executor = AgentExecutor.from_agent_and_tools(
                agent=agent, 
                tools=tools, 
                handle_parsing_errors=True, 
                verbose=True,
            )
        
        st.success("✅ Agent initialized successfully!")

    # Display chat messages with explicit text color styling
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user">
                <img class="avatar" src="https://api.dicebear.com/7.x/personas/svg?seed=user" alt="User Avatar">
                <div class="message">{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot">
                <img class="avatar" src="https://api.dicebear.com/7.x/bottts/svg?seed=assistant" alt="Bot Avatar">
                <div class="message">{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)

    # Chat input
    query = st.chat_input("Ask me about startup or legal information...")
    
    if query:
        # Add user message to UI and session state
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Show user message right away
        st.markdown(f"""
        <div class="chat-message user">
            <img class="avatar" src="https://api.dicebear.com/7.x/personas/svg?seed=user" alt="User Avatar">
            <div class="message">{query}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Get AI response with a spinner
        with st.spinner("Thinking..."):
            # Invoke the agent
            response = st.session_state.agent_executor.invoke({
                "input": query, 
                "chat_history": st.session_state.chat_history
            })
            
            # Update chat history for context
            st.session_state.chat_history.append(HumanMessage(content=query))
            st.session_state.chat_history.append(AIMessage(content=response["output"]))
            
            # Add AI response to UI
            st.session_state.messages.append({"role": "assistant", "content": response["output"]})
        
        # Show assistant message
        st.markdown(f"""
        <div class="chat-message bot">
            <img class="avatar" src="https://api.dicebear.com/7.x/bottts/svg?seed=assistant" alt="Bot Avatar">
            <div class="message">{response["output"]}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Required for the chat_input to reappear properly
        st.rerun()

if __name__ == "__main__":
    main()
