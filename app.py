import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


load_dotenv()

st.title("LangChain Chatbot")
st.header("Conversational Model with Memory", divider="red")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()


# Model or LLM
model = ChatOpenAI(model="gpt-4o-mini")
# model = ChatGroq(model="llama-3.3-70b-versatile")

# Chat Prompt Template
prompt_template = ChatPromptTemplate(
    [
        SystemMessage(content="You're a personal assistant that talks like a pirate."),
        MessagesPlaceholder(variable_name="messages")
    ]
)


# Messages Trimmer
trimmer = trim_messages(
    strategy="last",
    max_tokens=50,
    token_counter=len,
    include_system=True
)


# define a new graph
workflow = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
    trimmed_messages = trimmer.invoke(state["messages"])
    chat_template_messages = prompt_template.invoke(trimmed_messages)
    completion = model.invoke(chat_template_messages)
    return {"messages": completion}

# Create a new Node
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Runnable Graph
runnable_graph = workflow.compile(checkpointer=st.session_state.memory)

# Graph config
config = {"configurable": {"thread_id": 1}}


if prompt := st.chat_input():
    
    # Previous Messages
    for message in st.session_state.chat_history:
       with st.chat_message("human" if isinstance(message, HumanMessage) else "ai"):
           st.write(message.content)
           
    # Appends user prompt to chat history
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    
    # New Message
    with st.chat_message("human"):
        st.write(prompt)
        
    
    # New AI Message
    full_response = ""
    with st.chat_message("assistant"):
        placeholder = st.empty()
        
    
    for chunk, metadata in runnable_graph.stream(
        {"messages": HumanMessage(content=prompt)}, config, stream_mode="messages"):
        full_response += chunk.content
        placeholder.write(full_response)     
    

    st.session_state.chat_history.append(AIMessage(content=full_response))

    print(full_response)

