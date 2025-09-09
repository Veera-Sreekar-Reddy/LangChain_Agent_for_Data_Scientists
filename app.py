import streamlit as st
import pandas as pd
from agent import create_agent
from retriever import DataRetriever
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.agents import Tool

st.title("📊 LangChain + Ollama (CodeLLaMA) + RAG Data Science Assistant")

# -----------------------------
# Session State Initialization
# -----------------------------
if "agent" not in st.session_state:
    st.session_state.agent, st.session_state.ds_tools = create_agent()
if "retriever_engine" not in st.session_state:
    st.session_state.retriever_engine = DataRetriever()
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "retriever_tool" not in st.session_state:
    st.session_state.retriever_tool = None
if "df" not in st.session_state:
    st.session_state.df = None

# -----------------------------
# Sidebar: Upload CSV
# -----------------------------
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    # Save file to session state
    st.session_state.uploaded_file = uploaded_file
    if st.session_state.df is None:
        with open("uploaded.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.ds_tools.load_data("uploaded.csv")
        st.session_state.df = st.session_state.ds_tools.get_dataframe()
        st.success("✅ Dataset loaded into DataScienceTools.")

        # Index dataset for RAG
        st.session_state.retriever_engine.index_dataframe(st.session_state.df)
        st.info("🔍 Dataset indexed into vector store.")

        # Build retriever tool once
        if st.session_state.retriever_tool is None:
            st.session_state.retriever_tool = RetrievalQA.from_chain_type(
                llm=Ollama(model="codellama"),
                retriever=st.session_state.retriever_engine.get_retriever(),
                return_source_documents=True
            )
            st.session_state.agent.tools.append(
                Tool(
                    name="Query Dataset",
                    func=lambda query: st.session_state.retriever_tool.run(query),
                    description="Ask dataset questions after it's indexed"
                )
            )

# -----------------------------
# Display Dataset Overview
# -----------------------------
if st.session_state.df is not None:
    st.subheader("Dataset Overview")
    explore_output = st.session_state.ds_tools.explore_data()
    st.text(explore_output)

    # Target column selection for model suggestion
    target_column = st.selectbox(
        "Select target column for model suggestion",
        st.session_state.df.columns,
        index=len(st.session_state.df.columns) - 1
    )
    if st.button("Suggest Model"):
        model_output = st.session_state.ds_tools.suggest_model(target_column)
        st.subheader("Suggested Model")
        st.text(model_output)

# -----------------------------
# Always display the input box
# -----------------------------
st.subheader("Ask the Assistant")
query = st.text_input(
    "Type a query for the agent, e.g., 'Clean data', 'Show correlation between X an Y columns'"
)

if st.button("Run Query"):
    if query:
        response = st.session_state.agent.run(query)
        st.text_area("Response", response, height=300)
    else:
        st.warning("Please enter a query.")
