import streamlit as st
import tempfile
from tools import DataScienceTools
from agent import create_agent

st.set_page_config(page_title="Local Data Scientist Assistant", layout="wide")

st.title("🧠 Local Data Scientist Assistant (Ollama + LangChain + RAG)")

ds_tools = DataScienceTools()
agent = create_agent(ds_tools)

st.sidebar.header("📂 Upload CSV")
csv_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if csv_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(csv_file.getbuffer())
        tmp_path = tmp.name

    load_msg = ds_tools.load_csv(tmp_path)
    st.sidebar.success(load_msg)

    if st.sidebar.button("🔍 Setup RAG Index"):
        rag_msg = ds_tools.setup_rag()
        st.sidebar.success(rag_msg)

st.subheader("💬 Ask a Question in English")
query = st.text_input("Enter your question:")

if st.button("Run"):
    if query:
        output = agent.run(query)
        st.write(output)
    else:
        st.warning("Please enter a question.")
