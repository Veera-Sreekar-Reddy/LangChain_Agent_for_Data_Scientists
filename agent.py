from langchain.agents import Tool, initialize_agent
from langchain_community.llms import Ollama

def create_agent(ds_tools):
    llm = Ollama(model="codellama")

    tool_list = [
        Tool(
            name="Setup RAG",
            func=ds_tools.setup_rag,
            description="Create a RAG index from loaded CSV"
        ),
        Tool(
            name="Search Data",
            func=ds_tools.search_data,
            description="Search CSV data using RAG"
        ),
        Tool(
            name="AutoML",
            func=ds_tools.auto_ml,
            description="Automatically detect ML problem and train the best model"
        ),
    ]

    agent = initialize_agent(
        tools=tool_list,
        llm=llm,
        agent="zero-shot-react-description",
        handle_parsing_errors=True
    )

    return agent
