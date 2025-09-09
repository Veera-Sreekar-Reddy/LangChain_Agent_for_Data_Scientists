from langchain.agents import initialize_agent, Tool
from langchain_community.llms import Ollama
from tools import DataScienceTools

def create_agent():
    ds_tools = DataScienceTools()

    tools = [
        Tool(
            name="Load Data",
            func=lambda query: ds_tools.load_data(query),
            description="Load CSV dataset by providing file path"
        ),
        Tool(
            name="Explore Data",
            func=lambda query: ds_tools.explore_data(),
            description="Summarize dataset columns and first few rows"
        ),
        Tool(
            name="Clean Data",
            func=lambda query: ds_tools.clean_data(),
            description="Clean dataset (handle missing values, encode categoricals)"
        ),
        Tool(
            name="Suggest Model",
            func=lambda query: ds_tools.suggest_model(query.strip()),
            description="Suggest ML model. Pass target column name as query string"
        ),
    ]

    llm = Ollama(model="codellama")

    agent = initialize_agent(
        tools,
        llm,
        agent="zero-shot-react-description",
        verbose=True
    )

    return agent, ds_tools
