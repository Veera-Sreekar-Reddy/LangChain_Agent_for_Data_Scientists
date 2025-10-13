from langchain.agents import initialize_agent, Tool
from langchain_community.llms import Ollama
from tools import DataScienceTools

def create_agent():
    ds_tools = DataScienceTools()

    tools = [
        Tool(
            name="Explore Data",
            func=lambda query: ds_tools.explore_data(),
            description="Use this to view the dataset columns, data types, and first few rows. Input should be empty string."
        ),
        Tool(
            name="Clean Data",
            func=lambda query: ds_tools.clean_data(),
            description="Use this to clean the dataset by handling missing values and encoding categorical variables. Input should be empty string."
        ),
        Tool(
            name="Get Column Info",
            func=lambda query: ds_tools.get_column_info(query.strip()),
            description="Get detailed information about a specific column. Input should be the column name."
        ),
        Tool(
            name="Generate Plot",
            func=lambda query: ds_tools.generate_plot(query),
            description="Generate visualizations like scatter plot, box plot, histogram, bar plot, line plot, or correlation heatmap. Input should describe the plot and mention column names. Examples: 'scatter plot for duration and price', 'box plot for age', 'histogram of salary', 'correlation heatmap'."
        ),
    ]

    llm = Ollama(model="codellama")

    agent = initialize_agent(
        tools,
        llm,
        agent="zero-shot-react-description",
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )

    return agent, ds_tools
