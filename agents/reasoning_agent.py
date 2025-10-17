"""
Reasoning Agent - Mistral for data analysis and reasoning
"""
from langchain.agents import Tool
from langchain_community.llms import Ollama
from tools import DataScienceTools


class ReasoningAgent:
    """Mistral agent for understanding queries and data analysis"""
    
    def __init__(self):
        self.llm = Ollama(model="mistral", temperature=0.1)
        self.ds_tools = DataScienceTools()
    
    def create_tools(self):
        """Create reasoning and analysis tools"""
        return [
            Tool(
                name="Summarize Data",
                func=lambda query: self.ds_tools.summarize_data(),
                description="Use when user asks to 'summarize data', 'describe dataset', 'overview', 'show summary'. Input: empty string."
            ),
            Tool(
                name="Analyze Correlations",
                func=lambda query: self.ds_tools.analyze_correlations(),
                description="Use for correlation analysis, relationships between variables. Input: empty string."
            ),
            Tool(
                name="Explore Data",
                func=lambda query: self.ds_tools.explore_data(),
                description="View dataset columns, data types, and first few rows. Input: empty string."
            ),
            Tool(
                name="Get Column Info",
                func=lambda query: self.ds_tools.get_column_info(query.strip()),
                description="Get detailed information about a specific column. Input: column name."
            ),
            Tool(
                name="Clean Data",
                func=lambda query: self.ds_tools.clean_data(),
                description="Clean dataset by handling missing values and encoding categorical variables. Input: empty string."
            ),
        ]

