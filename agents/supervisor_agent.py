"""
Supervisor Agent - Routes queries to appropriate specialized agent
"""
from langchain.agents import initialize_agent
from langchain_community.llms import Ollama
from .reasoning_agent import ReasoningAgent
from .code_agent import CodeGeneratorAgent


class SupervisorAgent:
    """Supervisor that routes queries to appropriate agent"""
    
    def __init__(self):
        self.reasoning_agent = ReasoningAgent()
        self.code_agent = CodeGeneratorAgent()
        self.router_llm = Ollama(model="mistral", temperature=0.1)
    
    def set_dataframe(self, df):
        """Set DataFrame for both agents"""
        self.reasoning_agent.ds_tools.df = df
        self.code_agent.ds_tools.df = df
    
    def route_query(self, query: str) -> str:
        """Determine which agent should handle the query"""
        
        routing_prompt = f"""
Analyze this user query and decide which agent should handle it:

Query: "{query}"

Available agents:
1. REASONING - For data summaries, correlations, exploration, column info, data cleaning
2. CODE - For generating plots, visualizations, data transformations, custom analysis

If the query asks for:
- Summary, overview, correlations, describe, explore, column info → REASONING
- Plot, chart, graph, visualization, histogram, scatter, transform, calculate → CODE

Answer with ONLY one word: REASONING or CODE
"""
        
        decision = self.router_llm.invoke(routing_prompt).strip().upper()
        
        if "CODE" in decision:
            return "CODE"
        else:
            return "REASONING"
    
    def process_query(self, query: str) -> dict:
        """Process query using appropriate agent"""
        
        # Route to appropriate agent
        agent_type = self.route_query(query)
        
        try:
            if agent_type == "CODE":
                response = self.code_agent.generate_and_execute_code(query)
                return {
                    "agent": "CodeLLaMA",
                    "response": response,
                    "success": True
                }
            else:
                # Create reasoning agent with tools
                tools = self.reasoning_agent.create_tools()
                agent = initialize_agent(
                    tools,
                    self.reasoning_agent.llm,
                    agent="zero-shot-react-description",
                    verbose=True,
                    handle_parsing_errors=True,
                    max_iterations=5,
                    max_execution_time=60,
                    early_stopping_method="generate"
                )
                response = agent.run(query)
                return {
                    "agent": "Mistral",
                    "response": response,
                    "success": True
                }
                
        except Exception as e:
            return {
                "agent": agent_type,
                "response": f"❌ Error: {str(e)}",
                "success": False
            }

