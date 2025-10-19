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
        
        query_lower = query.lower()
        
        # Check for correlation + visualization keywords
        correlation_keywords = ['correlation', 'correlations', 'relationship']
        viz_keywords = ['heat map', 'heatmap', 'plot', 'chart', 'graph', 'visualization', 'visualize', 'show']
        
        has_correlation = any(kw in query_lower for kw in correlation_keywords)
        has_viz = any(kw in query_lower for kw in viz_keywords)
        
        # If asking for correlation WITH visualization, route to CODE
        if has_correlation and has_viz:
            return "CODE"
        
        routing_prompt = f"""
Analyze this user query and decide which agent should handle it:

Query: "{query}"

Available agents:
1. REASONING - For data summaries, correlations (text), exploration, column info, data cleaning
2. CODE - For generating plots, visualizations, data transformations, custom analysis

If the query asks for:
- Summary, overview, text-based correlations, describe, explore, column info → REASONING
- Plot, chart, graph, visualization, histogram, scatter, heatmap, transform, calculate → CODE

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
                result = self.code_agent.generate_and_execute_code(query)
                
                # Format response for display
                if result['success']:
                    # Check if this is a heatmap/correlation visualization
                    query_lower = query.lower()
                    is_heatmap = 'heatmap' in query_lower or 'heat map' in query_lower
                    is_correlation = 'correlation' in query_lower or 'correlations' in query_lower
                    
                    if result.get('plot') and (is_heatmap or is_correlation):
                        # Provide a concise summary for heatmap
                        response = "📊 **Correlation Heatmap Generated!**\n\n"
                        response += "**Summary:**\n"
                        response += "• The heatmap visualizes correlations between all numeric variables in your dataset\n"
                        response += "• Red/warm colors indicate strong positive correlations (variables increase together)\n"
                        response += "• Blue/cool colors indicate strong negative correlations (one increases, other decreases)\n"
                        response += "• White/neutral colors indicate weak or no correlation\n"
                        response += "• Values range from -1 (perfect negative) to +1 (perfect positive)\n\n"
                        response += "**Key Insights:**\n"
                        response += "• Look for dark red or dark blue cells for strongest relationships\n"
                        response += "• Diagonal is always 1 (variable correlates perfectly with itself)\n"
                        response += "• Use this to identify which variables are most related to each other\n\n"
                        response += f"**Generated Code:**\n```python\n{result['code']}\n```"
                    else:
                        response = f"✅ Code executed successfully!\n\n**Generated Code:**\n```python\n{result['code']}\n```\n\n**Output:**\n{result['output']}"
                else:
                    response = f"❌ Error: {result['error']}"
                
                return {
                    "agent": "CodeLLaMA",
                    "response": response,
                    "success": result['success'],
                    "plot": result.get('plot', None)
                }
            else:
                # Check for direct tool calls to avoid LLM summarization
                query_lower = query.lower()
                
                # Direct tool execution for summary queries to avoid truncation
                if any(keyword in query_lower for keyword in ['summary', 'summarize', 'describe', 'overview']):
                    response = self.reasoning_agent.ds_tools.summarize_data()
                    return {
                        "agent": "Mistral",
                        "response": response,
                        "success": True
                    }
                elif any(keyword in query_lower for keyword in ['correlation', 'correlations', 'relationship']):
                    response = self.reasoning_agent.ds_tools.analyze_correlations()
                    return {
                        "agent": "Mistral",
                        "response": response,
                        "success": True
                    }
                elif any(keyword in query_lower for keyword in ['explore', 'show data', 'view data', 'columns']):
                    response = self.reasoning_agent.ds_tools.explore_data()
                    return {
                        "agent": "Mistral",
                        "response": response,
                        "success": True
                    }
                else:
                    # Use LangChain agent for complex queries
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

