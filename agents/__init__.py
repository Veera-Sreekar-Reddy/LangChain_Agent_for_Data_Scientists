"""
Agents Package - Multi-Agent System
"""
from .single_agent import create_agent
from .supervisor_agent import SupervisorAgent
from .reasoning_agent import ReasoningAgent
from .code_agent import CodeGeneratorAgent

__all__ = ['create_agent', 'SupervisorAgent', 'ReasoningAgent', 'CodeGeneratorAgent']


def create_multi_agent():
    """Create and return the supervisor agent system"""
    return SupervisorAgent()

