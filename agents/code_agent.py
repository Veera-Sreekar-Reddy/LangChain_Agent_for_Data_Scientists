"""
Code Generator Agent - CodeLLaMA for Python code generation and execution
"""
from langchain_community.llms import Ollama
from tools import DataScienceTools
import io
import sys
from contextlib import redirect_stdout, redirect_stderr


class CodeGeneratorAgent:
    """CodeLLaMA agent for generating and executing Python code"""
    
    def __init__(self):
        self.llm = Ollama(model="codellama", temperature=0.2)
        self.ds_tools = DataScienceTools()
    
    def generate_and_execute_code(self, query: str) -> str:
        """Generate Python code using CodeLLaMA and execute it"""
        
        if self.ds_tools.df is None:
            return "⚠️ No data loaded."
        
        # Create prompt for code generation
        prompt = f"""
You are a Python data science code generator. Generate ONLY executable Python code.

Dataset info:
- Columns: {list(self.ds_tools.df.columns)}
- Shape: {self.ds_tools.df.shape}
- Data types: {self.ds_tools.df.dtypes.to_dict()}

User request: {query}

Rules:
1. The DataFrame is available as 'df'
2. Import statements: pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
3. For plots, use plt.figure() and don't call plt.show()
4. Print results to stdout
5. Generate ONLY code, no explanations

Code:
"""
        
        try:
            # Generate code
            code = self.llm.invoke(prompt)
            
            # Clean the code
            code = code.strip()
            if code.startswith("```python"):
                code = code[10:]
            if code.startswith("```"):
                code = code[3:]
            if code.endswith("```"):
                code = code[:-3]
            code = code.strip()
            
            # Execute code
            output = self._execute_code_safely(code)
            return f"✅ Code executed successfully!\n\n**Generated Code:**\n```python\n{code}\n```\n\n**Output:**\n{output}"
            
        except Exception as e:
            return f"❌ Error executing code: {str(e)}"
    
    def _execute_code_safely(self, code: str) -> str:
        """Execute code in a controlled environment"""
        
        # Prepare execution environment
        local_vars = {
            'df': self.ds_tools.df.copy(),
            'pd': __import__('pandas'),
            'np': __import__('numpy'),
            'plt': __import__('matplotlib.pyplot'),
            'sns': __import__('seaborn')
        }
        
        # Capture output
        output_buffer = io.StringIO()
        
        try:
            with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
                exec(code, local_vars)
            
            result = output_buffer.getvalue()
            return result if result else "Code executed without output"
            
        except Exception as e:
            return f"Execution error: {str(e)}"
        finally:
            output_buffer.close()

