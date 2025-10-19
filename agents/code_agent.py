"""
Code Generator Agent - CodeLLaMA for Python code generation and execution
"""
from langchain_community.llms import Ollama
from tools import DataScienceTools
import io
import sys
import base64
from contextlib import redirect_stdout, redirect_stderr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class CodeGeneratorAgent:
    """CodeLLaMA agent for generating and executing Python code"""
    
    def __init__(self):
        self.llm = Ollama(model="codellama", temperature=0.2)
        self.ds_tools = DataScienceTools()
    
    def generate_and_execute_code(self, query: str) -> dict:
        """Generate Python code using CodeLLaMA and execute it"""
        
        if self.ds_tools.df is None:
            return {
                "success": False,
                "error": "No data loaded",
                "code": None,
                "output": None,
                "plot": None
            }
        
        # Check for common patterns and use templates instead of LLM
        query_lower = query.lower()
        
        # Fast path for scatter plots
        if 'scatter' in query_lower and 'plot' in query_lower:
            code = self._get_scatter_code(query)
            execution_result = self._execute_code_safely(code)
            return {
                "success": True,
                "code": code,
                "output": execution_result["output"],
                "plot": execution_result["plot"],
                "error": None
            }
        
        # Fast path for correlation heatmap
        has_heatmap = 'heatmap' in query_lower or 'heat map' in query_lower
        has_correlation = 'correlation' in query_lower or 'correlations' in query_lower
        has_viz = any(word in query_lower for word in ['plot', 'chart', 'graph', 'visuali', 'show', 'display'])
        
        if has_heatmap or (has_correlation and has_viz):
            code = self._get_heatmap_code()
            execution_result = self._execute_code_safely(code)
            return {
                "success": True,
                "code": code,
                "output": execution_result["output"],
                "plot": execution_result["plot"],
                "error": None
            }
        
        # Create optimized prompt for other queries
        cols = list(self.ds_tools.df.columns)[:10]  # Limit columns shown
        prompt = f"""Generate Python code. DataFrame 'df' is loaded with columns: {cols}

Task: {query}

Rules:
- NO imports, NO pd.read_csv()
- df is ready to use
- For plots: DON'T call plt.show()
- Use print() for text results
- Code only, no markdown

Code:"""
        
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
            execution_result = self._execute_code_safely(code)
            
            return {
                "success": True,
                "code": code,
                "output": execution_result["output"],
                "plot": execution_result["plot"],
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "code": None,
                "output": None,
                "plot": None
            }
    
    def _get_scatter_code(self, query: str) -> str:
        """Generate scatter plot code based on query"""
        # Get all available columns
        all_cols = list(self.ds_tools.df.columns)
        numeric_cols = self.ds_tools.df.select_dtypes(include=['number']).columns.tolist()
        
        # Try to find specific columns mentioned in the query
        query_lower = query.lower()
        x_col = None
        y_col = None
        
        # Check for specific column mentions
        for col in all_cols:
            col_lower = col.lower()
            if col_lower in query_lower:
                if x_col is None:
                    x_col = col
                elif y_col is None:
                    y_col = col
                    break
        
        # If we found both columns, use them
        if x_col and y_col:
            pass  # Use the found columns
        # If we only found one column, try to find a related numeric column
        elif x_col and not y_col:
            # Look for related columns (e.g., if GDP is found, look for Energy Usage)
            if 'gdp' in query_lower:
                for col in all_cols:
                    if 'energy' in col.lower():
                        y_col = col
                        break
            elif 'energy' in query_lower:
                for col in all_cols:
                    if 'gdp' in col.lower():
                        y_col = col
                        break
            
            # If still no y_col, use first numeric column
            if not y_col and numeric_cols:
                y_col = numeric_cols[0]
        else:
            # Use first two numeric columns as fallback
            x_col = numeric_cols[0] if len(numeric_cols) > 0 else all_cols[0]
            y_col = numeric_cols[1] if len(numeric_cols) > 1 else (all_cols[1] if len(all_cols) > 1 else all_cols[0])
        
        return f"""# Generate scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['{x_col}'], df['{y_col}'], alpha=0.6, edgecolors='k', s=50)
plt.xlabel('{x_col}', fontsize=12)
plt.ylabel('{y_col}', fontsize=12)
plt.title('Scatter Plot: {x_col} vs {y_col}', fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()"""
    
    def _get_heatmap_code(self) -> str:
        """Generate optimized heatmap code"""
        return """# Generate correlation heatmap
plt.figure(figsize=(14, 10))
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap - All Numeric Columns', fontsize=16, pad=20)
plt.tight_layout()"""
    
    def _execute_code_safely(self, code: str) -> dict:
        """Execute code in a controlled environment and capture plots"""
        
        # Prepare execution environment
        local_vars = {
            'df': self.ds_tools.df.copy(),
            'pd': __import__('pandas'),
            'np': __import__('numpy'),
            'plt': plt,
            'sns': __import__('seaborn')
        }
        
        # Capture output
        output_buffer = io.StringIO()
        plot_data = None
        
        try:
            with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
                exec(code, local_vars)
            
            # Check if a plot was created
            fig = plt.gcf()
            if fig.get_axes():  # If there are any axes (plot exists)
                # Convert plot to base64
                img_buffer = io.BytesIO()
                fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                img_buffer.seek(0)
                plot_data = base64.b64encode(img_buffer.read()).decode('utf-8')
                plot_data = f'data:image/png;base64,{plot_data}'
                img_buffer.close()
                plt.close(fig)
            
            text_output = output_buffer.getvalue()
            
            return {
                "output": text_output if text_output else "Code executed successfully",
                "plot": plot_data
            }
            
        except Exception as e:
            return {
                "output": f"Execution error: {str(e)}",
                "plot": None
            }
        finally:
            output_buffer.close()
            plt.close('all')  # Clean up any remaining plots

