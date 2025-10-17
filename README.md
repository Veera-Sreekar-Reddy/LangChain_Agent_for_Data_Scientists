# 🤖 AI-Powered Data Science Assistant

A **production-ready Streamlit application** with multi-agent AI system for intelligent data analysis, cleaning, visualization, and machine learning.

**Key Technologies:** Multi-Agent LangChain + Mistral + CodeLLaMA (via Ollama) + Advanced Data Processing

---

## 🌟 Highlights

- **🤖 Multi-Agent AI System**: Mistral for reasoning + CodeLLaMA for code generation
- **🧹 Advanced Data Cleaning**: 10+ strategies with detailed logging
- **📊 Interactive Visualizations**: 10+ plot types with Plotly
- **🤖 Machine Learning**: Multi-model comparison with metrics
- **💬 Natural Language Interface**: Ask questions, get answers and code
- **📥 Export Capabilities**: CSV, Excel, HTML reports

---

## 📑 Table of Contents

1. [Features](#features)
2. [Quick Start](#quick-start)
3. [Multi-Agent System](#multi-agent-system)
4. [Data Cleaning](#data-cleaning)
5. [Usage Examples](#usage-examples)
6. [Architecture](#architecture)
7. [API Reference](#api-reference)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [Tech Stack](#tech-stack)

---

## ✨ Features

### 📊 Overview & Analytics
- 📂 Upload CSV files with instant statistics
- 🔍 Comprehensive dataset exploration (columns, dtypes, missing values)
- 📈 Automated statistics dashboard
- 📋 One-click EDA reports with `ydata-profiling`

### 🧹 Advanced Data Cleaning ⭐
- **🤖 Auto-Clean Modes**: Light, Standard, Aggressive
- **🔴 Missing Values**: 6 strategies (mean, median, mode, KNN, drop, fill)
- **🔄 Duplicates**: Smart duplicate removal with keep options
- **📊 Outliers**: IQR and Z-score detection with clipping
- **🔤 Encoding**: Label and One-Hot encoding
- **📏 Scaling**: Standard, MinMax, and Robust scaling
- **💾 Memory Optimization**: Automatic dtype optimization (50-90% reduction)
- **🗑️ Column Removal**: Constant and high-cardinality columns
- **📋 Detailed Logging**: Track all cleaning operations with before/after comparison

### 📈 Interactive Visualizations (10+ types)
- **Scatter plots** with color coding and 3D scatter
- **Box plots** and **Violin plots**
- **Histograms** and distribution plots
- **Correlation heatmaps**
- **Pair plots** (scatter matrix)
- **Line plots** and **Bar plots**
- All interactive with Plotly (zoom, pan, hover)

### 🤖 Machine Learning
- **Multi-model comparison**: Random Forest, XGBoost, Logistic/Linear Regression
- **Visual results**: Confusion matrices, prediction plots
- **Performance metrics**: Accuracy, R², MSE, MAE
- **Training time tracking**
- **AI-powered model suggestions**

### 💬 Multi-Agent AI Assistant ⭐
- **Dual AI System**: Mistral (reasoning) + CodeLLaMA (code generation)
- **Smart Routing**: Automatically selects appropriate agent
- **Natural Language**: Ask questions in plain English
- **Code Generation**: Creates and executes Python code for plots
- **Context-Aware**: Understands your data and intentions
- **Safe Execution**: Sandboxed code environment

### 📥 Export Capabilities
- Download cleaned CSV
- Export to Excel (.xlsx)
- Generate summary reports (TXT)
- Download comprehensive HTML EDA reports

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Ollama installed
- 8GB+ RAM recommended

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Veera-Sreekar-Reddy/LangChain_Agent_for_Data_Scientists.git
cd LangChain_Agent_for_Data_Scientists

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install Ollama (macOS)
brew install ollama

# 4. Pull AI models
ollama pull mistral
ollama pull codellama

# 5. Start Ollama service
ollama serve &

# 6. Run the application
streamlit run app.py
```

### Verify Models
```bash
ollama list
# Should show:
# mistral:latest
# codellama:latest
```

### Access Application
Open your browser and navigate to: **http://localhost:8501**

---

## 🤖 Multi-Agent System

### Overview

The application uses a **three-agent architecture**:

1. **SupervisorAgent** (Mistral): Routes queries to appropriate agent
2. **ReasoningAgent** (Mistral): Handles analysis, summaries, correlations
3. **CodeGeneratorAgent** (CodeLLaMA): Generates and executes Python code

### Architecture

```
User Query
    ↓
Supervisor Agent (Mistral)
    ↓
Intelligent Routing
    ├─→ Mistral (Reasoning)        ├─→ CodeLLaMA (Code Gen)
    │   • Summarize data            │   • Generate plots
    │   • Analyze correlations      │   • Create visualizations
    │   • Column info               │   • Transform data
    │   • Data exploration          │   • Custom analysis
    └───────────────────────────────┴──────────────────────
                        ↓
                    Response
```

### Query Examples

#### → Mistral (Reasoning)
```
"Summarize the dataset"
"Explain correlations in this data"
"Tell me about the price column"
"What are the missing values?"
"Describe the data distribution"
```

#### → CodeLLaMA (Code Generation)
```
"Create a scatter plot of price vs duration"
"Plot histogram of age distribution"
"Generate a correlation heatmap"
"Show bar chart of categories"
"Create 3D scatter plot of X, Y, Z"
```

### Agent Selection

The Supervisor analyzes queries and routes based on keywords:

**Mistral Keywords**: summary, summarize, describe, explain, analyze, correlation, explore, show, tell
**CodeLLaMA Keywords**: plot, chart, graph, visualize, create, generate, histogram, scatter, heatmap

### Benefits

✅ **Specialization**: Each agent excels at specific tasks  
✅ **Better Quality**: CodeLLaMA generates cleaner code, Mistral provides better insights  
✅ **Faster**: Optimized routing reduces response time  
✅ **Scalable**: Easy to add more specialized agents  
✅ **Flexible**: Can switch between multi-agent and single-agent modes  

### Usage

**In the UI:**
1. Select **"Multi-Agent (Recommended)"** in sidebar
2. Go to **"💬 AI Assistant"** tab
3. Type your question
4. See which agent responds
5. View generated code (for CodeLLaMA) or analysis (for Mistral)

**Programmatically:**
```python
from multi_agent import create_multi_agent

supervisor = create_multi_agent()
supervisor.set_dataframe(df)

result = supervisor.process_query("Create scatter plot of price vs duration")
print(f"Agent: {result['agent']}")
print(f"Response: {result['response']}")
```

---

## 🧹 Data Cleaning

### Overview

Production-ready data cleaning with 10+ strategies, detailed logging, and interactive UI.

### Auto-Clean Modes

#### 🌟 Light
- Remove duplicates
- Handle missing values (basic imputation)
- **Best for**: Quick cleaning, exploratory analysis

#### 🌟 Standard (Recommended)
- Everything in Light, plus:
- Outlier detection (IQR method)
- Categorical encoding (Label)
- Data type optimization
- **Best for**: Most use cases, ML preparation

#### 🌟 Aggressive
- Everything in Standard, plus:
- Remove constant columns
- Remove high cardinality columns (IDs)
- **Best for**: Feature reduction, production pipelines

### Manual Cleaning Options

#### 1. 🔴 Missing Values

**Strategies:**
- **auto**: Smart imputation (mean/median for numeric, mode for categorical)
- **drop**: Remove rows with missing values
- **impute**: Custom methods
  - Numeric: mean, median, mode, KNN
  - Categorical: mode, constant ('Unknown'), missing
- **fill_zero**: Fill all with 0
- **fill_forward**: Forward fill (time series)
- **fill_backward**: Backward fill

**When to Use:**
```python
# Financial data - use median (robust to outliers)
strategy='impute', numeric_method='median'

# Survey data - use mode
strategy='impute', categorical_method='mode'

# Time series - forward fill
strategy='fill_forward'
```

#### 2. 🔄 Duplicates

**Options:**
- `keep='first'`: Keep first occurrence
- `keep='last'`: Keep last occurrence (transaction data)
- `keep=False`: Remove all duplicates

#### 3. 📊 Outliers

**IQR Method (Interquartile Range):**
- Threshold 1.5: Standard (catches extreme outliers)
- Threshold 2.0: Moderate
- Threshold 3.0: Conservative

**Z-Score Method:**
- Threshold 2.0: Aggressive (2 std devs)
- Threshold 3.0: Standard (3 std devs)

**Action**: Clips values to bounds (doesn't remove rows)

#### 4. 🔤 Encoding

**Label Encoding:**
- Converts categories to numbers: A=0, B=1, C=2
- Best for: Tree-based models, ordinal data

**One-Hot Encoding:**
- Creates binary columns for each category
- Best for: Linear models, few categories (<10)

#### 5. 📏 Scaling

**Standard Scaling**: Mean=0, Std=1 (best for normal distribution)  
**MinMax Scaling**: Scales to [0, 1] (best for neural networks)  
**Robust Scaling**: Uses median and IQR (best for outliers)

**When to Scale:**
- ✅ SVM, Neural Networks, KNN, Linear Regression
- ❌ Tree-based models (Random Forest, XGBoost)

#### 6. 💾 Memory Optimization

Automatically optimizes data types:
- `int64` → `int8/16/32` (based on value range)
- `float64` → `float32`
- **Result**: 50-90% memory reduction

#### 7. 🗑️ Column Removal

**Constant Columns**: Only one unique value (no information)  
**High Cardinality**: >95% unique values (likely IDs)

### Cleaning Report

Example output:
```
🧹 DATA CLEANING REPORT
======================================================================

Operations Performed: 5

1. 🗑️ Removed 120 duplicate rows (kept=first)
2. 🔢 Filled 'age' missing with median (35.0)
3. 📝 Filled 'category' missing with mode ('A')
4. 📊 Clipped 45 outliers in 'price' using IQR method
5. 🔤 Label encoded 'category' (3 categories)

======================================================================
📊 BEFORE vs AFTER:

Shape: (1000, 15) → (880, 15)
Missing Values: 150 → 0
Duplicates: 120 → 0
Memory: 1.25 MB → 0.65 MB
```

### Usage

**In the UI:**
1. Upload CSV file
2. Go to **"🧹 Data Cleaning"** tab
3. Choose auto-clean level OR use manual controls
4. Click apply
5. Review cleaning report
6. See cleaned data preview

**Programmatically:**
```python
from tools import DataCleaner

# Auto-clean
cleaner = DataCleaner(df)
cleaner.auto_clean(level='standard')

# Manual cleaning
cleaner.handle_missing_values(strategy='impute', numeric_method='median')
cleaner.remove_duplicates(keep='first')
cleaner.handle_outliers(method='iqr', threshold=1.5)
cleaner.encode_categorical(method='label')
cleaner.scale_features(method='standard')
cleaner.optimize_dtypes()

# Get results
cleaned_df = cleaner.get_cleaned_dataframe()
report = cleaner.get_cleaning_report()
```

---

## 📖 Usage Examples

### Example 1: Quick Data Analysis

```python
# 1. Upload CSV in UI
# 2. Go to Overview tab → Review statistics
# 3. Go to AI Assistant tab
# 4. Ask: "Summarize the dataset"
# → Mistral provides comprehensive summary

# 5. Ask: "Create scatter plot of price vs quantity"
# → CodeLLaMA generates and executes code
```

### Example 2: ML Pipeline

```python
# 1. Upload data
# 2. Go to Data Cleaning tab
# 3. Run Auto-Clean (Standard)
# 4. Go to ML Models tab
# 5. Select target column
# 6. Choose models to compare
# 7. Train and compare results
```

### Example 3: Custom Cleaning

```python
from tools import DataCleaner

df = pd.read_csv("data.csv")
cleaner = DataCleaner(df)

# Step-by-step cleaning
cleaner.remove_duplicates(keep='last')
cleaner.handle_missing_values(
    strategy='impute',
    numeric_method='median',
    categorical_method='mode'
)
cleaner.handle_outliers(method='iqr', threshold=2.0)
cleaner.encode_categorical(method='onehot')
cleaner.scale_features(method='minmax')

# Get cleaned data
cleaned_df = cleaner.get_cleaned_dataframe()
print(cleaner.get_cleaning_report())
```

### Example 4: Multi-Agent Query

```python
from multi_agent import create_multi_agent

supervisor = create_multi_agent()
supervisor.set_dataframe(df)

# Analysis query → Routes to Mistral
result1 = supervisor.process_query("Explain correlations")
print(result1['agent'])  # "Mistral"

# Visualization query → Routes to CodeLLaMA
result2 = supervisor.process_query("Create histogram of age")
print(result2['agent'])  # "CodeLLaMA"
```

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                  Streamlit UI (app.py)                       │
│  Tabs: Overview | Cleaning | Viz | ML | AI | Export         │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
    ┌────▼────┐                    ┌────▼────────┐
    │ Single  │                    │ Multi-Agent │
    │ Agent   │                    │ System      │
    │(Mistral)│                    │             │
    └─────────┘                    └──────┬──────┘
                                          │
                        ┌─────────────────┴─────────────────┐
                        │                                   │
                ┌───────▼────────┐              ┌──────────▼────────┐
                │ Reasoning      │              │ Code Generator    │
                │ Agent          │              │ Agent             │
                │ (Mistral)      │              │ (CodeLLaMA)       │
                └───────┬────────┘              └──────────┬────────┘
                        │                                  │
                        ▼                                  ▼
            ┌──────────────────────┐        ┌──────────────────────┐
            │ DataScienceTools     │        │ Code Execution       │
            │ • Summarize          │        │ • Generate Python    │
            │ • Correlations       │        │ • Safe Sandbox       │
            │ • Column Info        │        │ • Plot Creation      │
            │ • Clean Data         │        │                      │
            └──────────┬───────────┘        └──────────┬───────────┘
                       │                               │
                       ▼                               ▼
            ┌──────────────────────┐        ┌──────────────────────┐
            │ DataCleaner          │        │ Visualization        │
            │ • 10+ Strategies     │        │ • Matplotlib         │
            │ • Detailed Logging   │        │ • Seaborn           │
            └──────────────────────┘        └──────────────────────┘
```

### File Structure

```
LangChain_Agent_for_Data_Scientists/
│
├── 📱 app.py                              # Main Streamlit Application (UI)
│
├── 📁 agents/                             # AI Agents Module
│   ├── __init__.py                        # Package exports
│   ├── single_agent.py                    # Single Mistral agent
│   ├── supervisor_agent.py                # Query router (supervisor)
│   ├── reasoning_agent.py                 # Mistral reasoning agent
│   └── code_agent.py                      # CodeLLaMA code generator
│
├── 📁 tools/                              # Data Science Tools Module
│   ├── __init__.py                        # Package exports
│   ├── data_science_tools.py              # Analysis, visualization, ML
│   ├── data_cleaner.py                    # Advanced data cleaning
│   └── data_retriever.py                  # FAISS vector retrieval
│
├── 📋 README.md                           # Complete documentation
├── 📦 requirements.txt                    # Python dependencies
└── 🔧 setup_multi_agent.sh                # Automated setup script
```

---

## 📦 Module Documentation

### `agents/` - AI Agents Package

#### `single_agent.py`
- **Purpose**: Single Mistral agent for general data science tasks
- **Function**: `create_agent()` → Returns (agent, ds_tools)
- **Use Case**: Simple queries, single-model inference

#### `supervisor_agent.py`  
- **Purpose**: Routes queries to specialized agents
- **Class**: `SupervisorAgent`
- **Methods**: `route_query()`, `process_query()`, `set_dataframe()`
- **Use Case**: Multi-agent orchestration

#### `reasoning_agent.py`
- **Purpose**: Mistral agent for analysis and reasoning
- **Class**: `ReasoningAgent`
- **Tools**: Summarize, Correlations, Explore, Column Info, Clean
- **Use Case**: Data summaries, statistical analysis

#### `code_agent.py`
- **Purpose**: CodeLLaMA for Python code generation
- **Class**: `CodeGeneratorAgent`
- **Methods**: `generate_and_execute_code()`, `_execute_code_safely()`
- **Use Case**: Visualizations, transformations, custom code

### `tools/` - Data Science Tools Package

#### `data_science_tools.py`
- **Purpose**: Core data science operations
- **Class**: `DataScienceTools`
- **Methods**:
  - `load_data()`, `get_dataframe()`
  - `explore_data()`, `summarize_data()`, `analyze_correlations()`
  - `get_column_info()`, `clean_data()`
  - `generate_plot()`, `suggest_model()`, `train_models()`

#### `data_cleaner.py`
- **Purpose**: Advanced data cleaning with multiple strategies
- **Class**: `DataCleaner`
- **Methods**:
  - `auto_clean(level)` - Light, Standard, Aggressive
  - `handle_missing_values()` - 6 strategies
  - `remove_duplicates()`, `handle_outliers()`
  - `encode_categorical()`, `scale_features()`
  - `optimize_dtypes()`, `remove_constant_columns()`
  - `get_cleaning_report()`, `reset()`

#### `data_retriever.py`
- **Purpose**: FAISS-based vector retrieval
- **Class**: `DataRetriever`
- **Methods**: `index_dataframe()`, `get_retriever()`
- **Use Case**: Semantic search over data

---

## 📚 API Reference

### DataCleaner Class

```python
from tools import DataCleaner

# Initialize
cleaner = DataCleaner(df)

# Auto-clean
cleaner.auto_clean(level='light' | 'standard' | 'aggressive')

# Manual operations
cleaner.handle_missing_values(
    strategy='auto' | 'drop' | 'impute' | 'fill_zero' | 'fill_forward' | 'fill_backward',
    numeric_method='mean' | 'median' | 'mode' | 'knn',
    categorical_method='mode' | 'constant' | 'missing'
)

cleaner.remove_duplicates(
    subset=None,  # Optional: columns to check
    keep='first' | 'last' | False
)

cleaner.handle_outliers(
    method='iqr' | 'zscore',
    columns=None,  # Optional: specific columns
    threshold=1.5  # IQR multiplier or Z-score threshold
)

cleaner.encode_categorical(
    method='label' | 'onehot',
    columns=None  # Optional: specific columns
)

cleaner.scale_features(
    method='standard' | 'minmax' | 'robust',
    columns=None  # Optional: specific columns
)

cleaner.optimize_dtypes()
cleaner.remove_constant_columns()
cleaner.remove_high_cardinality_columns(threshold=0.95)

# Get results
cleaned_df = cleaner.get_cleaned_dataframe()
report = cleaner.get_cleaning_report()
cleaner.reset()  # Reset to original
```

### Multi-Agent System

```python
from multi_agent import create_multi_agent

# Initialize
supervisor = create_multi_agent()
supervisor.set_dataframe(df)

# Process query
result = supervisor.process_query("Your question here")

# Result structure
{
    'agent': 'Mistral' | 'CodeLLaMA',
    'response': 'Response text or code output',
    'success': True | False
}

# Route query (see which agent will handle)
agent_type = supervisor.route_query("Your question")
```

### DataScienceTools

```python
from tools import DataScienceTools

ds_tools = DataScienceTools()

# Load data
ds_tools.load_data("file.csv")

# Analysis
ds_tools.summarize_data()
ds_tools.analyze_correlations()
ds_tools.explore_data()
ds_tools.get_column_info("column_name")

# Cleaning
ds_tools.clean_data(level='standard')

# Visualization
ds_tools.generate_plot("scatter plot of price vs duration")

# ML
ds_tools.suggest_model("target_column")
ds_tools.train_models(df, target, models, test_size)
```

---

## 💡 Best Practices

### Data Cleaning

#### ✅ Do:

1. **Always review data first**
   - Check Overview tab
   - Understand missing patterns
   - Identify outliers visually

2. **Clean incrementally**
   ```
   1. Remove duplicates
   2. Handle missing values
   3. Handle outliers
   4. Encode categorical (for ML only)
   5. Scale features (for ML only)
   ```

3. **Use appropriate strategies**
   - EDA: Light cleaning, keep outliers
   - ML: Standard cleaning, handle outliers
   - Production: Aggressive, optimize memory

4. **Read cleaning reports**
   - Verify operations make sense
   - Check rows/columns removed
   - Validate results

5. **Keep original data**
   - Use Reset button if needed
   - Save cleaned data separately

#### ❌ Don't:

1. **Blindly auto-clean** - Understand your data first
2. **Remove too much data** - Consider clipping vs removing
3. **Encode before EDA** - Categorical values are more interpretable
4. **Scale before visualization** - Original values are meaningful
5. **Ignore domain knowledge** - Some "outliers" might be valid

### Multi-Agent Usage

#### ✅ Do:

1. **Be specific in queries**
   - ✅ "Create scatter plot of price vs duration"
   - ❌ "Show me the data"

2. **Mention column names**
   - ✅ "Plot histogram of age column"
   - ❌ "Make a histogram"

3. **Use action words**
   - Analysis: summarize, explain, analyze, describe
   - Visualization: create, plot, generate, show

4. **Check agent responses**
   - See which agent handled query
   - Verify routing makes sense

5. **Start simple**
   - Test with basic queries first
   - Build up to complex requests

#### ❌ Don't:

1. **Mix multiple requests** - One query at a time
2. **Use vague language** - "analyze this" is too broad
3. **Expect instant results** - Code generation takes 3-8 seconds
4. **Request impossible plots** - Need appropriate column types

---

## 🔧 Troubleshooting

### Common Issues

#### Issue: "Ollama server not responding"
**Solution:**
```bash
ollama serve
# Wait 3-5 seconds for startup
```

#### Issue: "Model not found"
**Solution:**
```bash
ollama pull mistral
ollama pull codellama
ollama list  # Verify
```

#### Issue: Wrong agent selected
**Solution:** Be more specific in query
- ❌ "Show me data" → Ambiguous
- ✅ "Summarize the dataset" → Mistral
- ✅ "Create scatter plot" → CodeLLaMA

#### Issue: Code generation fails
**Causes:**
1. Column names don't exist
2. Wrong data type for operation
3. Query too complex

**Solution:**
- Check DataFrame columns
- Simplify query
- Try manual visualization in Visualizations tab

#### Issue: Too much data removed
**Solution:**
- Use less aggressive threshold (2.0 or 3.0)
- Check cleaning report
- Reset and try different strategy

#### Issue: Memory error
**Solution:**
- Optimize dtypes first
- Sample large datasets: `df.sample(10000)`
- Process in chunks

#### Issue: Encoding creates too many columns
**Solution:**
- Use label encoding instead of one-hot
- Remove high cardinality columns first
- Combine rare categories

### Performance Tips

**Large Datasets (>100K rows):**
1. Sample first: `df_sample = df.sample(10000)`
2. Use efficient methods (mean/median, not KNN)
3. Optimize dtypes early

**Many Columns (>50):**
1. Select relevant columns first
2. Remove high cardinality early
3. Use label encoding

**Slow Queries:**
1. Reduce max_iterations (in `agent.py`)
2. Use Single Agent mode
3. Simplify queries

---

## ⚙️ Tech Stack

### Frontend
- **Streamlit**: Web interface
- **Plotly**: Interactive visualizations
- **Matplotlib/Seaborn**: Static plots

### AI/ML
- **LangChain**: Agent framework
- **Ollama**: Local LLM hosting
- **Mistral 7B**: Reasoning agent
- **CodeLLaMA 7B**: Code generation
- **Scikit-learn**: ML models
- **XGBoost**: Gradient boosting
- **SHAP**: Model explainability

### Data Processing
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **SciPy**: Statistical functions
- **ydata-profiling**: EDA reports
- **imbalanced-learn**: Sampling methods

### Storage & Export
- **openpyxl**: Excel export
- **FAISS**: Vector storage (for future features)

---

## 📦 Requirements

```txt
streamlit
langchain
langchain-community
pandas
scikit-learn
faiss-cpu
ollama
matplotlib
seaborn
ydata-profiling
plotly
xgboost
shap
scipy
imbalanced-learn
openpyxl
```

---

## 🧪 Testing

### Test Multi-Agent System
```bash
python test_multi_agent.py
```

### Test Data Cleaning
```bash
python test_data_cleaning.py
```

### Test in UI
```bash
streamlit run app.py
# 1. Upload sample CSV
# 2. Try each tab
# 3. Test AI Assistant with both agents
# 4. Test data cleaning with different strategies
```

---

## 🚀 Deployment

### Local Development
```bash
ollama serve &
streamlit run app.py
```

### Docker (Coming Soon)
```bash
docker-compose up
```

### Cloud Deployment
- Requires GPU for Ollama models (8GB+ VRAM)
- Or use cloud LLM APIs (OpenAI, Anthropic)
- Configure in environment variables

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] Additional cleaning strategies
- [ ] More specialized agents (SQL, Time Series, NLP)
- [ ] Support for more file formats
- [ ] Advanced feature engineering
- [ ] Model deployment pipeline
- [ ] Collaborative features

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🙏 Credits

- **LangChain** - Agent framework
- **Ollama** - Local LLM hosting
- **Mistral AI** - Reasoning model
- **Meta** - CodeLLaMA model
- **Streamlit** - UI framework
- **Scikit-learn** - ML library

---

## 📞 Support

- **Issues**: GitHub Issues
- **Documentation**: This README
- **Examples**: `test_*.py` scripts

---

## 🎉 Quick Reference

### Common Commands
```bash
# Start Ollama
ollama serve &

# Pull models
ollama pull mistral
ollama pull codellama

# Run app
streamlit run app.py

# Test features
python test_multi_agent.py
python test_data_cleaning.py
```

### Quick Queries
```
Analysis (Mistral):
- "Summarize the data"
- "Explain correlations"
- "Tell me about column X"

Visualization (CodeLLaMA):
- "Create scatter plot of X vs Y"
- "Plot histogram of Z"
- "Generate correlation heatmap"
```

### Quick Cleaning
```python
# UI: Data Cleaning tab → Auto-Clean (Standard)

# Code:
from tools import DataCleaner
cleaner = DataCleaner(df)
cleaner.auto_clean(level='standard')
cleaned_df = cleaner.get_cleaned_dataframe()
```

---

**Built with ❤️ using AI and Open Source**

**Star ⭐ this repo if you find it useful!**

---

*Last Updated: October 2025*
