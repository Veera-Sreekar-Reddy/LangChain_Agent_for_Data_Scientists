# 🤖 AI-Powered Data Science Assistant

This is an **enhanced Streamlit-based application** that allows users to interact with their CSV datasets using natural language.
The app leverages [CodeLLaMA](https://ollama.com/library/codellama) running locally via [Ollama](https://ollama.com) along with [LangChain](https://www.langchain.com/) and advanced analytics to explore, clean, visualize, and build machine learning models for uploaded data.

---

## 🚀 Features

### 📊 Overview & Analytics
- 📂 Upload CSV files with instant statistics
- 🔍 Comprehensive dataset exploration (columns, dtypes, missing values)
- 📈 Automated statistics dashboard
- 📋 One-click EDA reports with `ydata-profiling`
- 🧹 Data cleaning (handles missing values, encodes categorical variables)

### 📈 Interactive Visualizations (10+ types)
- **Scatter plots** with color coding
- **3D scatter plots** for multivariate analysis
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
- AI-powered model suggestions

### 💬 AI Assistant
- Natural language queries with CodeLLaMA
- Auto-generates visualizations from text
- Context-aware responses
- Smart tool selection

### 📥 Export Capabilities
- Download cleaned CSV
- Export to Excel (.xlsx)
- Generate summary reports (TXT)
- Download comprehensive HTML EDA reports

---

## ⚙️ Tech Stack

- **Python 3.8+**
- **Frontend**: Streamlit (with tabs and advanced layouts)
- **AI/ML**: LangChain + CodeLLaMA (via Ollama)
- **Visualization**: Plotly (interactive), Matplotlib, Seaborn
- **Machine Learning**: scikit-learn, XGBoost, SHAP
- **Data Processing**: pandas, numpy, scipy
- **EDA Reports**: ydata-profiling
- **Export**: openpyxl (Excel), io (CSV)

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Install Ollama

```
brew install ollama
```

### 4. Start ollama service

```
ollama serve
```

### 5. Pull Codellama model

```
ollama pull codellama
```

### 6. Run codellama

```
ollama run codellama
```

### 7. Run streamlit app

```
streamlit run app.py
```

### 8. Upload the file in streamlit app in web browser

### 9. Ask a question to the Agent

### 10. Select the target column for model suggestion
