# Local AI-Powered Data Science Assistant

This is a **Streamlit-based application** that allows users to interact with their CSV datasets using natural language.
The app leverages [CodeLLaMA](https://ollama.com/library/codellama) running locally via [Ollama](https://ollama.com) along with [LangChain](https://www.langchain.com/) and **RAG** to explore, clean, and suggest machine learning models for uploaded data.

---

## 🚀 Features

- 📂 Upload a CSV file and preview the dataset.
- 🔍 Automatic dataset exploration (columns, dtypes, summary, head).
- 🧹 Data cleaning (handles missing values, encodes categorical variables).
- 🤖 ML model suggestions (classification or regression based on target column).
- 💬 Ask natural language questions about your data.
- 📊 Auto-generates Python code using CodeLLaMA and executes securely.
- 🧠 Supports large datasets with **Retrieval-Augmented Generation (RAG)**.

---

## ⚙️ Tech Stack

- **Python 3.8+**
- **Streamlit**
- **LangChain**
- **Ollama + CodeLLaMA**
- **scikit-learn, pandas, faiss**

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
