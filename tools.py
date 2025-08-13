import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import Document
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class DataScienceTools:
    def __init__(self):
        self.df = None
        self.vectorstore = None

    def load_csv(self, file_path):
        self.df = pd.read_csv(file_path)
        return f"CSV loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns."

    def setup_rag(self, _input=None):
        if self.df is None:
            return "No CSV loaded."
        text_data = self.df.astype(str).apply(lambda row: " ".join(row), axis=1).tolist()
        docs = [Document(page_content=txt) for txt in text_data]
        embeddings = OllamaEmbeddings(model="codellama:7b-python")
        self.vectorstore = FAISS.from_documents(docs, embeddings)
        return "RAG index created."

    def search_data(self, query):
        if self.vectorstore is None:
            return "RAG not initialized."
        results = self.vectorstore.similarity_search(query, k=3)
        return "\n".join([r.page_content for r in results])

    def auto_ml(self, target_col):
        if self.df is None:
            return "No CSV loaded."
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]
        X = pd.get_dummies(X)
        problem_type = "classification" if y.nunique() < 20 and y.dtype != float else "regression"
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if problem_type == "classification":
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            report = classification_report(y_test, preds)
        else:
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            report = f"MSE: {mean_squared_error(y_test, preds)}"
        return f"Problem type: {problem_type}\n\nResults:\n{report}"
