import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.docstore.document import Document

class DataRetriever:
    def __init__(self):
        self.vectorstore = None

    def index_dataframe(self, df: pd.DataFrame):
        docs = []
        for col in df.columns:
            docs.append(Document(page_content=f"Column: {col}, dtype: {df[col].dtype}, examples: {df[col].head(5).tolist()}"))

        for i in range(0, len(df), 100):
            chunk = df.iloc[i:i+100]
            docs.append(Document(page_content=chunk.to_string()))

        embeddings = OllamaEmbeddings(model="codellama")
        self.vectorstore = FAISS.from_documents(docs, embeddings)
        return "✅ Dataset indexed for retrieval."

    def get_retriever(self):
        if self.vectorstore is None:
            raise ValueError("⚠️ Dataset not indexed yet. Please upload a CSV first.")
        return self.vectorstore.as_retriever(search_kwargs={"k": 3})
