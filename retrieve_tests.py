import os
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings

# ----- Load environment variables -----
load_dotenv()

# ----- Custom Embeddings using Ollama -----
class OllamaEmbeddings(Embeddings):
    def __init__(self, model: str = "mxbai-embed-large:latest", base_url="http://localhost:11434"):
        self.model = model
        self.url = f"{base_url}/api/embeddings"

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        response = requests.post(self.url, json={"model": self.model, "prompt": text})
        response.raise_for_status()
        return response.json()["embedding"]

# ----- Initialize ChromaDB -----
persist_directory = "./chroma_db"
embedding_function = OllamaEmbeddings()

vector_store = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_function
)

# ----- Create Retriever -----
retriever = vector_store.as_retriever(
    search_type="similarity",  # Cosine similarity search
    search_kwargs={"k": 3}
)

# ----- Query -----
query = input("Enter your query: ")
docs = retriever.get_relevant_documents(query)

# ----- Output -----
print("\n--- Retrieved Documents ---")
for i, doc in enumerate(docs, 1):
    print(f"\nDocument {i}:\n{doc.page_content}")
