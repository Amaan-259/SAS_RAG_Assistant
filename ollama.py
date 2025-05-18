import streamlit as st
import os
import requests
from dotenv import load_dotenv
import time

from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.schema import Document

# Optional: Use LangChain's Ollama wrapper instead of raw API if needed
# from langchain_community.llms import Ollama

# Load environment variables
load_dotenv()

# ----------------- Streamlit UI Setup -----------------
st.title("SAS Code Assistant")
start_time = time.time()

# ----------------- Custom Ollama Embeddings -----------------
class OllamaEmbeddings(Embeddings):
    def __init__(self, model: str = "mxbai-embed-large:latest", base_url="http://localhost:11434"):
        self.model = model
        self.url = f"{base_url}/api/embeddings"

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        try:
            response = requests.post(self.url, json={"model": self.model, "prompt": text})
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            st.error(f"Embedding error: {e}")
            return [0.0] * 768  # Fallback zero vector (you can adjust dimension)

# ----------------- Initialize Chroma Vector Store -----------------
persist_directory = "./chroma_db"
embedding_function = OllamaEmbeddings()

vector_store = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_function
)

# ----------------- Initialize Chat Session State -----------------
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content="Hello! Ask me anything about SAS.")]

# ----------------- Display Chat History -----------------
for message in st.session_state.messages:
    with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
        st.markdown(message.content)

# ----------------- User Input Prompt -----------------
prompt = st.chat_input("Ask your SAS programming question...")

# ----------------- Ollama LLM Configuration -----------------
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3:8b"

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.messages.append(HumanMessage(content=prompt))

    # ----------------- Document Retrieval -----------------
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(prompt)
    docs_text = "\n\n".join(d.page_content for d in docs)

    # ----------------- Construct Prompt for LLM -----------------
    system_prompt = (
        "You are a SAS Programming Assistant. "
        "You need to answer the presented questions precisely with explanation and code examples where needed. "
        "Mention the source which is the last line of context"
        f"Context:\n{docs_text}\n\n"
        f"Question:\n{prompt}"
    )

    # ----------------- Call LLM -----------------
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": system_prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        answer = result.get("response", "[No response returned]")
    except Exception as e:
        answer = f"Error calling Ollama: {e}"

    # ----------------- Display Assistant Response -----------------
    with st.chat_message("assistant"):
        st.markdown(answer)
        st.session_state.messages.append(AIMessage(content=answer))

    # ----------------- Timing Info -----------------
    end_time = time.time()
    st.write(f"⏱️ Time Taken: {round(end_time - start_time, 2)} seconds")
