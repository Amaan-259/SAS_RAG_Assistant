import langchain
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
import time

start=time.time()
DATA_Path=r"C:\Users\amaan\OneDrive\Documents\Final_rag\pdf_files"
def load_doc():
    document_loader= PyPDFDirectoryLoader(DATA_Path)
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

documents=load_doc()
chunks=split_documents(documents)

print(chunks[0])

with open(r"C:\Users\amaan\OneDrive\Documents\Final_rag\text_files\file2.txt", "w", encoding="utf-8") as f:
    for chunk in chunks:
        # Write each chunk's content followed by a separator
        f.write(chunk.page_content)
        f.write("\n")
        f.write(chunk.metadata["title"])
        f.write("\n\n--- DOCUMENT SEPARATOR ---\n\n")

# Function to load documents from file.txt
def load_from_text_file(file_path):
    documents = []
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        # Split by the separator we used when creating the file
        chunks = content.split("--- DOCUMENT SEPARATOR ---")
        
        for i, chunk in enumerate(chunks):
            if chunk.strip():  # Skip empty chunks
                doc = Document(
                    page_content=chunk.strip(),
                    metadata={"source": file_path, "chunk": i}
                )
                documents.append(doc)
    
    return documents

# Function to split documents if needed
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

# Function to create vector database from documents
def create_vector_db(documents, db_directory):
    # Initialize the Ollama embeddings
    embeddings = OllamaEmbeddings(
        base_url="http://localhost:11434",
        model="mxbai-embed-large:latest"  # Use an embedding model you've pulled in Ollama
    )
    
    # Create Chroma vector store
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=db_directory
    )
    
    # Persist the database to disk
    vectordb.persist()
    
    return vectordb

# Main execution
if __name__ == "__main__":
    # Path to your text file
    text_file_path = "file.txt"
    
    # Directory where vector DB will be stored
    vector_db_directory = "chroma_db"
    
    # Load documents from text file
    print("Loading documents from text file...")
    documents = load_from_text_file(text_file_path)
    print(f"Loaded {len(documents)} documents")
    
    # Create vector database
    print("Creating vector database...")
    vectordb = create_vector_db(documents, vector_db_directory)
    print(f"Vector database created and saved to {vector_db_directory}")
    

end=time.time()

print("Time take to convert pdf files to embeddings:"+str(start-end))