"""
Vector store management module.
"""
import os
from typing import List
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_vector_store(documents: List[Document], embeddings: Embeddings) -> VectorStore:
    """
    Initialize a vector store with document embeddings.
    
    Args:
        documents: List of documents to embed
        embeddings: Embeddings model to use
        
    Returns:
        Vector store instance
    """
    logger.info(f"Initializing vector store with {len(documents)} documents")
    
    # We'll use FAISS for this project
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # Save the vector store locally
    vector_store.save_local("data/vector_db/faiss_index")
    
    logger.info("Vector store initialized and saved successfully")
    return vector_store

def load_vector_store(embeddings: Embeddings) -> VectorStore:
    """
    Load a previously saved vector store.
    
    Args:
        embeddings: Embeddings model to use
        
    Returns:
        Vector store instance
    """
    if not os.path.exists("data/vector_db/faiss_index"):
        raise FileNotFoundError("No vector store found. Please process some URLs first.")
    
    logger.info("Loading existing vector store")
    vector_store = FAISS.load_local("data/vector_db/faiss_index", embeddings)
    return vector_store

def get_retriever(vector_store: VectorStore):
    """
    Get a retriever from a vector store.
    
    Args:
        vector_store: Vector store to create retriever from
        
    Returns:
        A retriever instance
    """
    # Create retriever with search parameters
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}  # Return top 4 most similar chunks
    )
    
    return retriever