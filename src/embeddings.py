"""
Embeddings module for creating vector representations of text.
"""
import os
from langchain_openai import OpenAIEmbeddings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_embeddings():
    """
    Create an embeddings model instance.
    
    Returns:
        An embeddings model instance for generating text embeddings
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    
    logger.info("Initializing OpenAI embeddings model")
    
    # Create embeddings model with OpenAI
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=api_key,
    )
    
    return embeddings