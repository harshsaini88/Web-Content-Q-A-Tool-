"""
Configuration utilities.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

def get_model_name():
    """
    Get the OpenAI model name to use.
    
    Returns:
        The model name string
    """
    # Default model name
    return os.environ.get("OPENAI_MODEL_NAME", "gpt-3.5-turbo")