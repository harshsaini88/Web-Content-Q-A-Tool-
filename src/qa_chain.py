"""
Question-answering chain module using LangChain.
"""
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define prompt template for the QA chain
QA_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based only on the provided context. 
If you don't know the answer based on the context, say "I don't have enough information to answer this question."
Do not use any other knowledge beyond what's in the provided context.

Context: {context}

Question: {question}

Answer:"""

def create_qa_chain(retriever: BaseRetriever, model_name: str = "gpt-3.5-turbo"):
    """
    Create a question-answering chain with a retriever.
    
    Args:
        retriever: The retriever to use for fetching relevant documents
        model_name: The name of the OpenAI model to use
        
    Returns:
        A RetrievalQA chain
    """
    logger.info("Creating QA chain...")
    # Create the LLM
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=0,
    )
    
    # Create prompt
    prompt = PromptTemplate(
        template=QA_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    
    # Create RetrievalQA chain with correct input key
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    # Add an ingestion method to handle the mismatch between 'question' and 'query'
    # RetrievalQA expects 'query' but we're using 'question' in our code
    class QuestionHandler:
        def __init__(self, chain):
            self.chain = chain
            
        def invoke(self, input_dict):
            # If 'question' is in input_dict, rename it to 'query'
            if "question" in input_dict and "query" not in input_dict:
                input_dict = {"query": input_dict["question"]}
            return self.chain.invoke(input_dict)
    
    # Wrap the chain with our handler
    qa_with_handler = QuestionHandler(qa)
    
    logger.info(f"QA chain created with model: {model_name}")
    return qa_with_handler