import streamlit as st
import os
from pathlib import Path
from src.scraper import scrape_urls
from src.embeddings import create_embeddings
from src.vector_store import initialize_vector_store, get_retriever
from src.qa_chain import create_qa_chain
from utils.config import get_model_name
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(page_title="URL Q&A App", layout="centered", page_icon="🔍")

def main():
    st.title("URL Q&A App")
    st.markdown("""
    Ask questions about the content of web pages! Enter URLs below, and I'll scrape the content and answer your questions based on what's found there.
    """)
    
    # Initialize session state variables if they don't exist
    if "urls_processed" not in st.session_state:
        st.session_state.urls_processed = False
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # API Key input
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter your OpenAI API Key:", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        st.markdown("---")
        st.markdown("### How it works")
        st.markdown("""
        1. Enter one or more URLs
        2. The app scrapes the content
        3. Ask questions about the content
        4. Get answers based only on the scraped data
        """)

    # URL Input Section
    st.header("Step 1: Enter URLs")
    urls_input = st.text_area("Enter one or more URLs (one per line):", height=100)
    
    col1, col2 = st.columns([1, 5])
    process_urls = col1.button("Process URLs", type="primary")
    
    if process_urls and urls_input and api_key:
        with st.spinner("Processing URLs..."):
            # Split the input by newlines and strip whitespace
            urls = [url.strip() for url in urls_input.split("\n") if url.strip()]
            
            if not urls:
                st.error("Please enter at least one valid URL.")
                return
                
            try:
                # Create data directory if it doesn't exist
                Path("data/vector_db").mkdir(parents=True, exist_ok=True)
                
                # Scrape content from URLs
                scraped_content = scrape_urls(urls)
                
                if not scraped_content:
                    st.error("Could not extract content from any of the provided URLs.")
                    return
                
                # Create embeddings and store in vector DB
                embeddings = create_embeddings()
                vector_store = initialize_vector_store(scraped_content, embeddings)
                
                # Save the vector_store to session state
                st.session_state.vector_store = vector_store
                st.session_state.urls_processed = True
                
                st.success(f"Successfully processed {len(urls)} URLs!")
                
            except Exception as e:
                st.error(f"An error occurred while processing URLs: {str(e)}")
                return
    
    # Question Answering Section
    st.header("Step 2: Ask Questions")
    question = st.text_input("Ask a question about the content:")
    
    col1, col2 = st.columns([1, 5])
    ask_button = col1.button("Ask", type="primary")
    
    # Show a warning if no URLs have been processed
    if not st.session_state.urls_processed:
        st.warning("Please process URLs before asking questions.")
    
    if ask_button and question and st.session_state.urls_processed and api_key:
        if st.session_state.vector_store is None:
            st.error("Please process URLs first before asking questions.")
            return
            
        with st.spinner("Searching for answer..."):
            try:
                # Get retriever from vector store
                retriever = get_retriever(st.session_state.vector_store)
                
                # Create QA chain
                model_name = get_model_name()
                qa_chain = create_qa_chain(retriever, model_name)
                
                # Log the input to the chain
                logger.info(f"Invoking QA chain with question: {question}")
                
                # FIX: Change the input dictionary key from 'question' to 'query'
                input_dict = {"query": question}
                logger.info(f"Input dictionary: {input_dict}")
                
                # Generate answer using invoke
                response = qa_chain.invoke(input_dict)
                
                # Log the response
                logger.info(f"QA chain response: {response}")
                
                # Print response to terminal
                print("QA Chain Response:")
                print(f"Question: {question}")
                print(f"Answer: {response['result']}")
                if "source_documents" in response:
                    print("Sources:")
                    for i, doc in enumerate(response["source_documents"]):
                        print(f"  Source {i+1}:")
                        print(f"    URL: {doc.metadata.get('source', 'N/A')}")
                        print(f"    Content: {doc.page_content[:300]}...")
                
                # Display answer in Streamlit UI
                st.subheader("Answer:")
                st.write(response["result"])
                
                # # Display sources in Streamlit UI
                # with st.expander("Sources"):
                #     st.markdown(response.get("sources", "No source information available"))
                    
                #     # If we have source documents, display them
                #     if "source_documents" in response:
                #         for i, doc in enumerate(response["source_documents"]):
                #             st.markdown(f"**Source {i+1}:**")
                #             st.markdown(f"**URL:** {doc.metadata.get('source', 'N/A')}")
                #             st.markdown(f"**Content:** {doc.page_content[:300]}...")
                #             st.markdown("---")
                
            except Exception as e:
                logger.error(f"Error in QA chain invocation: {str(e)}")
                st.error(f"An error occurred while generating the answer: {str(e)}")
    
    elif ask_button and not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")

if __name__ == "__main__":
    main()