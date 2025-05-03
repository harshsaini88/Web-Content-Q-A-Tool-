"""
URL content scraper module.
"""
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_url_content(url: str) -> Optional[str]:
    """
    Fetch content from a URL.
    
    Args:
        url: The URL to fetch content from
        
    Returns:
        The text content of the URL or None if there was an error
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        logger.error(f"Error fetching URL {url}: {str(e)}")
        return None

def extract_text_from_html(html_content: str) -> str:
    """
    Extract main text content from HTML.
    
    Args:
        html_content: The HTML content to parse
        
    Returns:
        Extracted text content
    """
    try:
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Remove script and style elements
        for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            script_or_style.decompose()
        
        # Get text content
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up text (remove extra whitespace)
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text from HTML: {str(e)}")
        return ""

def split_text_into_chunks(text: str, url: str) -> List[Document]:
    """
    Split text into chunks for processing.
    
    Args:
        text: The text to split
        url: Source URL for metadata
        
    Returns:
        List of Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = text_splitter.create_documents([text], metadatas=[{"source": url}])
    return chunks

def scrape_urls(urls: List[str]) -> List[Document]:
    """
    Scrape content from a list of URLs and prepare for embedding.
    
    Args:
        urls: List of URLs to scrape
        
    Returns:
        List of Document objects ready for embeddings
    """
    all_documents = []
    
    for url in urls:
        logger.info(f"Scraping URL: {url}")
        html_content = fetch_url_content(url)
        
        if not html_content:
            logger.warning(f"Skipping URL {url} - could not fetch content")
            continue
            
        text_content = extract_text_from_html(html_content)
        
        if not text_content:
            logger.warning(f"Skipping URL {url} - could not extract text content")
            continue
            
        documents = split_text_into_chunks(text_content, url)
        all_documents.extend(documents)
        
        logger.info(f"Successfully processed {url} - extracted {len(documents)} chunks")
    
    return all_documents