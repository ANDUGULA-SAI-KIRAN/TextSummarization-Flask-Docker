import re
from typing import List
from src.utils import logger

def clean_text(text: str) -> str:
    """
    Cleans the input text by removing extra whitespace and stripping.
    
    Args:
        text (str): The input text to clean.
        
    Returns:
        str: The cleaned text.
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove literal newlines and tabs that often break JSON
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def concatenate_texts(texts: List[str]) -> str:
    """
    Concatenates a list of text documents into a single string.
    
    Args:
        texts (List[str]): List of texts to concatenate.
        
    Returns:
        str: The concatenated text.
    """
    if not texts:
        return ""
        
    cleaned_texts = []
    for i, t in enumerate(texts):
        cleaned = clean_text(t)
        if cleaned:
            cleaned_texts.append(cleaned)
        else:
            logger.debug(f"Document at index {i} was empty after cleaning.")
            
    logger.info(f"Preprocessing: Combined {len(cleaned_texts)} documents into one stream.")
    return " ".join(cleaned_texts)
