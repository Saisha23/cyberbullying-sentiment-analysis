"""
Text Preprocessing Module
==========================
Handles text cleaning and normalization for cyberbullying detection and sentiment analysis.

Key Functions:
- clean_text(): Removes noise, standardizes format
- tokenize_and_normalize(): Prepares text for transformer models

Why Preprocessing Matters:
- Removes noise (URLs, special characters, emojis)
- Standardizes text for consistent model predictions
- Improves model robustness and accuracy
"""

import re
import string


def clean_text(text: str) -> str:
    """
    Cleans and normalizes input text for NLP processing.
    
    Args:
        text (str): Raw input text
        
    Returns:
        str: Cleaned and normalized text
        
    Process:
        1. Convert to lowercase for consistency
        2. Remove URLs
        3. Remove email addresses
        4. Remove extra spaces
        5. Remove leading/trailing whitespace
        
    Note: We preserve @ and # symbols as they provide context to the model
          about mentions and hashtags, which is important for understanding
          targeted harassment patterns.
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    
    # Remove extra whitespace (including multiple dots)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.{2,}', '.', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def preprocess_for_model(text: str, max_length: int = 512) -> str:
    """
    Prepares text for transformer model input.
    
    Args:
        text (str): Input text to preprocess
        max_length (int): Maximum token length (default 512 for BERT-based models)
        
    Returns:
        str: Preprocessed text ready for model inference
        
    Note:
        - Transformer models handle most tokenization internally
        - We just ensure text is clean and at reasonable length
        - Truncation is handled by the tokenizer, not here
    """
    # First, apply basic cleaning
    text = clean_text(text)
    
    # If text is too long, truncate with ellipsis
    if len(text) > max_length:
        text = text[:max_length-3] + "..."
    
    return text


def get_text_statistics(text: str) -> dict:
    """
    Returns basic text statistics for UI display.
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Statistics including word count, character count, and text length info
    """
    words = text.split()
    
    return {
        "word_count": len(words),
        "char_count": len(text),
        "avg_word_length": round(sum(len(w) for w in words) / len(words), 2) if words else 0
    }
