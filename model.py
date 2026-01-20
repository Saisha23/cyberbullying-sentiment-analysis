"""
Model Module: Cyberbullying Detection & Sentiment Analysis
===========================================================

This module handles:
1. Loading pre-trained transformer models from HuggingFace
2. Running inference for cyberbullying detection
3. Running inference for sentiment analysis
4. Returning predictions with confidence scores

Why Transformer Models (BERT/DistilBERT)?
------------------------------------------
- Pre-trained on massive text corpora (Wikipedia, Books)
- Context-aware: Understands meaning based on surrounding words
- Transfer Learning: Leverages learned linguistic patterns
- State-of-the-art accuracy with less training data
- Built for production use with excellent documentation

Real-World Use Cases:
- Social Media Moderation (Twitter, Facebook, Instagram)
- Customer Service Monitoring (detect hostile interactions)
- Online Gaming Communities (reduce toxic behavior)
- School/Corporate Environments (safety monitoring)

Limitations & Future Improvements:
- Context blindness: May not understand sarcasm or multi-turn conversations
- Language-specific: Primarily English-focused
- Future: Multi-modal analysis (text + images), language detection, fine-tuning on domain-specific data
"""

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class CyberbullyingDetector:
    """
    Detects cyberbullying and toxic language in text using transformer models.
    
    Models Used:
    - Sentiment: distilbert-base-uncased-finetuned-sst-2-english
    - Hate Speech Detection: distilbert-base-uncased-finetuned-sst-2-english (baseline)
    
    Both models are DistilBERT variants - lighter, faster versions of BERT
    with 40% fewer parameters but 97% of performance.
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize the detector with pre-trained models.
        
        Args:
            device (str): "cuda" for GPU, "cpu" for CPU
        """
        self.device = device
        self.sentiment_pipeline = None
        self.hate_speech_pipeline = None
        self._load_models()
    
    def _load_models(self):
        """
        Loads pre-trained models from HuggingFace model hub.
        Uses caching to avoid re-downloading.
        """
        try:
            # Sentiment Analysis Pipeline
            # Finetuned on SST-2 dataset (Stanford Sentiment Treebank)
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if self.device == "cuda" and torch.cuda.is_available() else -1
            )
            logger.info("✓ Sentiment model loaded successfully")
            
            # For cyberbullying, we use a zero-shot classification approach
            # This is more robust than a single fine-tuned model
            self.hate_speech_pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.device == "cuda" and torch.cuda.is_available() else -1
            )
            logger.info("✓ Cyberbullying detection model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def detect_sentiment(self, text: str) -> Dict:
        """
        Analyzes sentiment of the input text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict with keys:
                - label: "POSITIVE", "NEGATIVE"
                - score: confidence (0-1)
                - readable: human-readable format
        """
        if not text or len(text.strip()) == 0:
            return {"label": "NEUTRAL", "score": 0.0, "readable": "No text provided"}
        
        try:
            result = self.sentiment_pipeline(text, top_k=1)[0]
            
            # Convert to more readable labels
            label_map = {
                "POSITIVE": "Positive 😊",
                "NEGATIVE": "Negative 😞"
            }
            
            return {
                "label": result["label"],
                "score": round(result["score"], 4),
                "readable": label_map.get(result["label"], result["label"])
            }
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {"label": "ERROR", "score": 0.0, "readable": "Analysis failed"}
    
    def detect_cyberbullying(self, text: str) -> Dict:
        """
        Detects cyberbullying and toxic language using zero-shot classification.
        
        Zero-Shot Classification:
        - No need for labeled training data
        - Works by computing similarity to predefined labels
        - More generalizable than traditional supervised learning
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict with keys:
                - is_bullying: Boolean prediction
                - label: "Bullying" or "Non-Bullying"
                - score: confidence (0-1)
                - readable: human-readable format with emoji
        """
        if not text or len(text.strip()) == 0:
            return {
                "is_bullying": False,
                "label": "Non-Bullying",
                "score": 0.0,
                "readable": "No text provided for analysis"
            }
        
        try:
            # Define candidate labels for classification - more nuanced labels to reduce false positives
            candidate_labels = [
                "This text contains harassment, threats, or abusive language targeting someone",
                "This text is respectful, kind, or neutral communication"
            ]
            
            result = self.hate_speech_pipeline(
                text,
                candidate_labels,
                multi_class=False
            )
            
            # Result is ordered by confidence
            top_label = result["labels"][0]
            confidence = result["scores"][0]
            
            # Only flag as bullying if:
            # 1. Top label is harassment/threats AND
            # 2. Confidence is above 0.65 threshold (reduces false positives)
            is_bullying = ("harassment" in top_label.lower() or "threats" in top_label.lower() or "abusive" in top_label.lower()) and confidence > 0.65
            
            label = "Bullying ⚠️" if is_bullying else "Non-Bullying ✓"
            
            return {
                "is_bullying": is_bullying,
                "label": "Bullying" if is_bullying else "Non-Bullying",
                "score": round(confidence, 4),
                "readable": label
            }
        except Exception as e:
            logger.error(f"Cyberbullying detection error: {e}")
            return {
                "is_bullying": False,
                "label": "Non-Bullying",
                "score": 0.0,
                "readable": "Analysis failed"
            }
    
    def predict(self, text: str) -> Dict:
        """
        Complete prediction pipeline: sentiment + cyberbullying detection.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict containing both sentiment and cyberbullying results
        """
        sentiment = self.detect_sentiment(text)
        cyberbullying = self.detect_cyberbullying(text)
        
        return {
            "sentiment": sentiment,
            "cyberbullying": cyberbullying,
            "input_text": text
        }


# Global model instance (singleton pattern for efficiency)
_detector_instance = None


def get_detector(device: str = "cpu") -> CyberbullyingDetector:
    """
    Lazy-loads detector instance to save memory and startup time.
    Uses singleton pattern to avoid reloading models.
    
    Args:
        device (str): "cuda" or "cpu"
        
    Returns:
        CyberbullyingDetector: Shared detector instance
    """
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = CyberbullyingDetector(device=device)
    return _detector_instance
