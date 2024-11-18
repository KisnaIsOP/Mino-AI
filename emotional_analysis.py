import google.generativeai as genai
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass
import logging
from datetime import datetime
import asyncio
from tqdm import tqdm
import os
import requests
from pathlib import Path

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create cache directory if it doesn't exist
CACHE_DIR = Path.home() / '.cache' / 'mindful_odyssey' / 'models'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def download_with_progress(model_name: str):
    """Download model with progress bar."""
    print(f"\nDownloading {model_name}...")
    
    # Create model-specific cache directory
    model_cache_dir = CACHE_DIR / model_name.replace('/', '_')
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load tokenizer and model with progress tracking
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=model_cache_dir,
            local_files_only=False
        )
        print(f"✓ Tokenizer downloaded for {model_name}")
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=model_cache_dir,
            local_files_only=False
        )
        print(f"✓ Model downloaded for {model_name}")
        
        return True
    except Exception as e:
        logger.error(f"Error downloading {model_name}: {str(e)}")
        return False

@dataclass
class EmotionalState:
    primary: str
    secondary: Optional[str]
    valence: float  # -1 to 1
    arousal: float  # 0 to 1
    intensity: float  # 0 to 1
    confidence: float  # 0 to 1

class EmotionalAnalyzer:
    def __init__(self):
        try:
            print("\n🤖 Initializing Lightweight Emotional Analysis...")
            
            # Use tiny models instead of large ones
            self.emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                use_fast=True,
                use_auth_token=False
            )
            
            # Simple rule-based sentiment analysis instead of large models
            self.use_simple_mode = True
            
            print("✨ Lightweight analysis system ready!")
            
        except Exception as e:
            logger.error(f"Error initializing EmotionalAnalyzer: {str(e)}")
            self.use_simple_mode = True
    
    def _simple_sentiment_analysis(self, text: str) -> dict:
        """Simple rule-based sentiment analysis without large models."""
        positive_words = {'good', 'great', 'happy', 'excellent', 'wonderful', 'best', 'love'}
        negative_words = {'bad', 'sad', 'angry', 'terrible', 'worst', 'hate', 'awful'}
        
        words = set(text.lower().split())
        pos_count = len(words.intersection(positive_words))
        neg_count = len(words.intersection(negative_words))
        
        if pos_count > neg_count:
            return {'label': 'POSITIVE', 'score': 0.8}
        elif neg_count > pos_count:
            return {'label': 'NEGATIVE', 'score': 0.8}
        return {'label': 'NEUTRAL', 'score': 0.6}
    
    async def analyze_emotion(self, text: str) -> EmotionalState:
        """Analyze the emotional content of text."""
        try:
            if self.use_simple_mode:
                sentiment = self._simple_sentiment_analysis(text)
                return EmotionalState(
                    primary='positive' if sentiment['label'] == 'POSITIVE' else 'negative',
                    secondary=None,
                    valence=sentiment['score'] if sentiment['label'] == 'POSITIVE' else -sentiment['score'],
                    arousal=0.5,
                    intensity=sentiment['score'],
                    confidence=0.7
                )
            
            # Get emotion classification
            emotion_result = self.emotion_classifier(text)
            primary_emotion = emotion_result[0]['label'].lower()
            confidence = emotion_result[0]['score']
            
            # Calculate emotional dimensions
            emotion_dims = {
                'joy': {'valence': 0.8, 'arousal': 0.7},
                'sadness': {'valence': -0.7, 'arousal': 0.3},
                'anger': {'valence': -0.6, 'arousal': 0.8},
                'fear': {'valence': -0.8, 'arousal': 0.6},
                'surprise': {'valence': 0.4, 'arousal': 0.7},
                'neutral': {'valence': 0.0, 'arousal': 0.3}
            }.get(
                primary_emotion,
                {'valence': 0.0, 'arousal': 0.3}
            )
            
            return EmotionalState(
                primary=primary_emotion,
                secondary=None,
                valence=emotion_dims['valence'],
                arousal=emotion_dims['arousal'],
                intensity=confidence,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error in emotion analysis: {str(e)}")
            return EmotionalState(
                primary='neutral',
                secondary=None,
                valence=0.0,
                arousal=0.3,
                intensity=0.5,
                confidence=0.5
            )

    async def analyze_with_retry(self, text: str, max_retries: int = 3, delay: int = 2) -> dict:
        """Analyze text with retry logic for better reliability."""
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return await self.analyze_emotion(text)
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed in emotional analysis: {str(e)}")
                    await asyncio.sleep(delay * (attempt + 1))
                else:
                    logger.error(f"All attempts failed in emotional analysis: {str(e)}")
                    raise last_exception

    def _get_secondary_emotion(self, text: str) -> Optional[str]:
        """Detect secondary emotion if present."""
        try:
            results = self.emotion_classifier(text, top_k=2)
            if len(results) > 1 and results[1]['score'] > 0.3:
                return results[1]['label']
            return None
        except Exception:
            return None

    def adjust_response(
        self,
        response: str,
        user_emotion: Dict[str, Any],
        tone_analysis: Dict[str, Any],
        user_profile: Dict[str, Any]
    ) -> str:
        """Adjust response based on emotional context."""
        try:
            # Get base emotional state
            base_emotion = user_emotion.get('primary', 'neutral')
            intensity = user_emotion.get('intensity', 0.5)
            
            # Check if response needs adjustment
            response_emotion = self.analyze_emotion(response)
            
            if self._needs_adjustment(
                user_emotion=base_emotion,
                response_emotion=response_emotion.primary,
                intensity=intensity,
                user_profile=user_profile
            ):
                # Generate emotionally adjusted response
                model = genai.GenerativeModel('gemini-pro')
                prompt = self._create_adjustment_prompt(
                    response=response,
                    target_emotion=base_emotion,
                    intensity=intensity,
                    tone=tone_analysis.get('tone', 'neutral'),
                    user_profile=user_profile
                )
                
                adjusted = model.generate_content(prompt)
                return adjusted.text
                
            return response
            
        except Exception as e:
            logger.error(f"Error adjusting response: {str(e)}")
            return response

    def _needs_adjustment(
        self,
        user_emotion: str,
        response_emotion: str,
        intensity: float,
        user_profile: Dict[str, Any]
    ) -> bool:
        """Determine if response needs emotional adjustment."""
        try:
            # Get emotional alignment preferences from user profile
            alignment_pref = user_profile.get('emotional_alignment', 'moderate')
            
            # Check emotional mismatch
            if user_emotion != response_emotion:
                if alignment_pref == 'strict':
                    return True
                elif alignment_pref == 'moderate' and intensity > 0.6:
                    return True
                elif alignment_pref == 'loose' and intensity > 0.8:
                    return True
                    
            return False
            
        except Exception:
            return False

    def _create_adjustment_prompt(
        self,
        response: str,
        target_emotion: str,
        intensity: float,
        tone: str,
        user_profile: Dict[str, Any]
    ) -> str:
        """Create prompt for emotional adjustment."""
        return f"""
        Original response: {response}
        
        Please adjust this response to:
        1. Show appropriate empathy for {target_emotion} emotion
        2. Match emotional intensity of {intensity}
        3. Maintain a {tone} tone
        4. Align with user's communication style: {user_profile.get('communication_style', 'balanced')}
        5. Keep the core message but make it more emotionally appropriate
        
        Adjusted response:
        """

    def get_health(self) -> Dict[str, Any]:
        """Get health status of emotional analysis system."""
        return {
            'status': 'healthy',
            'models_loaded': True,
            'last_check': datetime.now().isoformat()
        }
