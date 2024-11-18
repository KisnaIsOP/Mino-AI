from transformers import pipeline
import numpy as np
from typing import Dict, List, Any
import torch
import spacy
from collections import OrderedDict
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

class LRUCache:
    def __init__(self, maxsize=128):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def get(self, key):
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value  # move to end
            return value
        return None

    def set(self, key, value):
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.maxsize:
            self.cache.popitem(last=False)
        self.cache[key] = value

class PerformanceMetrics:
    def __init__(self):
        self.metrics = {}

    def measure_time(self, func_name):
        import time
        start_time = time.time()
        yield
        end_time = time.time()
        self.metrics[func_name] = end_time - start_time

    def get_metrics(self):
        return self.metrics

class ToneAnalyzer:
    def __init__(self):
        """Initialize the tone analyzer with enhanced models and caching."""
        # Initialize models with GPU support if available
        device = 0 if torch.cuda.is_available() else -1
        
        self.emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True,
            device=device
        )
        
        self.tone_classifier = pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
            return_all_scores=True,
            device=device
        )
        
        self.style_converter = pipeline(
            "text2text-generation",
            model="facebook/bart-large-cnn",
            device=device
        )
        
        # Enhanced tone categories with more nuanced characteristics
        self.tone_categories = {
            'formal': {
                'characteristics': ['professional', 'structured', 'respectful', 'precise', 'objective'],
                'keywords': ['therefore', 'moreover', 'consequently', 'furthermore', 'accordingly'],
                'style_markers': {
                    'sentence_length': 'long',
                    'vocabulary_level': 'advanced',
                    'contractions': False,
                    'personal_pronouns': False
                }
            },
            'casual': {
                'characteristics': ['friendly', 'relaxed', 'approachable', 'conversational', 'personal'],
                'keywords': ['hey', 'cool', 'awesome', 'basically', 'pretty much'],
                'style_markers': {
                    'sentence_length': 'short',
                    'vocabulary_level': 'simple',
                    'contractions': True,
                    'personal_pronouns': True
                }
            },
            'empathetic': {
                'characteristics': ['understanding', 'supportive', 'caring', 'compassionate', 'validating'],
                'keywords': ['understand', 'feel', 'appreciate', 'acknowledge', 'support'],
                'style_markers': {
                    'sentence_length': 'medium',
                    'vocabulary_level': 'moderate',
                    'contractions': True,
                    'personal_pronouns': True
                }
            },
            'assertive': {
                'characteristics': ['confident', 'direct', 'clear', 'decisive', 'authoritative'],
                'keywords': ['definitely', 'certainly', 'absolutely', 'clearly', 'strongly'],
                'style_markers': {
                    'sentence_length': 'medium',
                    'vocabulary_level': 'advanced',
                    'contractions': False,
                    'personal_pronouns': True
                }
            },
            'enthusiastic': {
                'characteristics': ['energetic', 'positive', 'excited', 'passionate', 'dynamic'],
                'keywords': ['amazing', 'fantastic', 'excellent', 'incredible', 'wonderful'],
                'style_markers': {
                    'sentence_length': 'varied',
                    'vocabulary_level': 'moderate',
                    'contractions': True,
                    'personal_pronouns': True
                }
            }
        }
        
        # Initialize cache for tone conversion
        self.conversion_cache = LRUCache(maxsize=1000)
        
        # Load spaCy model for linguistic analysis
        self.nlp = spacy.load('en_core_web_trf')
        
        # Initialize performance metrics
        self.metrics = PerformanceMetrics()
    
    async def analyze_tone(self, text: str) -> Dict[str, Any]:
        """Enhanced tone analysis with performance optimization."""
        try:
            with self.metrics.measure_time('tone_analysis'):
                # Get emotion scores with caching
                cache_key = f"emotion_{hash(text)}"
                emotion_scores = self.conversion_cache.get(cache_key)
                if not emotion_scores:
                    emotion_scores = await self._get_emotion_scores(text)
                    self.conversion_cache.set(cache_key, emotion_scores)
                
                # Get enhanced tone profile
                tone_profile = await self._get_enhanced_tone_profile(text)
                
                # Determine dominant and secondary tones
                tones = self._get_ranked_tones(tone_profile)
                
                # Generate tone-specific suggestions
                suggestions = await self._generate_enhanced_suggestions(text, tone_profile)
                
                # Generate alternative phrasings
                alternatives = await self._generate_enhanced_alternatives(text, tone_profile)
                
                return {
                    'tones': tones,
                    'tone_profile': tone_profile,
                    'suggestions': suggestions,
                    'alternatives': alternatives,
                    'metrics': self.metrics.get_metrics()
                }
                
        except Exception as e:
            logger.error(f"Error in tone analysis: {str(e)}")
            raise
    
    async def convert_tone(
        self,
        text: str,
        target_tone: str,
        preserve_meaning: bool = True
    ) -> Dict[str, Any]:
        """Convert text to target tone while preserving meaning."""
        try:
            cache_key = f"conversion_{hash(text)}_{target_tone}"
            cached_result = self.conversion_cache.get(cache_key)
            if cached_result:
                return cached_result
            
            # Analyze current tone
            current_tone = await self.analyze_tone(text)
            
            # Generate tone conversion prompt
            prompt = self._generate_conversion_prompt(
                text,
                current_tone,
                target_tone,
                preserve_meaning
            )
            
            # Convert tone using style converter
            converted = self.style_converter(
                prompt,
                max_length=150,
                min_length=10,
                do_sample=True,
                top_p=0.9,
                temperature=0.7
            )[0]['generated_text']
            
            # Verify tone preservation
            verification = await self._verify_tone_conversion(
                original_text=text,
                converted_text=converted,
                target_tone=target_tone
            )
            
            result = {
                'converted_text': converted,
                'original_tone': current_tone['tones'],
                'target_tone': target_tone,
                'verification': verification
            }
            
            self.conversion_cache.set(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error in tone conversion: {str(e)}")
            raise
    
    async def _get_enhanced_tone_profile(self, text: str) -> Dict[str, Any]:
        """Generate enhanced tone profile with linguistic analysis."""
        doc = self.nlp(text)
        
        # Analyze linguistic features
        linguistic_features = {
            'sentence_length': np.mean([len(sent) for sent in doc.sents]),
            'vocabulary_complexity': self._calculate_vocabulary_complexity(doc),
            'formality_markers': self._extract_formality_markers(doc),
            'personal_pronouns': self._count_personal_pronouns(doc),
            'sentiment_markers': self._extract_sentiment_markers(doc)
        }
        
        # Get emotion distribution
        emotions = await self._get_emotion_scores(text)
        
        # Calculate tone characteristics
        characteristics = []
        for tone, info in self.tone_categories.items():
            score = self._calculate_tone_match(
                linguistic_features,
                emotions,
                info['style_markers']
            )
            characteristics.append({
                'tone': tone,
                'score': score,
                'features': self._extract_tone_features(text, info)
            })
        
        return {
            'linguistic_features': linguistic_features,
            'emotions': emotions,
            'characteristics': characteristics
        }
    
    async def _verify_tone_conversion(
        self,
        original_text: str,
        converted_text: str,
        target_tone: str
    ) -> Dict[str, Any]:
        """Verify the quality of tone conversion."""
        # Analyze converted text tone
        converted_analysis = await self.analyze_tone(converted_text)
        
        # Calculate semantic similarity
        similarity = self._calculate_semantic_similarity(
            original_text,
            converted_text
        )
        
        # Verify target tone match
        tone_match = self._calculate_tone_match_score(
            converted_analysis['tone_profile'],
            target_tone
        )
        
        return {
            'semantic_preservation': similarity,
            'tone_match': tone_match,
            'success_rate': (similarity + tone_match) / 2
        }
    
    def _calculate_vocabulary_complexity(self, doc) -> float:
        """Calculate vocabulary complexity score."""
        word_lengths = [len(token.text) for token in doc if token.is_alpha]
        unique_words = len(set(token.text.lower() for token in doc if token.is_alpha))
        total_words = len([token for token in doc if token.is_alpha])
        
        return {
            'avg_word_length': np.mean(word_lengths),
            'lexical_diversity': unique_words / total_words if total_words > 0 else 0
        }
    
    def _extract_formality_markers(self, doc) -> Dict[str, int]:
        """Extract formal language markers."""
        markers = {
            'conjunctions': len([t for t in doc if t.pos_ == 'CCONJ']),
            'prepositions': len([t for t in doc if t.pos_ == 'ADP']),
            'determiners': len([t for t in doc if t.pos_ == 'DET']),
            'adverbs': len([t for t in doc if t.pos_ == 'ADV'])
        }
        return markers
    
    def _count_personal_pronouns(self, doc) -> Dict[str, int]:
        """Count personal pronouns by type."""
        return {
            'first_person': len([t for t in doc if t.text.lower() in ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']]),
            'second_person': len([t for t in doc if t.text.lower() in ['you', 'your', 'yours']]),
            'third_person': len([t for t in doc if t.text.lower() in ['he', 'him', 'his', 'she', 'her', 'hers', 'they', 'them', 'their', 'theirs']])
        }
    
    def _extract_sentiment_markers(self, doc) -> Dict[str, List[str]]:
        """Extract sentiment-bearing words."""
        return {
            'positive': [token.text for token in doc if token.sentiment > 0],
            'negative': [token.text for token in doc if token.sentiment < 0]
        }
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)
        return doc1.similarity(doc2)
    
    def _calculate_tone_match_score(
        self,
        tone_profile: Dict[str, Any],
        target_tone: str
    ) -> float:
        """Calculate how well the text matches the target tone."""
        target_characteristics = self.tone_categories[target_tone]['characteristics']
        matching_chars = sum(
            1 for char in tone_profile['characteristics']
            if char['tone'] == target_tone
        )
        return matching_chars / len(target_characteristics)
