import google.generativeai as genai
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import Dict, List, Any, Optional
import torch
import json
import logging
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import user_profiling
import emotional_analysis
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Structured analysis result"""
    intent: Dict[str, float]
    sentiment: Dict[str, float]
    emotions: Dict[str, float]
    topics: List[str]
    key_points: List[str]
    context_relevance: float
    suggested_actions: List[str]
    confidence: float
    user_profile: Dict[str, Any]

class AdvancedAnalyzer:
    def __init__(self, model_name: str = 'gemini-pro'):
        # Initialize Gemini
        self.gemini = genai.GenerativeModel(model_name)
        
        # Enhanced NLP Models
        self.intent_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1
        )
        
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="siebert/sentiment-roberta-large-english",
            device=0 if torch.cuda.is_available() else -1
        )
        
        self.emotion_detector = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device=0 if torch.cuda.is_available() else -1
        )
        
        self.entity_recognizer = pipeline(
            "ner",
            model="xlm-roberta-large-finetuned-conll03-english",
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1
        )
        
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Load spaCy for advanced NLP tasks
        self.nlp = spacy.load("en_core_web_trf")
        
        # Initialize context memory
        self.context_memory = defaultdict(list)
        self.max_context_length = 10
        
        # Enhanced intent categories
        self.intent_categories = [
            "question_asking",
            "information_seeking",
            "task_completion",
            "clarification_request",
            "problem_reporting",
            "feedback_giving",
            "general_conversation",
            "emotional_expression",
            "opinion_sharing",
            "suggestion_making",
            "agreement_expression",
            "disagreement_expression",
            "gratitude_expression",
            "apology_making"
        ]
        
        # Initialize components
        self.user_manager = user_profiling.UserProfileManager()
        self.emotional_analyzer = emotional_analysis.EmotionalAnalyzer()
        
        logger.info("Advanced Analyzer initialized with enhanced NLP models")
    
    async def analyze_message(
        self,
        message: str,
        conversation_id: str,
        user_id: str,
        context: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Enhanced message analysis using multiple advanced models"""
        try:
            # Update context memory
            if context:
                self.update_context_memory(conversation_id, context)
            
            # Get user profile and preferences
            profile = self.user_manager.get_or_create_profile(user_id)
            personalization = self.user_manager.get_personalization_params(user_id)
            
            # Parallel analysis tasks
            tasks = [
                self._analyze_intent(message),
                self._analyze_sentiment(message),
                self._analyze_emotions(message),
                self._analyze_entities(message),
                self._generate_summary(message),
                self._analyze_syntax(message),
                self._analyze_context_relevance(message, conversation_id)
            ]
            
            # Execute tasks concurrently
            results = await asyncio.gather(*tasks)
            
            # Combine results
            analysis = {
                'intent': results[0],
                'sentiment': results[1],
                'emotions': results[2],
                'entities': results[3],
                'summary': results[4],
                'syntax': results[5],
                'context_relevance': results[6]
            }
            
            # Generate advanced insights
            insights = await self._generate_insights(analysis, message, profile)
            analysis['insights'] = insights
            
            # Calculate confidence scores
            confidence = self._calculate_confidence(analysis)
            analysis['confidence'] = confidence
            
            return analysis
            
        except Exception as e:
            logger.error(f"Advanced analysis error: {str(e)}")
            raise
    
    async def _analyze_intent(self, message: str) -> Dict[str, Any]:
        """Enhanced intent analysis with multi-label classification"""
        try:
            results = self.intent_classifier(
                message,
                candidate_labels=self.intent_categories,
                multi_label=True
            )
            
            # Filter relevant intents
            relevant_intents = [
                {'label': label, 'score': score}
                for label, score in zip(results['labels'], results['scores'])
                if score > 0.3  # Confidence threshold
            ]
            
            return {
                'intents': relevant_intents,
                'primary_intent': max(relevant_intents, key=lambda x: x['score'])
            }
        except Exception as e:
            logger.error(f"Intent analysis error: {str(e)}")
            return {'intents': [], 'primary_intent': None}

    async def _analyze_sentiment(self, message: str) -> Dict[str, Any]:
        """Advanced sentiment analysis with aspect-based sentiment"""
        try:
            # Overall sentiment
            sentiment = self.sentiment_analyzer(message)[0]
            
            # Aspect-based sentiment using spaCy
            doc = self.nlp(message)
            aspects = {}
            
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'PRODUCT', 'EVENT']:
                    aspect_text = message[max(0, ent.start_char-50):min(len(message), ent.end_char+50)]
                    aspect_sentiment = self.sentiment_analyzer(aspect_text)[0]
                    aspects[ent.text] = {
                        'sentiment': aspect_sentiment['label'],
                        'score': aspect_sentiment['score']
                    }
            
            return {
                'overall': {
                    'label': sentiment['label'],
                    'score': sentiment['score']
                },
                'aspects': aspects
            }
        except Exception as e:
            logger.error(f"Sentiment analysis error: {str(e)}")
            return {'overall': {'label': 'NEUTRAL', 'score': 0.5}, 'aspects': {}}

    async def _analyze_emotions(self, message: str) -> Dict[str, Any]:
        """Enhanced emotion analysis with intensity and transitions"""
        try:
            emotions = self.emotion_detector(message)[0]
            
            # Calculate emotional intensity
            intensity = sum(e['score'] for e in emotions if e['score'] > 0.2)
            
            # Get dominant emotions (those with score > 0.2)
            dominant = [
                {'emotion': e['label'], 'score': e['score']}
                for e in emotions
                if e['score'] > 0.2
            ]
            
            return {
                'emotions': dominant,
                'intensity': intensity,
                'complexity': len(dominant)
            }
        except Exception as e:
            logger.error(f"Emotion analysis error: {str(e)}")
            return {'emotions': [], 'intensity': 0, 'complexity': 0}

    async def _analyze_entities(self, message: str) -> Dict[str, Any]:
        """Advanced entity recognition with relationship mapping"""
        try:
            # Get named entities
            entities = self.entity_recognizer(message)
            
            # Process with spaCy for additional insights
            doc = self.nlp(message)
            
            # Extract relationships between entities
            relationships = []
            for token in doc:
                if token.dep_ in ['nsubj', 'dobj', 'pobj']:
                    relationships.append({
                        'source': token.text,
                        'relation': token.dep_,
                        'target': token.head.text
                    })
            
            return {
                'entities': entities,
                'relationships': relationships,
                'key_phrases': [chunk.text for chunk in doc.noun_chunks]
            }
        except Exception as e:
            logger.error(f"Entity analysis error: {str(e)}")
            return {'entities': [], 'relationships': [], 'key_phrases': []}

    async def _generate_summary(self, message: str) -> Dict[str, Any]:
        """Generate abstractive and extractive summaries"""
        try:
            if len(message.split()) < 30:  # Only summarize longer texts
                return {'summary': message, 'type': 'original'}
            
            # Generate abstractive summary
            summary = self.summarizer(
                message,
                max_length=130,
                min_length=30,
                do_sample=False
            )[0]['summary_text']
            
            return {
                'summary': summary,
                'type': 'abstractive',
                'length_ratio': len(summary.split()) / len(message.split())
            }
        except Exception as e:
            logger.error(f"Summary generation error: {str(e)}")
            return {'summary': message[:100] + '...', 'type': 'truncated'}

    async def _analyze_syntax(self, message: str) -> Dict[str, Any]:
        """Advanced syntactic analysis"""
        try:
            doc = self.nlp(message)
            
            return {
                'sentence_structure': [{
                    'text': sent.text,
                    'root_verb': sent.root.text if sent.root.pos_ == 'VERB' else None,
                    'subjects': [tok.text for tok in sent if tok.dep_ == 'nsubj'],
                    'objects': [tok.text for tok in sent if tok.dep_ in ['dobj', 'pobj']]
                } for sent in doc.sents],
                'complexity_metrics': {
                    'sentence_count': len(list(doc.sents)),
                    'avg_token_length': sum(len(token.text) for token in doc) / len(doc),
                    'dependency_depth': max(token.head.i - token.i for token in doc)
                }
            }
        except Exception as e:
            logger.error(f"Syntax analysis error: {str(e)}")
            return {'sentence_structure': [], 'complexity_metrics': {}}

    async def _generate_insights(
        self,
        analysis: Dict[str, Any],
        message: str,
        profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate advanced insights from combined analysis"""
        try:
            # Combine multiple analysis aspects
            insights = {
                'communication_style': {
                    'formality': self._calculate_formality(analysis),
                    'directness': self._calculate_directness(analysis),
                    'engagement': self._calculate_engagement(analysis)
                },
                'user_alignment': self._calculate_user_alignment(analysis, profile),
                'suggested_approaches': self._generate_approach_suggestions(analysis)
            }
            
            return insights
        except Exception as e:
            logger.error(f"Insight generation error: {str(e)}")
            return {}
    
    def _calculate_formality(self, analysis: Dict[str, Any]) -> float:
        """Calculate formality score based on syntax and vocabulary"""
        # TO DO: implement formality calculation
        return 0.5
    
    def _calculate_directness(self, analysis: Dict[str, Any]) -> float:
        """Calculate directness score based on intent and syntax"""
        # TO DO: implement directness calculation
        return 0.5
    
    def _calculate_engagement(self, analysis: Dict[str, Any]) -> float:
        """Calculate engagement score based on sentiment and emotions"""
        # TO DO: implement engagement calculation
        return 0.5
    
    def _calculate_user_alignment(self, analysis: Dict[str, Any], profile: Dict[str, Any]) -> float:
        """Calculate user alignment score based on sentiment and user profile"""
        # TO DO: implement user alignment calculation
        return 0.5
    
    def _generate_approach_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate approach suggestions based on insights"""
        # TO DO: implement approach suggestion generation
        return []
    
    def _calculate_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall confidence score"""
        confidence_factors = [
            max(analysis['intent']['intents'], key=lambda x: x['score'])['score'],
            analysis['sentiment']['overall']['score'],
            analysis['emotions']['intensity'],
            analysis['context_relevance']
        ]
        
        return float(np.mean(confidence_factors))
    
    def update_context_memory(
        self,
        conversation_id: str,
        context: List[Dict[str, str]]
    ):
        """Update conversation context memory"""
        self.context_memory[conversation_id] = context[-self.max_context_length:]
    
    def get_context_summary(self, conversation_id: str) -> Dict[str, Any]:
        """Get summary of conversation context"""
        if conversation_id not in self.context_memory:
            return {"error": "No context found"}
        
        context = self.context_memory[conversation_id]
        
        # Analyze context
        all_topics = []
        sentiments = []
        
        for msg in context:
            doc = self.nlp(msg['content'])
            # Extract topics
            topics = [
                chunk.text for chunk in doc.noun_chunks
                if chunk.root.pos_ in ['NOUN', 'PROPN']
            ]
            all_topics.extend(topics)
            
            # Analyze sentiment
            sentiment = self.sentiment_analyzer(msg['content'])[0]
            sentiments.append(sentiment['score'])
        
        return {
            'message_count': len(context),
            'common_topics': list(set(all_topics)),
            'average_sentiment': float(np.mean(sentiments)),
            'context_coherence': self._calculate_context_coherence(context)
        }
    
    def _calculate_context_coherence(
        self,
        context: List[Dict[str, str]]
    ) -> float:
        """Calculate coherence of conversation context"""
        if len(context) < 2:
            return 1.0
        
        # Calculate embeddings
        embeddings = [
            self.nlp(msg['content']).vector
            for msg in context
        ]
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i+1].reshape(1, -1)
            )[0][0]
            similarities.append(sim)
        
        return float(np.mean(similarities))

# Initialize analyzer
analyzer = AdvancedAnalyzer()
