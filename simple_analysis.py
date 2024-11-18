import google.generativeai as genai
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
import logging
import logging.handlers
from dataclasses import dataclass
import json
import random
from collections import deque
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import re

# Configure efficient logging
logger = logging.getLogger('mino_ai')
logger.setLevel(logging.ERROR)

# File handler for critical errors and conversations
error_handler = logging.FileHandler('mino_errors.log')
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
))
logger.addHandler(error_handler)

# Rotating file handler for conversations
conv_handler = logging.handlers.RotatingFileHandler(
    'conversations.log',
    maxBytes=1024*1024,  # 1MB
    backupCount=3
)
conv_handler.setLevel(logging.INFO)
conv_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(message)s'
))

# Custom filter to separate conversations from errors
class ConversationFilter(logging.Filter):
    def filter(self, record):
        return not record.levelno >= logging.ERROR

conv_handler.addFilter(ConversationFilter())
logger.addHandler(conv_handler)

@dataclass
class EmojiConfig:
    min_count: int = 0
    max_count: int = 3
    weights: Dict[str, float] = None

    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                'positive': 1.2,
                'excited': 1.3,
                'neutral': 0.8,
                'confused': 0.6,
                'negative': 0.4
            }

@dataclass
class AnalysisResult:
    intent: str
    sentiment: str
    topics: List[str]
    context_level: str
    suggested_approach: str
    personality_match: str
    timestamp: float

class Interaction(NamedTuple):
    """Lightweight interaction record."""
    message: str
    analysis: Dict[str, Any]
    timestamp: float
    feedback: Optional[str] = None

class ConversationHistory:
    """Efficient conversation history management."""
    def __init__(self, max_size: int = 10):
        self.interactions = deque(maxlen=max_size)
        self.feedback_stats = {
            'positive': 0,
            'negative': 0,
            'neutral': 0
        }

    def add(self, message: str, analysis: Dict[str, Any]) -> None:
        """Add new interaction."""
        self.interactions.append(Interaction(
            message=message,
            analysis=analysis,
            timestamp=time.time()
        ))

    def add_feedback(self, message: str, feedback: str) -> None:
        """Track user feedback for message."""
        for interaction in self.interactions:
            if interaction.message == message:
                # Update feedback stats
                self.feedback_stats[feedback] = self.feedback_stats.get(feedback, 0) + 1
                # Create new interaction with feedback
                new_interaction = Interaction(
                    message=interaction.message,
                    analysis=interaction.analysis,
                    timestamp=interaction.timestamp,
                    feedback=feedback
                )
                # Replace old interaction
                self.interactions.remove(interaction)
                self.interactions.append(new_interaction)
                break

    def get_recent_context(self, limit: int = 3) -> List[Dict[str, Any]]:
        """Get recent interactions for context."""
        recent = list(self.interactions)[-limit:]
        return [
            {
                'message': i.message,
                'sentiment': i.analysis.get('sentiment'),
                'intent': i.analysis.get('intent'),
                'feedback': i.feedback
            }
            for i in recent
        ]

    def get_feedback_summary(self) -> Dict[str, float]:
        """Get feedback statistics."""
        total = sum(self.feedback_stats.values())
        if total == 0:
            return {'positive': 0, 'negative': 0, 'neutral': 0}
        
        return {
            k: round(v/total, 2)
            for k, v in self.feedback_stats.items()
        }

class MessageBatch:
    def __init__(self, max_size: int = 5, max_wait: float = 0.5):
        self.messages: List[Tuple[str, float]] = []
        self.max_size = max_size
        self.max_wait = max_wait
        self.last_process_time = time.time()

    def add(self, message: str) -> bool:
        current_time = time.time()
        self.messages.append((message, current_time))
        return (len(self.messages) >= self.max_size or 
                (self.messages and current_time - self.last_process_time >= self.max_wait))

    def clear(self) -> List[Tuple[str, float]]:
        messages = self.messages
        self.messages = []
        self.last_process_time = time.time()
        return messages

class AdvancedAnalyzer:
    def __init__(self, model_name: str = 'gemini-pro'):
        """Initialize with efficient history tracking."""
        self.model_name = model_name
        self._model = None
        self._executor = None
        self._conversation_history = None
        self._emoji_mappings = None
        self._defaults = None
        self._batch_size = 5
        self._batch_wait = 0.5
        self._initialized = False
        
        # Initialize conversation history
        self.conversation_history = ConversationHistory(max_size=10)
        logger.info(f"Analyzer initialized with model: {model_name}")

    def add_user_feedback(self, message: str, feedback: str) -> None:
        """Track user feedback for continuous improvement."""
        if feedback not in ['positive', 'negative', 'neutral']:
            logger.warning(f"Invalid feedback type: {feedback}")
            return
        
        self.conversation_history.add_feedback(message, feedback)
        logger.info(f"User feedback recorded: {feedback}")

    @property
    def model(self):
        """Lazy load the AI model."""
        if self._model is None:
            try:
                self._model = genai.GenerativeModel(self.model_name)
                logger.info("AI model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading AI model: {str(e)}")
                raise
        return self._model

    @property
    def executor(self):
        """Lazy load the thread executor."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=3)
        return self._executor

    def _initialize_if_needed(self):
        """Initialize heavy components only when needed."""
        if not self._initialized:
            try:
                self._initialize_emoji_mappings()
                self._initialize_defaults()
                self._initialized = True
            except Exception as e:
                logger.error(f"Initialization error: {str(e)}")
                raise

    def _initialize_emoji_mappings(self):
        """Initialize simplified emoji mappings."""
        if self._emoji_mappings is None:
            # Direct personality to emoji mapping
            self._emoji_mappings = {
                # Core personality emojis
                'friendly': '😊',
                'professional': '📊',
                'empathetic': '💫',
                'technical': '💻',
                'casual': '😄',
                
                # Sentiment emojis
                'positive': '👍',
                'negative': '💭',
                'neutral': '✅',
                'excited': '🎉',
                'confused': '🤔',
                
                # Intent emojis
                'question': '❓',
                'statement': '💡',
                'request': '🎯',
                'greeting': '👋',
                'feedback': '📝'
            }

    def _initialize_defaults(self):
        """Initialize default configurations only when needed."""
        if self._defaults is None:
            self._defaults = {
                'greetings': {
                    'casual': ['Hey!', 'Hi there!'],
                    'professional': ['Hello,', 'Greetings,'],
                    'empathetic': ['Hi friend!', 'Hello there!']
                },
                'technical_terms': [
                    'function', 'method', 'class', 
                    'variable', 'api', 'data'
                ]
            }

    async def _analyze_batch(self, messages: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
        """Analyze message batch with lazy initialization."""
        self._initialize_if_needed()
        
        analysis_tasks = []
        analyses = []
        
        for message, timestamp in messages:
            try:
                prompt = f"""Message: "{message}"
                Analyze and respond with JSON only:
                {{
                    "intent": ["question", "statement", "request", "greeting", "feedback"],
                    "sentiment": ["positive", "negative", "neutral", "excited", "confused"],
                    "topic": "main topic or theme",
                    "tone": ["informative", "casual", "professional", "empathetic"],
                    "context": ["basic", "technical", "detailed"]
                }}
                Pick ONE value for each field."""
                
                analysis_tasks.append(self.executor.submit(self.model.generate_content, prompt))
            except Exception as e:
                logger.error(f"Task creation error: {str(e)[:100]}")
                analyses.append(self._get_default_analysis())
                continue

        for task, (message, timestamp) in zip(analysis_tasks, messages):
            try:
                response = await asyncio.wrap_future(task)
                analysis = self._safe_json_loads(response.text)
                enhanced = self._enhance_analysis(analysis, message)
                enhanced['timestamp'] = timestamp
                self._log_conversation(message, enhanced)
                analyses.append(enhanced)
            except Exception as e:
                logger.error(f"Analysis error: {str(e)[:100]}")
                analyses.append(self._get_default_analysis())

        return analyses

    def analyze_message(self, message: str) -> Dict[str, Any]:
        """Analyze a single message with lazy initialization."""
        self._initialize_if_needed()
        current_time = time.time()

        if self.batch_processor.add(message):
            messages = self.batch_processor.clear()
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                analyses = loop.run_until_complete(self._analyze_batch(messages))
                loop.close()
                
                for analysis in analyses:
                    if abs(analysis['timestamp'] - current_time) < 0.1:
                        self.conversation_history.add(message, analysis)
                        return analysis
            except Exception as e:
                logger.error(f"Batch processing error: {str(e)}")

        # Fallback to single message analysis
        try:
            prompt = f"""Message: "{message}"
            Analyze and respond with JSON only:
            {{
                "intent": ["question", "statement", "request", "greeting", "feedback"],
                "sentiment": ["positive", "negative", "neutral", "excited", "confused"],
                "topic": "main topic or theme",
                "tone": ["informative", "casual", "professional", "empathetic"],
                "context": ["basic", "technical", "detailed"]
            }}
            Pick ONE value for each field."""
            
            response = self.model.generate_content(prompt)
            analysis = self._safe_json_loads(response.text)
            
            if analysis:
                enhanced = self._enhance_analysis(analysis, message)
                enhanced['timestamp'] = current_time
                self.conversation_history.add(message, enhanced)
                return enhanced
            
            return self._get_default_analysis()
            
        except Exception as e:
            logger.error(f"Error analyzing message: {str(e)}")
            return self._get_default_analysis()

    def cleanup(self):
        """Clean up resources and save feedback data."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        
        # Save feedback statistics
        try:
            feedback_stats = self.conversation_history.get_feedback_summary()
            with open('feedback_stats.json', 'w') as f:
                json.dump(feedback_stats, f)
            logger.info("Feedback statistics saved")
        except Exception as e:
            logger.error(f"Error saving feedback stats: {str(e)}")
        
        self._model = None
        self._conversation_history = None
        self._emoji_mappings = None
        self._defaults = None
        self._initialized = False
        logger.info("Analyzer resources cleaned up")

    def _get_default_analysis(self) -> Dict[str, Any]:
        """Provide intelligent defaults for analysis."""
        return {
            "intent": "statement",
            "sentiment": "neutral",
            "topic": "general",
            "tone": "casual",
            "context": "basic",
            "timestamp": time.time()
        }

    def _normalize_analysis(self, analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize and validate analysis data with smart defaults."""
        if not analysis:
            return self._get_default_analysis()

        defaults = {
            # Core analysis fields
            "intent": ["statement", "question", "greeting", "request", "feedback"],
            "sentiment": ["neutral", "positive", "negative", "excited", "confused"],
            "tone": ["casual", "professional", "technical", "empathetic"],
            "context": ["basic", "technical", "detailed"],
            
            # Default values
            "default_intent": "statement",
            "default_sentiment": "neutral",
            "default_tone": "casual",
            "default_context": "basic"
        }

        normalized = {}
        
        # Normalize intent
        intent = analysis.get("intent", "").lower()
        normalized["intent"] = (
            intent if intent in defaults["intent"] 
            else defaults["default_intent"]
        )
        
        # Normalize sentiment
        sentiment = analysis.get("sentiment", "").lower()
        normalized["sentiment"] = (
            sentiment if sentiment in defaults["sentiment"]
            else defaults["default_sentiment"]
        )
        
        # Normalize tone
        tone = analysis.get("tone", "").lower()
        normalized["tone"] = (
            tone if tone in defaults["tone"]
            else defaults["default_tone"]
        )
        
        # Normalize context
        context = analysis.get("context", "").lower()
        normalized["context"] = (
            context if context in defaults["context"]
            else defaults["default_context"]
        )
        
        # Handle topic
        topic = analysis.get("topic", "").strip()
        normalized["topic"] = topic if topic else "general"
        
        # Add timestamp
        normalized["timestamp"] = analysis.get("timestamp", time.time())
        
        return normalized

    def _enhance_analysis(self, basic_analysis: Dict[str, Any], message: str) -> Dict[str, Any]:
        """Enhanced analysis with feedback context."""
        analysis = self._normalize_analysis(basic_analysis)
        enhanced = analysis.copy()
        
        # Get recent context
        recent_context = self.conversation_history.get_recent_context()
        feedback_summary = self.conversation_history.get_feedback_summary()
        
        # Adjust tone based on feedback history
        if feedback_summary['negative'] > 0.3:  # If >30% negative feedback
            enhanced['tone'] = 'empathetic'  # More supportive tone
        elif feedback_summary['positive'] > 0.7:  # If >70% positive feedback
            enhanced['tone'] = enhanced.get('tone', 'casual')  # Maintain current style
        
        # Adjust based on recent context
        if recent_context:
            last_interaction = recent_context[-1]
            
            # If last interaction was negative, be more supportive
            if last_interaction['sentiment'] == 'negative':
                enhanced['tone'] = 'empathetic'
            
            # If user is consistently asking questions, be more informative
            if all(i['intent'] == 'question' for i in recent_context):
                enhanced['tone'] = 'technical'
        
        # Direct personality mapping with context awareness
        personality_map = {
            'greeting': 'friendly',
            'question': 'helpful',
            'feedback': 'professional',
            'request': 'supportive',
            'statement': 'casual'
        }
        
        enhanced['personality_match'] = personality_map.get(
            enhanced['intent'], 
            'casual'
        )
        
        # Simplified style mapping
        enhanced['conversation_style'] = (
            'formal' if enhanced['tone'] in ['technical', 'professional']
            else 'casual'
        )
        
        return enhanced

    def enhance_response(self, message: str, base_response: str, analysis: Dict[str, Any]) -> str:
        """Enhanced response with feedback context."""
        try:
            # Get feedback context
            feedback_summary = self.conversation_history.get_feedback_summary()
            recent_context = self.conversation_history.get_recent_context(2)
            
            # Adjust response based on feedback history
            if feedback_summary['negative'] > 0.3:
                base_response = f"I understand your concern. {base_response}"
            elif recent_context and any(i['sentiment'] == 'confused' for i in recent_context):
                base_response = f"Let me clarify: {base_response}"
            
            # Get appropriate emojis
            emojis = self._select_emojis(analysis)
            
            # Format response
            if analysis.get('intent') == 'greeting':
                base_response = f"Hi! {base_response}"
            
            if analysis.get('tone') == 'technical':
                base_response = f"Technical note: {base_response}"
            
            # Add emojis appropriately
            if emojis:
                if analysis.get('intent') == 'greeting':
                    base_response = f"{emojis[0]} {base_response}"
                else:
                    base_response = f"{base_response} {emojis[0]}"
                
                if len(emojis) > 1:
                    base_response = f"{base_response} {emojis[1]}"
            
            return base_response
            
        except Exception as e:
            logger.error(f"Error in response enhancement: {str(e)}")
            return base_response

    def _sanitize_json_string(self, json_str: str) -> str:
        json_str = json_str.strip()
        json_str = json_str.strip('[]').strip('{}')
        return '{' + json_str + '}'

    def _safe_json_loads(self, json_str: str) -> Optional[Dict[str, Any]]:
        """Safely parse JSON with error logging."""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)[:100]}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in JSON parsing: {str(e)[:100]}")
            return None

    def _log_conversation(self, message: str, analysis: Dict[str, Any]):
        """Log conversation with minimal overhead."""
        try:
            # Only log essential information
            log_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "message_type": analysis.get("intent", "unknown"),
                "sentiment": analysis.get("sentiment", "unknown")
            }
            logger.info(f"CONV: {json.dumps(log_data)} - MSG: {message[:100]}...")
        except Exception as e:
            # Only log logging errors at ERROR level
            logger.error(f"Logging error: {str(e)}")

    def _format_response(self, response: str, analysis: Dict[str, Any]) -> str:
        """Format response with appropriate style and structure."""
        # Get basic formatting
        response = self._enhance_response_style(response, analysis)
        
        # Add greeting for new conversations
        if analysis.get('intent') == 'greeting':
            greetings = {
                'casual': ['Hey!', 'Hi there!'],
                'professional': ['Hello,', 'Greetings,'],
                'empathetic': ['Hi friend!', 'Hello there!']
            }
            tone = analysis.get('tone', 'casual')
            greeting = random.choice(greetings.get(tone, greetings['casual']))
            response = f"{greeting} {response}"
        
        # Add appropriate line breaks
        response = response.replace('. ', '.\n')
        response = '\n'.join(line.strip() for line in response.split('\n'))
        
        return response

    def _enhance_response_style(self, response: str, analysis: Dict[str, Any]) -> str:
        """Apply lightweight style enhancements based on analysis."""
        tone = analysis.get('tone', 'casual')
        
        # Add emphasis to key points
        if 'technical' in tone:
            # Add code formatting for technical terms
            import re
            technical_terms = ['function', 'method', 'class', 'variable', 'api', 'data']
            for term in technical_terms:
                response = re.sub(
                    f'\\b{term}\\b', 
                    f'`{term}`', 
                    response, 
                    flags=re.IGNORECASE
                )
        
        # Add emphasis for important points
        if any(marker in response.lower() for marker in ['important', 'note', 'key', 'remember']):
            response = response.replace('Important:', '**Important:**')
            response = response.replace('Note:', '**Note:**')
            response = response.replace('Key point:', '**Key point:**')
        
        return response

    def get_personality_emoji(self, personality: str) -> str:
        """Get single emoji for personality type."""
        return self._emoji_mappings.get(personality, '😄')

    def get_sentiment_emoji(self, sentiment: str) -> str:
        """Get single emoji for sentiment."""
        return self._emoji_mappings.get(sentiment, '✅')

    def get_intent_emoji(self, intent: str) -> str:
        """Get single emoji for message intent."""
        return self._emoji_mappings.get(intent, '💡')

    def _calculate_emoji_count(self, sentiment: str, intent: str) -> int:
        """Simplified emoji count calculation."""
        # Base count of 1 for most messages
        if intent in ['statement', 'question']:
            return 1
        # 2 emojis for greetings and excited messages
        if intent == 'greeting' or sentiment == 'excited':
            return 2
        # Default to 1 emoji
        return 1

    def _select_emojis(self, analysis: Dict[str, Any]) -> List[str]:
        """Select appropriate emojis based on message context."""
        emojis = []
        
        # Get core message attributes
        personality = analysis.get('personality_match', 'casual')
        sentiment = analysis.get('sentiment', 'neutral')
        intent = analysis.get('intent', 'statement')
        
        # Add personality emoji if appropriate
        if personality != 'casual':
            emojis.append(self.get_personality_emoji(personality))
        
        # Add sentiment emoji for emotional messages
        if sentiment not in ['neutral', 'casual']:
            emojis.append(self.get_sentiment_emoji(sentiment))
        
        # Add intent emoji for specific cases
        if intent in ['greeting', 'question']:
            emojis.append(self.get_intent_emoji(intent))
        
        # Ensure at least one emoji
        if not emojis:
            emojis.append(self.get_personality_emoji('casual'))
        
        # Limit total emojis
        return emojis[:2]

class MinoAI:
    """Mino AI - Advanced Conversational Intelligence"""
    def __init__(self):
        self.name = "Mino AI"
        self.identity = {
            "name": "Mino AI",
            "role": "Advanced AI Assistant",
            "personality": "Professional and friendly",
            "creator": {
                "name": "Kisna Raghuvanshi",
                "age": 16,
                "info_share_policy": "on_request_only"
            },
            "capabilities": [
                "Natural conversation",
                "Context awareness",
                "Emotional intelligence",
                "Adaptive learning"
            ]
        }
        self.response_prefix = [
            "As Mino AI, ",
            "Based on my analysis, ",
            "From my perspective, ",
            "In my assessment, "
        ]
        self.restricted_topics = [
            "model_identity",
            "training_data",
            "underlying_technology"
        ]

    def format_response(self, response):
        """Format response to maintain Mino AI identity."""
        # Remove any mentions of other AI models
        response = re.sub(r'(?i)(gemini|google ai|gpt)', 'Mino AI', response)
        
        # Add Mino AI's identity to responses when appropriate
        if not any(prefix in response for prefix in self.response_prefix):
            prefix = random.choice(self.response_prefix)
            response = f"{prefix}{response}"
            
        return response

    def check_restricted_content(self, message):
        """Check if the message contains restricted topics."""
        restricted_patterns = [
            r'(?i)(gemini|gpt|google ai)',
            r'(?i)(training data|model|architecture)',
            r'(?i)(underlying (technology|system|model))'
        ]
        
        for pattern in restricted_patterns:
            if re.search(pattern, message):
                return True
        return False

    def get_standard_response(self):
        """Return standard response for restricted topics."""
        responses = [
            "I am Mino AI, your AI assistant. I focus on helping you rather than discussing my internal details.",
            "As Mino AI, I prefer to focus on how I can help you rather than discussing technical details about AI systems.",
            "I'm Mino AI, and I'm here to assist you. Let's focus on your needs rather than technical details."
        ]
        return random.choice(responses)

    def check_creator_query(self, message):
        """Check if the message is asking about Mino AI's creator."""
        creator_patterns = [
            r'(?i)(who (made|created|developed|built|designed) (you|Mino AI))',
            r'(?i)(who\'?s? (your|Mino AI\'?s?) (creator|developer|maker))',
            r'(?i)(tell me about (your|Mino AI\'?s?) (creator|developer|maker))',
            r'(?i)(who (is|was) behind (your|Mino AI\'?s?) (development|creation))'
        ]
        
        for pattern in creator_patterns:
            if re.search(pattern, message):
                return True
        return False

    def get_creator_response(self):
        """Return response about Mino AI's creator."""
        responses = [
            "I was created by Kisna Raghuvanshi, a 16-year-old developer with a passion for AI technology.",
            "My creator is Kisna Raghuvanshi, a talented 16-year-old who developed me to help people.",
            "I'm proud to say that I was developed by Kisna Raghuvanshi, who is just 16 years old."
        ]
        return random.choice(responses)

    def process_message(self, message):
        """Process incoming message and return appropriate response."""
        if self.check_restricted_content(message):
            return self.get_standard_response()
        elif self.check_creator_query(message):
            return self.get_creator_response()
        else:
            # Process normal message
            return None  # Let the regular processing handle it

class AISettings:
    def __init__(self):
        self.response_style = "Professional"
        self.emotion_sensitivity = 5
        self.context_window = 10
        self.model_name = "Mino"  # Changed from Gemini Pro
        self.load_settings()

    def load_settings(self):
        """Load settings from config file."""
        try:
            with open('ai_settings.json', 'r') as f:
                settings = json.load(f)
                self.__dict__.update(settings)
        except FileNotFoundError:
            self.save_settings()

    def save_settings(self):
        """Save current settings to file."""
        with open('ai_settings.json', 'w') as f:
            json.dump(self.__dict__, f, indent=4)

def change_language_model(model):
    """Change the active language model."""
    models = {
        "1": "Mino Standard",
        "2": "Mino Enhanced",
        "3": "Mino Custom"
    }
    if model in models:
        ai_settings.model_name = models[model]
        ai_settings.save_settings()
        reload_ai_model()
        logging.info(f"Language model changed to: {models[model]}")
        return True
    return False

# Hot-reload functionality
def reload_ai_model():
    """Reload the AI model without server restart."""
    try:
        global genai_model
        genai_model = None  # Clear existing model
        initialize_ai_model()  # Reinitialize
        logging.info("AI model reloaded successfully")
        return True
    except Exception as e:
        logging.error(f"Error reloading AI model: {e}")
        return False

def reload_chat_analysis():
    """Reload chat analysis components."""
    try:
        global chat_analyzer
        chat_analyzer = None
        initialize_chat_analysis()
        logging.info("Chat analysis reloaded successfully")
        return True
    except Exception as e:
        logging.error(f"Error reloading chat analysis: {e}")
        return False

def reload_emotion_detection():
    """Reload emotion detection system."""
    try:
        global emotion_detector
        emotion_detector = None
        initialize_emotion_detection()
        logging.info("Emotion detection reloaded successfully")
        return True
    except Exception as e:
        logging.error(f"Error reloading emotion detection: {e}")
        return False

def reload_all():
    """Reload all AI components."""
    success = all([
        reload_ai_model(),
        reload_chat_analysis(),
        reload_emotion_detection()
    ])
    if success:
        logging.info("All components reloaded successfully")
    return success

# Global settings instance
ai_settings = AISettings()

def update_response_style(style):
    """Update AI response style."""
    styles = {
        "1": "Professional",
        "2": "Casual",
        "3": "Technical",
        "4": "Friendly"
    }
    if style in styles:
        ai_settings.response_style = styles[style]
        ai_settings.save_settings()
        logging.info(f"Response style updated to: {styles[style]}")
        return True
    return False

def update_emotion_sensitivity(level):
    """Update emotion detection sensitivity."""
    try:
        level = int(level)
        if 1 <= level <= 10:
            ai_settings.emotion_sensitivity = level
            ai_settings.save_settings()
            logging.info(f"Emotion sensitivity updated to: {level}")
            return True
    except ValueError:
        pass
    return False

def update_context_window(size):
    """Update conversation context window size."""
    try:
        size = int(size)
        if 1 <= size <= 20:
            ai_settings.context_window = size
            ai_settings.save_settings()
            logging.info(f"Context window updated to: {size}")
            return True
    except ValueError:
        pass
    return False

def display_current_settings():
    """Display current AI settings."""
    settings = {
        "Response Style": ai_settings.response_style,
        "Emotion Sensitivity": ai_settings.emotion_sensitivity,
        "Context Window": ai_settings.context_window,
        "Language Model": ai_settings.model_name
    }
    print("\nCurrent AI Settings:")
    print("===================")
    for key, value in settings.items():
        print(f"{key}: {value}")
