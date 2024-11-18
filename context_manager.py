from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional, Any

class ConversationContext:
    def __init__(self, expiry_minutes: int = 30, max_context_length: int = 10):
        self.contexts: Dict[str, Dict[str, Any]] = {}
        self.expiry_minutes = expiry_minutes
        self.max_context_length = max_context_length
        self.logger = logging.getLogger(__name__)
    
    def add_to_context(self, user_id: str, message: str, response: str, analysis: dict) -> None:
        """Add a conversation turn to the context"""
        try:
            # Initialize context if it doesn't exist
            if user_id not in self.contexts:
                self.contexts[user_id] = {
                    'history': [],
                    'last_active': datetime.now(),
                    'metadata': {
                        'turn_count': 0,
                        'total_tokens': 0,
                        'topics': set()
                    }
                }
            
            # Update context
            context = self.contexts[user_id]
            context['last_active'] = datetime.now()
            
            # Add new turn
            turn = {
                'timestamp': datetime.now().isoformat(),
                'content': message,
                'role': 'user'
            }
            context['history'].append(turn)
            
            # Add response
            response_turn = {
                'timestamp': datetime.now().isoformat(),
                'content': response,
                'role': 'assistant',
                'analysis': analysis
            }
            context['history'].append(response_turn)
            
            # Update metadata
            context['metadata']['turn_count'] += 1
            context['metadata']['total_tokens'] += len(message.split()) + len(response.split())
            if 'topics' in analysis:
                context['metadata']['topics'].update(analysis['topics'])
            
            # Trim context if too long
            if len(context['history']) > self.max_context_length * 2:  # *2 because each turn has user + assistant
                context['history'] = context['history'][-self.max_context_length * 2:]
            
            self.logger.debug(f"Added context for user {user_id}. Context size: {len(context['history'])}")
            
        except Exception as e:
            self.logger.error(f"Error adding to context: {str(e)}")
            raise
    
    def get_context(self, user_id: str) -> List[Dict[str, Any]]:
        """Get conversation context for a user"""
        try:
            # Clean expired contexts first
            self._cleanup_expired_contexts()
            
            # Return empty list if no context exists
            if user_id not in self.contexts:
                return []
            
            context = self.contexts[user_id]
            
            # Check if context has expired
            if self._is_context_expired(context['last_active']):
                self.clear_context(user_id)
                return []
            
            # Update last active time
            context['last_active'] = datetime.now()
            
            return context['history']
            
        except Exception as e:
            self.logger.error(f"Error getting context: {str(e)}")
            return []
    
    def clear_context(self, user_id: str) -> None:
        """Clear context for a user"""
        try:
            if user_id in self.contexts:
                del self.contexts[user_id]
                self.logger.info(f"Cleared context for user {user_id}")
        except Exception as e:
            self.logger.error(f"Error clearing context: {str(e)}")
            raise
    
    def _is_context_expired(self, last_active: datetime) -> bool:
        """Check if context has expired"""
        return datetime.now() - last_active > timedelta(minutes=self.expiry_minutes)
    
    def _cleanup_expired_contexts(self) -> None:
        """Remove expired contexts"""
        try:
            expired_users = [
                user_id for user_id, context in self.contexts.items()
                if self._is_context_expired(context['last_active'])
            ]
            
            for user_id in expired_users:
                self.clear_context(user_id)
            
            if expired_users:
                self.logger.info(f"Cleaned up {len(expired_users)} expired contexts")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up contexts: {str(e)}")
    
    def get_context_summary(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of the context"""
        try:
            if user_id not in self.contexts:
                return None
            
            context = self.contexts[user_id]
            
            return {
                'turn_count': context['metadata']['turn_count'],
                'total_tokens': context['metadata']['total_tokens'],
                'topics': list(context['metadata']['topics']),
                'context_age_minutes': (datetime.now() - context['last_active']).seconds / 60,
                'turns_in_context': len(context['history']),
                'expiry_in_minutes': self.expiry_minutes - 
                    (datetime.now() - context['last_active']).seconds / 60
            }
            
        except Exception as e:
            self.logger.error(f"Error getting context summary: {str(e)}")
            return None
    
    def update_expiry(self, user_id: str, additional_minutes: int = 30) -> bool:
        """Extend context expiry time"""
        try:
            if user_id not in self.contexts:
                return False
            
            self.contexts[user_id]['last_active'] = datetime.now()
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating expiry: {str(e)}")
            return False

class ErrorHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def handle_error(self, error_type: str, error_message: str) -> Dict[str, Any]:
        """Handle errors and generate appropriate responses"""
        try:
            # Log the error
            self.logger.error(f"Error type: {error_type}, Message: {error_message}")
            
            # Define user-friendly error messages
            error_responses = {
                'RateLimitError': {
                    'message': "You're sending too many requests. Please wait a moment.",
                    'code': 429
                },
                'AuthenticationError': {
                    'message': "There was a problem with authentication. Please try again.",
                    'code': 401
                },
                'ValidationError': {
                    'message': "There was a problem with your input. Please check and try again.",
                    'code': 400
                },
                'APIError': {
                    'message': "There was a problem with the AI service. Please try again later.",
                    'code': 503
                },
                'DatabaseError': {
                    'message': "There was a problem accessing your conversation history.",
                    'code': 500
                },
                'ContextError': {
                    'message': "There was a problem managing your conversation context.",
                    'code': 500
                }
            }
            
            # Get appropriate error response or use generic one
            error_info = error_responses.get(error_type, {
                'message': "An unexpected error occurred. Please try again.",
                'code': 500
            })
            
            return {
                'error': True,
                'error_type': error_type,
                'message': error_info['message'],
                'code': error_info['code'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in error handler: {str(e)}")
            return {
                'error': True,
                'message': "An unexpected error occurred. Please try again.",
                'code': 500,
                'timestamp': datetime.now().isoformat()
            }
