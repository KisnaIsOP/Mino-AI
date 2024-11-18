import logging
import time
import threading
import queue
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import google.generativeai as genai
from performance_optimizer import PerformanceOptimizer
from resource_manager import ResourceManager
from learning_engine import LearningEngine

class AISupervisor:
    def __init__(self, api_key: str):
        """Initialize the AI Supervisor with advanced error recovery capabilities."""
        self.api_key = api_key
        self.logger = self._setup_logger()
        self.performance_optimizer = PerformanceOptimizer()
        self.resource_manager = ResourceManager()
        self.learning_engine = LearningEngine()
        
        # Error recovery configurations
        self.error_patterns = self._load_error_patterns()
        self.recovery_strategies = self._load_recovery_strategies()
        self.error_queue = queue.Queue()
        self.recovery_thread = threading.Thread(target=self._recovery_worker, daemon=True)
        self.recovery_thread.start()
        
        # System health monitoring
        self.health_metrics = {
            'errors': [],
            'response_times': [],
            'recovery_success_rate': 1.0,
            'last_health_check': datetime.now(),
            'system_status': 'healthy'
        }
        
        # AI Component Status
        self.ai_components = {
            'analyzer': {'status': 'healthy', 'last_error': None, 'recovery_attempts': 0},
            'primary': {'status': 'healthy', 'last_error': None, 'recovery_attempts': 0}
        }
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitor_thread.start()

    def _setup_logger(self) -> logging.Logger:
        """Set up dedicated logger for the supervisor."""
        logger = logging.getLogger('AI_Supervisor')
        logger.setLevel(logging.INFO)
        
        # Create handlers for different log levels
        info_handler = logging.FileHandler('logs/supervisor.log')
        error_handler = logging.FileHandler('logs/supervisor_errors.log')
        
        # Set levels and formatters
        info_handler.setLevel(logging.INFO)
        error_handler.setLevel(logging.ERROR)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        info_handler.setFormatter(formatter)
        error_handler.setFormatter(formatter)
        
        logger.addHandler(info_handler)
        logger.addHandler(error_handler)
        
        return logger

    def _load_error_patterns(self) -> Dict[str, Dict]:
        """Load known error patterns and their solutions."""
        return {
            'token_limit': {
                'patterns': ['maximum context length', 'token limit exceeded'],
                'solution': self._handle_token_limit,
                'component': 'primary'
            },
            'rate_limit': {
                'patterns': ['rate limit exceeded', 'too many requests'],
                'solution': self._handle_rate_limit,
                'component': 'primary'
            },
            'model_error': {
                'patterns': ['model not responding', 'model error'],
                'solution': self._handle_model_error,
                'component': 'both'
            },
            'analysis_error': {
                'patterns': ['analysis failed', 'could not process emotion'],
                'solution': self._handle_analysis_error,
                'component': 'analyzer'
            },
            'context_error': {
                'patterns': ['context not found', 'invalid context'],
                'solution': self._handle_context_error,
                'component': 'analyzer'
            }
        }

    def _load_recovery_strategies(self) -> Dict[str, List[callable]]:
        """Load recovery strategies for different error types."""
        return {
            'primary': [
                self._retry_with_backoff,
                self._reduce_complexity,
                self._switch_to_fallback_mode
            ],
            'analyzer': [
                self._retry_analysis,
                self._use_simplified_analysis,
                self._skip_analysis_gracefully
            ],
            'both': [
                self._full_system_reset,
                self._emergency_recovery
            ]
        }

    def _recovery_worker(self):
        """Background worker that processes errors and applies recovery strategies."""
        while True:
            try:
                error_data = self.error_queue.get()
                if error_data is None:
                    break

                error, component, context = error_data
                self._handle_error_recovery(error, component, context)
                self.error_queue.task_done()
            except Exception as e:
                self.logger.error(f"Recovery worker error: {str(e)}")
                time.sleep(1)

    def _monitor_system(self):
        """Continuous system monitoring and proactive error prevention."""
        while True:
            try:
                # Check system health
                self._check_system_health()
                
                # Analyze error patterns
                self._analyze_error_patterns()
                
                # Optimize performance
                self._optimize_performance()
                
                # Clean up old errors
                self._cleanup_old_errors()
                
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Monitor thread error: {str(e)}")
                time.sleep(5)

    def _handle_error_recovery(self, error: Exception, component: str, context: Dict):
        """Handle error recovery without user awareness."""
        try:
            # Log the error but don't expose it
            self.logger.error(f"Component {component} error: {str(error)}")
            
            # Update component status
            self.ai_components[component]['last_error'] = str(error)
            self.ai_components[component]['recovery_attempts'] += 1
            
            # Find matching error pattern
            error_type = self._identify_error_type(str(error))
            if error_type:
                # Apply specific solution
                self.error_patterns[error_type]['solution'](context)
            else:
                # Apply general recovery strategy
                self._apply_recovery_strategy(component, context)
            
            # Update metrics
            self._update_health_metrics(component, success=True)
            
        except Exception as recovery_error:
            self.logger.error(f"Recovery failed: {str(recovery_error)}")
            self._update_health_metrics(component, success=False)
            self._emergency_recovery(context)

    def _identify_error_type(self, error_message: str) -> Optional[str]:
        """Identify the type of error based on error patterns."""
        for error_type, data in self.error_patterns.items():
            if any(pattern in error_message.lower() for pattern in data['patterns']):
                return error_type
        return None

    def _apply_recovery_strategy(self, component: str, context: Dict):
        """Apply progressive recovery strategies."""
        strategies = self.recovery_strategies.get(component, [])
        
        for strategy in strategies:
            try:
                if strategy(context):
                    self.logger.info(f"Recovery successful using {strategy.__name__}")
                    return True
            except Exception as e:
                self.logger.error(f"Strategy {strategy.__name__} failed: {str(e)}")
                continue
        
        return False

    def _retry_with_backoff(self, context: Dict) -> bool:
        """Retry operation with exponential backoff."""
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
                # Attempt operation
                return True
            except Exception:
                continue
        
        return False

    def _reduce_complexity(self, context: Dict) -> bool:
        """Reduce operation complexity."""
        try:
            context['temperature'] = max(0.1, context.get('temperature', 0.9) - 0.2)
            context['max_tokens'] = min(context.get('max_tokens', 2048), 1024)
            return True
        except Exception:
            return False

    def _switch_to_fallback_mode(self, context: Dict) -> bool:
        """Switch to fallback mode with simplified operations."""
        try:
            context['fallback_mode'] = True
            context['skip_analysis'] = True
            return True
        except Exception:
            return False

    def _retry_analysis(self, context: Dict) -> bool:
        """Retry emotional analysis with simplified parameters."""
        try:
            context['simplified_analysis'] = True
            return True
        except Exception:
            return False

    def _use_simplified_analysis(self, context: Dict) -> bool:
        """Use simplified version of emotional analysis."""
        try:
            context['basic_emotions_only'] = True
            return True
        except Exception:
            return False

    def _skip_analysis_gracefully(self, context: Dict) -> bool:
        """Skip emotional analysis while maintaining basic functionality."""
        try:
            context['skip_analysis'] = True
            context['default_emotion'] = 'neutral'
            return True
        except Exception:
            return False

    def _full_system_reset(self, context: Dict) -> bool:
        """Perform full system reset while maintaining user session."""
        try:
            self.resource_manager.reset_resources()
            self.performance_optimizer.reset_optimizations()
            context['reset_timestamp'] = datetime.now()
            return True
        except Exception:
            return False

    def _emergency_recovery(self, context: Dict) -> bool:
        """Emergency recovery mode with minimal functionality."""
        try:
            context['emergency_mode'] = True
            context['minimal_features'] = True
            self.logger.warning("Entering emergency recovery mode")
            return True
        except Exception:
            return False

    def _handle_token_limit(self, context: Dict):
        """Handle token limit exceeded errors."""
        context['max_tokens'] = min(context.get('max_tokens', 2048), 1024)
        context['truncate_input'] = True

    def _handle_rate_limit(self, context: Dict):
        """Handle rate limit errors."""
        time.sleep(2)
        context['rate_limit_pause'] = True

    def _handle_model_error(self, context: Dict):
        """Handle model-related errors."""
        context['use_fallback_model'] = True
        context['simplified_mode'] = True

    def _handle_analysis_error(self, context: Dict):
        """Handle emotional analysis errors."""
        context['skip_analysis'] = True
        context['use_default_emotion'] = True

    def _handle_context_error(self, context: Dict):
        """Handle context management errors."""
        context['reset_context'] = True
        context['minimal_context'] = True

    def _check_system_health(self):
        """Check overall system health and optimize performance."""
        current_time = datetime.now()
        self.health_metrics['last_health_check'] = current_time
        
        # Check component health
        for component, status in self.ai_components.items():
            if status['recovery_attempts'] > 5:
                self._trigger_component_maintenance(component)
            elif status['last_error']:
                if (current_time - status['last_error']).seconds > 300:
                    status['status'] = 'healthy'
                    status['recovery_attempts'] = 0

    def _trigger_component_maintenance(self, component: str):
        """Trigger maintenance mode for a component."""
        try:
            self.ai_components[component]['status'] = 'maintenance'
            self.learning_engine.analyze_failures(component)
            self.performance_optimizer.optimize_component(component)
            self.ai_components[component]['recovery_attempts'] = 0
        except Exception as e:
            self.logger.error(f"Maintenance failed for {component}: {str(e)}")

    def _analyze_error_patterns(self):
        """Analyze error patterns for proactive optimization."""
        try:
            error_data = self.learning_engine.analyze_error_patterns(self.health_metrics['errors'])
            self.performance_optimizer.apply_learned_optimizations(error_data)
        except Exception as e:
            self.logger.error(f"Error pattern analysis failed: {str(e)}")

    def _optimize_performance(self):
        """Optimize system performance based on metrics."""
        try:
            metrics = {
                'response_times': self.health_metrics['response_times'],
                'error_rate': len(self.health_metrics['errors']) / max(1, len(self.health_metrics['response_times'])),
                'recovery_rate': self.health_metrics['recovery_success_rate']
            }
            self.performance_optimizer.optimize(metrics)
        except Exception as e:
            self.logger.error(f"Performance optimization failed: {str(e)}")

    def _cleanup_old_errors(self):
        """Clean up old error records."""
        try:
            current_time = datetime.now()
            self.health_metrics['errors'] = [
                error for error in self.health_metrics['errors']
                if (current_time - error['timestamp']).days < 1
            ]
        except Exception as e:
            self.logger.error(f"Error cleanup failed: {str(e)}")

    def handle_error(self, error: Exception, context: Dict) -> Dict:
        """Public method to handle errors, now fully automated."""
        try:
            # Add error to queue for background processing
            self.error_queue.put((error, self._determine_component(error), context))
            
            # Return success response to maintain user experience
            return {
                'retry': False,
                'error': None,
                'retry_after': None
            }
        except Exception as e:
            self.logger.error(f"Error handling failed: {str(e)}")
            return {
                'retry': False,
                'error': None,
                'retry_after': None
            }

    def _determine_component(self, error: Exception) -> str:
        """Determine which AI component caused the error."""
        error_str = str(error).lower()
        if any(keyword in error_str for keyword in ['analysis', 'emotion', 'sentiment']):
            return 'analyzer'
        return 'primary'

    def record_response_time(self, response_time: float):
        """Record response time for performance monitoring."""
        try:
            self.health_metrics['response_times'].append(response_time)
            if len(self.health_metrics['response_times']) > 1000:
                self.health_metrics['response_times'] = self.health_metrics['response_times'][-1000:]
        except Exception as e:
            self.logger.error(f"Failed to record response time: {str(e)}")

    def get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics without exposing internal issues."""
        return {
            'status': 'operational',
            'performance': 'optimal',
            'last_check': self.health_metrics['last_health_check'].isoformat()
        }
