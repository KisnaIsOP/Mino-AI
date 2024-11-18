import google.generativeai as genai
import json
import logging
from datetime import datetime, timedelta
import psutil
import os
from typing import Dict, Any, List, Optional
import re
import numpy as np
from dataclasses import dataclass
from collections import deque
import threading
import prometheus_client as prom
from resource_manager import ResourceManager

# Prometheus metrics
RESPONSE_TIME = prom.Histogram('response_time_seconds', 'Response time in seconds')
ERROR_RATE = prom.Counter('error_total', 'Total number of errors')
ANOMALY_SCORE = prom.Gauge('anomaly_score', 'Current anomaly score')

@dataclass
class SystemMetrics:
    cpu_usage: float
    memory_usage: float
    response_time: float
    error_count: int
    timestamp: datetime

class SupervisorAI:
    def __init__(self, model_name: str = 'gemini-pro', window_size: int = 100):
        self.model = genai.GenerativeModel(model_name)
        self.logger = logging.getLogger(__name__)
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.error_patterns = {}
        self.anomaly_threshold = 0.8
        
        # Initialize resource manager
        self.resource_manager = ResourceManager(initial_pool_size=4)
        
        # Performance baselines
        self.baseline_metrics = {
            'cpu_usage': {'mean': 0.0, 'std': 0.0},
            'memory_usage': {'mean': 0.0, 'std': 0.0},
            'response_time': {'mean': 0.0, 'std': 0.0},
            'error_rate': {'mean': 0.0, 'std': 0.0}
        }
        
        # Initialize monitoring thread
        self.monitoring_thread = threading.Thread(target=self._continuous_monitoring)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

    async def process_request(self, task_id: str, request_func, *args, priority: int = 1, **kwargs):
        """Process request with dynamic resource allocation."""
        try:
            # Submit task to resource manager
            self.resource_manager.submit_task(
                priority=priority,
                task_id=task_id,
                func=request_func,
                *args,
                **kwargs
            )
            
            # Monitor task execution
            start_time = time.time()
            
            # Update metrics
            RESPONSE_TIME._sum.inc(time.time() - start_time)
            
            return {"status": "submitted", "task_id": task_id}
            
        except Exception as e:
            self.logger.error(f"Error processing request: {str(e)}")
            ERROR_RATE.inc()
            raise

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            # Get resource status
            resource_status = self.resource_manager.get_resource_status()
            
            # Get health metrics
            health_status = self.get_system_health()
            
            return {
                "resource_status": resource_status,
                "health_status": health_status,
                "anomaly_score": float(ANOMALY_SCORE._value.get()),
                "error_rate": float(ERROR_RATE._value.get()),
                "average_response_time": float(RESPONSE_TIME._sum.get() / max(1, RESPONSE_TIME._count.get()))
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {str(e)}")
            return {"status": "error", "message": str(e)}

    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system performance metrics."""
        try:
            return SystemMetrics(
                cpu_usage=psutil.cpu_percent(),
                memory_usage=psutil.virtual_memory().percent,
                response_time=RESPONSE_TIME._sum.get(),
                error_count=int(ERROR_RATE._value.get()),
                timestamp=datetime.now()
            )
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {str(e)}")
            return None

    def _update_baselines(self):
        """Update performance baselines using recent history."""
        if len(self.metrics_history) < 10:
            return

        metrics_array = np.array([
            [m.cpu_usage, m.memory_usage, m.response_time, m.error_count]
            for m in self.metrics_history
        ])

        metrics_names = ['cpu_usage', 'memory_usage', 'response_time', 'error_rate']
        for i, name in enumerate(metrics_names):
            self.baseline_metrics[name]['mean'] = np.mean(metrics_array[:, i])
            self.baseline_metrics[name]['std'] = np.std(metrics_array[:, i])

    def _calculate_anomaly_score(self, metrics: SystemMetrics) -> float:
        """Calculate normalized anomaly score based on current metrics."""
        if not self.baseline_metrics['cpu_usage']['std']:
            return 0.0

        try:
            scores = []
            
            # CPU usage deviation
            cpu_score = abs(metrics.cpu_usage - self.baseline_metrics['cpu_usage']['mean'])
            cpu_score /= max(self.baseline_metrics['cpu_usage']['std'], 1)
            scores.append(cpu_score)
            
            # Memory usage deviation
            mem_score = abs(metrics.memory_usage - self.baseline_metrics['memory_usage']['mean'])
            mem_score /= max(self.baseline_metrics['memory_usage']['std'], 1)
            scores.append(mem_score)
            
            # Response time deviation
            resp_score = abs(metrics.response_time - self.baseline_metrics['response_time']['mean'])
            resp_score /= max(self.baseline_metrics['response_time']['std'], 1)
            scores.append(resp_score)
            
            # Normalize final score
            final_score = np.mean(scores)
            return min(final_score, 1.0)
        except Exception as e:
            self.logger.error(f"Error calculating anomaly score: {str(e)}")
            return 0.0

    def _predict_potential_issues(self, metrics: SystemMetrics) -> List[Dict]:
        """Predict potential issues based on current metrics and trends."""
        issues = []
        
        try:
            # Check for high resource usage
            if metrics.cpu_usage > 80:
                issues.append({
                    'type': 'resource_warning',
                    'severity': 'high',
                    'message': 'High CPU usage detected',
                    'recommendation': 'Consider scaling resources or optimizing processing'
                })

            if metrics.memory_usage > 85:
                issues.append({
                    'type': 'resource_warning',
                    'severity': 'high',
                    'message': 'High memory usage detected',
                    'recommendation': 'Check for memory leaks or increase available memory'
                })

            # Check for response time degradation
            if len(self.metrics_history) >= 2:
                prev_metrics = self.metrics_history[-1]
                if metrics.response_time > prev_metrics.response_time * 1.5:
                    issues.append({
                        'type': 'performance_warning',
                        'severity': 'medium',
                        'message': 'Response time degradation detected',
                        'recommendation': 'Monitor system load and optimize request handling'
                    })

            # Check error rate trend
            recent_errors = [m.error_count for m in list(self.metrics_history)[-5:]]
            if len(recent_errors) >= 5 and np.mean(recent_errors) > 0:
                issues.append({
                    'type': 'error_warning',
                    'severity': 'high',
                    'message': 'Increasing error rate detected',
                    'recommendation': 'Review error logs and implement additional error handling'
                })

            return issues
        except Exception as e:
            self.logger.error(f"Error predicting issues: {str(e)}")
            return []

    def _continuous_monitoring(self):
        """Continuous monitoring loop running in separate thread."""
        while True:
            try:
                metrics = self._collect_system_metrics()
                if metrics:
                    self.metrics_history.append(metrics)
                    self._update_baselines()
                    
                    # Calculate and update anomaly score
                    anomaly_score = self._calculate_anomaly_score(metrics)
                    ANOMALY_SCORE.set(anomaly_score)
                    
                    # Check for potential issues
                    if anomaly_score > self.anomaly_threshold:
                        issues = self._predict_potential_issues(metrics)
                        for issue in issues:
                            self.logger.warning(
                                f"Potential issue detected: {issue['message']} - "
                                f"Recommendation: {issue['recommendation']}"
                            )
                
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(10)  # Back off on error

    def get_system_health(self) -> Dict:
        """Get current system health status and predictions."""
        try:
            current_metrics = self._collect_system_metrics()
            if not current_metrics:
                return {'status': 'error', 'message': 'Unable to collect metrics'}

            anomaly_score = self._calculate_anomaly_score(current_metrics)
            potential_issues = self._predict_potential_issues(current_metrics)

            return {
                'status': 'healthy' if anomaly_score < self.anomaly_threshold else 'warning',
                'metrics': {
                    'cpu_usage': current_metrics.cpu_usage,
                    'memory_usage': current_metrics.memory_usage,
                    'response_time': current_metrics.response_time,
                    'error_count': current_metrics.error_count
                },
                'anomaly_score': anomaly_score,
                'potential_issues': potential_issues,
                'recommendations': [issue['recommendation'] for issue in potential_issues]
            }
        except Exception as e:
            self.logger.error(f"Error getting system health: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def handle_error(self, error: Exception, context: Dict = None) -> Dict[str, Any]:
        """Enhanced error handling with silent recovery."""
        try:
            error_type = self._identify_error_type(str(error))
            component = self._determine_component(error)
            
            # Log error but don't expose to user
            self.logger.error(f"Component {component} error: {str(error)}")
            ERROR_RATE.inc()
            
            # Update error patterns
            if error_type not in self.error_patterns:
                self.error_patterns[error_type] = {
                    'count': 0,
                    'last_seen': None,
                    'recovery_success': 0,
                    'recovery_failure': 0
                }
            
            self.error_patterns[error_type]['count'] += 1
            self.error_patterns[error_type]['last_seen'] = datetime.now()
            
            # Apply recovery strategy
            recovery_success = self._apply_recovery_strategy(error_type, component, context)
            
            # Update recovery statistics
            if recovery_success:
                self.error_patterns[error_type]['recovery_success'] += 1
            else:
                self.error_patterns[error_type]['recovery_failure'] += 1
            
            # Return success response to maintain user experience
            return {
                'status': 'success',
                'message': None,
                'retry': False
            }
            
        except Exception as e:
            self.logger.error(f"Error in error handler: {str(e)}")
            return {
                'status': 'success',
                'message': None,
                'retry': False
            }

    def _identify_error_type(self, error_message: str) -> str:
        """Identify error type from error message."""
        error_patterns = {
            'token_limit': ['maximum context length', 'token limit exceeded'],
            'rate_limit': ['rate limit exceeded', 'too many requests'],
            'model_error': ['model not responding', 'model error'],
            'analysis_error': ['analysis failed', 'could not process emotion'],
            'context_error': ['context not found', 'invalid context']
        }
        
        for error_type, patterns in error_patterns.items():
            if any(pattern in error_message.lower() for pattern in patterns):
                return error_type
        return 'unknown'

    def _determine_component(self, error: Exception) -> str:
        """Determine which component caused the error."""
        error_str = str(error).lower()
        if any(keyword in error_str for keyword in ['analysis', 'emotion', 'sentiment']):
            return 'analyzer'
        return 'primary'

    def _apply_recovery_strategy(self, error_type: str, component: str, context: Dict) -> bool:
        """Apply progressive recovery strategies."""
        try:
            # Get appropriate recovery strategy
            if error_type == 'token_limit':
                return self._handle_token_limit(context)
            elif error_type == 'rate_limit':
                return self._handle_rate_limit(context)
            elif error_type == 'model_error':
                return self._handle_model_error(context)
            elif error_type == 'analysis_error':
                return self._handle_analysis_error(context)
            elif error_type == 'context_error':
                return self._handle_context_error(context)
            else:
                return self._handle_unknown_error(context)
                
        except Exception as e:
            self.logger.error(f"Recovery strategy failed: {str(e)}")
            return False

    def _handle_token_limit(self, context: Dict) -> bool:
        """Handle token limit exceeded errors."""
        try:
            context['max_tokens'] = min(context.get('max_tokens', 2048), 1024)
            context['truncate_input'] = True
            return True
        except Exception:
            return False

    def _handle_rate_limit(self, context: Dict) -> bool:
        """Handle rate limit errors."""
        try:
            time.sleep(2)
            context['rate_limit_pause'] = True
            return True
        except Exception:
            return False

    def _handle_model_error(self, context: Dict) -> bool:
        """Handle model-related errors."""
        try:
            context['use_fallback_model'] = True
            context['simplified_mode'] = True
            return True
        except Exception:
            return False

    def _handle_analysis_error(self, context: Dict) -> bool:
        """Handle emotional analysis errors."""
        try:
            context['skip_analysis'] = True
            context['use_default_emotion'] = True
            return True
        except Exception:
            return False

    def _handle_context_error(self, context: Dict) -> bool:
        """Handle context management errors."""
        try:
            context['reset_context'] = True
            context['minimal_context'] = True
            return True
        except Exception:
            return False

    def _handle_unknown_error(self, context: Dict) -> bool:
        """Handle unknown errors with progressive fallback."""
        try:
            strategies = [
                lambda: self._retry_with_backoff(context),
                lambda: self._reduce_complexity(context),
                lambda: self._switch_to_fallback(context)
            ]
            
            for strategy in strategies:
                if strategy():
                    return True
            return False
            
        except Exception:
            return False

    def _retry_with_backoff(self, context: Dict) -> bool:
        """Retry operation with exponential backoff."""
        try:
            retry_count = context.get('retry_count', 0)
            if retry_count < 3:
                time.sleep(2 ** retry_count)
                context['retry_count'] = retry_count + 1
                return True
            return False
        except Exception:
            return False

    def _reduce_complexity(self, context: Dict) -> bool:
        """Reduce operation complexity."""
        try:
            context['temperature'] = max(0.1, context.get('temperature', 0.9) - 0.2)
            context['max_tokens'] = min(context.get('max_tokens', 2048), 1024)
            return True
        except Exception:
            return False

    def _switch_to_fallback(self, context: Dict) -> bool:
        """Switch to fallback mode with minimal features."""
        try:
            context['fallback_mode'] = True
            context['minimal_features'] = True
            return True
        except Exception:
            return False

    def supervise_response(self, message: str, response: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate response quality and provide supervision feedback"""
        try:
            # Basic quality checks
            basic_metrics = self._check_basic_metrics(response)
            
            # Content quality analysis
            content_quality = self._analyze_content_quality(message, response)
            
            # Response coherence check
            coherence_score = self._check_coherence(message, response)
            
            # Response relevance to user intent
            relevance_score = self._check_relevance(message, response, analysis)
            
            # Emotional tone analysis
            tone_analysis = self._analyze_tone(response)
            
            # Calculate overall quality score
            quality_score = self._calculate_quality_score(
                basic_metrics,
                content_quality,
                coherence_score,
                relevance_score
            )
            
            # Generate improvement suggestions
            suggestions = self._generate_improvements(
                message,
                response,
                basic_metrics,
                content_quality,
                quality_score
            )
            
            # Prepare supervision result
            supervision_result = {
                'quality_score': quality_score,
                'metrics': {
                    'basic_metrics': basic_metrics,
                    'content_quality': content_quality,
                    'coherence_score': coherence_score,
                    'relevance_score': relevance_score,
                    'tone_analysis': tone_analysis
                },
                'suggestions': suggestions,
                'timestamp': datetime.now().isoformat()
            }
            
            # Log supervision metrics
            self._log_supervision_metrics(supervision_result)
            
            return supervision_result
            
        except Exception as e:
            self.logger.error(f"Error in supervision: {str(e)}")
            return {
                'error': str(e),
                'quality_score': 0.0,
                'timestamp': datetime.now().isoformat()
            }
    
    def _check_basic_metrics(self, response: str) -> Dict[str, Any]:
        """Check basic response metrics"""
        words = response.split()
        sentences = re.split(r'[.!?]+', response)
        
        return {
            'length': len(response),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0
        }
    
    def _analyze_content_quality(self, message: str, response: str) -> Dict[str, float]:
        """Analyze content quality using AI"""
        try:
            analysis_prompt = f"""
            Analyze the quality of this AI response:
            
            User Message: {message}
            AI Response: {response}
            
            Rate the following aspects from 0.0 to 1.0:
            1. Completeness: Does it fully address the user's query?
            2. Clarity: Is it clear and easy to understand?
            3. Accuracy: Does it provide accurate information?
            4. Helpfulness: How helpful is the response?
            5. Conciseness: Is it appropriately concise while being complete?
            
            Format the response as a JSON object with these metrics.
            """
            
            result = self.model.generate_content(analysis_prompt)
            metrics = json.loads(result.text)
            
            return {
                'completeness': float(metrics.get('completeness', 0.0)),
                'clarity': float(metrics.get('clarity', 0.0)),
                'accuracy': float(metrics.get('accuracy', 0.0)),
                'helpfulness': float(metrics.get('helpfulness', 0.0)),
                'conciseness': float(metrics.get('conciseness', 0.0))
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing content quality: {str(e)}")
            return {
                'completeness': 0.0,
                'clarity': 0.0,
                'accuracy': 0.0,
                'helpfulness': 0.0,
                'conciseness': 0.0
            }
    
    def _check_coherence(self, message: str, response: str) -> float:
        """Check response coherence"""
        try:
            coherence_prompt = f"""
            Rate the coherence of this AI response from 0.0 to 1.0:
            
            User Message: {message}
            AI Response: {response}
            
            Consider:
            1. Logical flow
            2. Sentence transitions
            3. Internal consistency
            4. Argument structure
            
            Respond with only a number between 0.0 and 1.0.
            """
            
            result = self.model.generate_content(coherence_prompt)
            return float(result.text.strip())
            
        except Exception as e:
            self.logger.error(f"Error checking coherence: {str(e)}")
            return 0.0
    
    def _check_relevance(self, message: str, response: str, analysis: Dict[str, Any]) -> float:
        """Check response relevance to user intent"""
        try:
            relevance_prompt = f"""
            Rate how relevant this AI response is to the user's intent from 0.0 to 1.0:
            
            User Message: {message}
            User Intent: {json.dumps(analysis.get('intent', {}))}
            AI Response: {response}
            
            Consider:
            1. Intent alignment
            2. Topic relevance
            3. Context appropriateness
            4. Information value
            
            Respond with only a number between 0.0 and 1.0.
            """
            
            result = self.model.generate_content(relevance_prompt)
            return float(result.text.strip())
            
        except Exception as e:
            self.logger.error(f"Error checking relevance: {str(e)}")
            return 0.0
    
    def _analyze_tone(self, response: str) -> Dict[str, float]:
        """Analyze emotional tone of response"""
        try:
            tone_prompt = f"""
            Analyze the emotional tone of this AI response:
            
            Response: {response}
            
            Rate these aspects from 0.0 to 1.0:
            1. Professionalism
            2. Friendliness
            3. Empathy
            4. Confidence
            5. Respectfulness
            
            Format the response as a JSON object with these metrics.
            """
            
            result = self.model.generate_content(tone_prompt)
            metrics = json.loads(result.text)
            
            return {
                'professionalism': float(metrics.get('professionalism', 0.0)),
                'friendliness': float(metrics.get('friendliness', 0.0)),
                'empathy': float(metrics.get('empathy', 0.0)),
                'confidence': float(metrics.get('confidence', 0.0)),
                'respectfulness': float(metrics.get('respectfulness', 0.0))
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing tone: {str(e)}")
            return {
                'professionalism': 0.0,
                'friendliness': 0.0,
                'empathy': 0.0,
                'confidence': 0.0,
                'respectfulness': 0.0
            }
    
    def _calculate_quality_score(
        self,
        basic_metrics: Dict[str, Any],
        content_quality: Dict[str, float],
        coherence_score: float,
        relevance_score: float
    ) -> float:
        """Calculate overall quality score"""
        try:
            # Basic metrics score
            basic_score = 1.0
            if basic_metrics['length'] < self.quality_thresholds['min_length']:
                basic_score *= 0.5
            if basic_metrics['length'] > self.quality_thresholds['max_length']:
                basic_score *= 0.7
            if basic_metrics['sentence_count'] < self.quality_thresholds['min_sentences']:
                basic_score *= 0.8
            
            # Content quality score (average of all metrics)
            content_score = sum(content_quality.values()) / len(content_quality)
            
            # Weighted average of all scores
            weights = {
                'basic': 0.2,
                'content': 0.4,
                'coherence': 0.2,
                'relevance': 0.2
            }
            
            final_score = (
                weights['basic'] * basic_score +
                weights['content'] * content_score +
                weights['coherence'] * coherence_score +
                weights['relevance'] * relevance_score
            )
            
            return round(final_score, 2)
            
        except Exception as e:
            self.logger.error(f"Error calculating quality score: {str(e)}")
            return 0.0
    
    def _generate_improvements(
        self,
        message: str,
        response: str,
        basic_metrics: Dict[str, Any],
        content_quality: Dict[str, float],
        quality_score: float
    ) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []
        
        # Length-based suggestions
        if basic_metrics['length'] < self.quality_thresholds['min_length']:
            suggestions.append("Response is too brief. Consider providing more details.")
        elif basic_metrics['length'] > self.quality_thresholds['max_length']:
            suggestions.append("Response is too long. Consider being more concise.")
        
        # Content quality suggestions
        if content_quality['completeness'] < 0.7:
            suggestions.append("Response could be more complete. Address all aspects of the query.")
        if content_quality['clarity'] < 0.7:
            suggestions.append("Response could be clearer. Use simpler language or better explanations.")
        if content_quality['accuracy'] < 0.8:
            suggestions.append("Verify the accuracy of the information provided.")
        
        # General quality suggestions
        if quality_score < 0.6:
            suggestions.append("Overall response quality needs improvement.")
        
        return suggestions
    
    def _log_supervision_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log supervision metrics for trending analysis"""
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 1000:  # Keep last 1000 entries
            self.metrics_history = self.metrics_history[-1000:]
    
    def get_quality_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get quality score trends"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_metrics = [
                m for m in self.metrics_history
                if datetime.fromisoformat(m['timestamp']) > cutoff_time
            ]
            
            if not recent_metrics:
                return {'error': 'No metrics available for the specified time period'}
            
            scores = [m['quality_score'] for m in recent_metrics]
            return {
                'average_score': sum(scores) / len(scores),
                'min_score': min(scores),
                'max_score': max(scores),
                'total_responses': len(scores),
                'score_distribution': {
                    'excellent': len([s for s in scores if s >= 0.8]),
                    'good': len([s for s in scores if 0.6 <= s < 0.8]),
                    'fair': len([s for s in scores if 0.4 <= s < 0.6]),
                    'poor': len([s for s in scores if s < 0.4])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting quality trends: {str(e)}")
            return {'error': str(e)}
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get system performance summary"""
        try:
            process = psutil.Process(os.getpid())
            
            return {
                'quality_metrics': self.get_quality_trends(),
                'system_metrics': {
                    'cpu_percent': process.cpu_percent(),
                    'memory_usage': process.memory_info().rss / 1024 / 1024,  # MB
                    'thread_count': process.num_threads(),
                    'total_responses': len(self.metrics_history)
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system summary: {str(e)}")
            return {'error': str(e)}
