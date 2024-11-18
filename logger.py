import logging
from logging.handlers import RotatingFileHandler
import os
import psutil
import time

class Logger:
    def __init__(self):
        self.logger = logging.getLogger('MINO_AI')
        self.logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            'logs/mino_ai.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        
        # Format for logs
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        
        # Performance monitoring
        self.process = psutil.Process(os.getpid())
    
    def log_request(self, user_id, message):
        self.logger.info(f'Request from user {user_id}: {message}')
    
    def log_response(self, user_id, response_time):
        self.logger.info(f'Response to user {user_id} took {response_time:.2f}s')
    
    def log_error(self, error_type, error_message):
        self.logger.error(f'{error_type}: {error_message}')
    
    def get_system_metrics(self):
        return {
            'cpu_percent': self.process.cpu_percent(),
            'memory_usage': self.process.memory_info().rss / 1024 / 1024,  # MB
            'threads': self.process.num_threads(),
            'open_files': len(self.process.open_files())
        }
    
    def log_metrics(self, db):
        metrics = self.get_system_metrics()
        db.log_system_metrics(
            response_time=time.time(),
            api_calls=0,  # This would be tracked in the app
            error_count=0,  # This would be tracked in the app
            memory_usage=metrics['memory_usage']
        )
        self.logger.info(f'System Metrics: {metrics}')
