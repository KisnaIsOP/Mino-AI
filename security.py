from functools import wraps
from flask import request, jsonify, session
from datetime import datetime, timedelta
import jwt
import os
from oauthlib.oauth2 import WebApplicationClient
import requests
from cryptography.fernet import Fernet
import secrets
import re
import logging
import time
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import bleach
import html
from dataclasses import dataclass
from threading import Lock
import sqlite3
import hashlib
import hmac

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    filename='error.log',
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class RateLimitRule:
    """Rate limit rule configuration"""
    requests: int
    window: int  # seconds
    block_duration: int = 300  # 5 minutes block by default

class Security:
    def __init__(self, app):
        self.app = app
        # Generate a secure secret key for JWT
        self.jwt_secret = os.getenv('JWT_SECRET_KEY', secrets.token_hex(32))
        # Initialize encryption key for sensitive data
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # OAuth 2.0 client configuration
        self.oauth_client = WebApplicationClient(
            os.getenv('GOOGLE_CLIENT_ID')
        )
        
        # Rate limiting configuration
        self.rate_limits = {}
        
        # Initialize security manager
        self.security_manager = SecurityManager()
        
    def generate_token(self, user_id):
        """Generate JWT token for authenticated users"""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(days=1),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
    
    def verify_token(self, token):
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return payload['user_id']
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def encrypt_data(self, data):
        """Encrypt sensitive data"""
        return self.cipher_suite.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data):
        """Decrypt sensitive data"""
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()
    
    def rate_limit(self, requests_per_minute=60):
        """Rate limiting decorator"""
        def decorator(f):
            @wraps(f)
            def wrapper(*args, **kwargs):
                # Get client IP
                client_ip = request.remote_addr
                current_time = datetime.now()
                
                # Initialize or update rate limit data
                if client_ip not in self.rate_limits:
                    self.rate_limits[client_ip] = {
                        'requests': 0,
                        'reset_time': current_time + timedelta(minutes=1)
                    }
                
                # Reset counter if time window has passed
                if current_time >= self.rate_limits[client_ip]['reset_time']:
                    self.rate_limits[client_ip] = {
                        'requests': 0,
                        'reset_time': current_time + timedelta(minutes=1)
                    }
                
                # Check rate limit
                if self.rate_limits[client_ip]['requests'] >= requests_per_minute:
                    return jsonify({
                        'error': 'Rate limit exceeded',
                        'retry_after': (self.rate_limits[client_ip]['reset_time'] - current_time).seconds
                    }), 429
                
                # Increment request counter
                self.rate_limits[client_ip]['requests'] += 1
                
                return f(*args, **kwargs)
            return wrapper
        return decorator
    
    def require_auth(self, f):
        """Authentication decorator"""
        @wraps(f)
        def decorated(*args, **kwargs):
            token = request.headers.get('Authorization')
            
            if not token:
                return jsonify({'error': 'No token provided'}), 401
            
            # Remove 'Bearer ' prefix if present
            if token.startswith('Bearer '):
                token = token[7:]
            
            user_id = self.verify_token(token)
            if not user_id:
                return jsonify({'error': 'Invalid or expired token'}), 401
            
            return f(*args, **kwargs)
        return decorated
    
    def oauth_login(self):
        """Initiate OAuth 2.0 login flow"""
        google_provider_cfg = requests.get(
            "https://accounts.google.com/.well-known/openid-configuration"
        ).json()
        
        authorization_endpoint = google_provider_cfg["authorization_endpoint"]
        
        request_uri = self.oauth_client.prepare_request_uri(
            authorization_endpoint,
            redirect_uri=request.base_url + "/callback",
            scope=["openid", "email", "profile"],
        )
        
        return request_uri
    
    def validate_ip(self, ip_address):
        """Validate IP address against blocklist"""
        # Implementation of IP validation logic
        # Could include checking against known bad IP lists
        return True
    
    def sanitize_input(self, data):
        """Sanitize user input"""
        # Basic input sanitization
        if isinstance(data, str):
            # HTML escape and strip dangerous tags/attributes
            cleaned = bleach.clean(
                data,
                tags=[],  # No HTML tags allowed
                attributes={},
                protocols=['http', 'https'],
                strip=True
            )
            # Additional sanitization
            cleaned = html.escape(cleaned)
            # Remove potential SQL injection patterns
            cleaned = re.sub(r'[\'";\-\-]', '', cleaned)
            return cleaned
        elif isinstance(data, dict):
            return {k: self.sanitize_input(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_input(item) for item in data]
        return data
    
    def security_manager(self):
        return self.security_manager
    
class SecurityManager:
    def __init__(self):
        self.rate_limits: Dict[str, List[float]] = {}
        self.blocked_ips: Dict[str, float] = {}
        self.rate_limit_lock = Lock()
        
        # Default rate limit rules
        self.rules = {
            'default': RateLimitRule(requests=60, window=60),  # 60 requests per minute
            'auth': RateLimitRule(requests=5, window=60),      # 5 login attempts per minute
            'api': RateLimitRule(requests=30, window=60)       # 30 API requests per minute
        }
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Initialize security database"""
        try:
            conn = sqlite3.connect('security.db')
            cursor = conn.cursor()
            
            # Create security events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS security_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    event_type TEXT NOT NULL,
                    ip_address TEXT,
                    user_id TEXT,
                    details TEXT,
                    severity TEXT
                )
            ''')
            
            # Create blocked IPs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS blocked_ips (
                    ip_address TEXT PRIMARY KEY,
                    blocked_until DATETIME,
                    reason TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Database initialization error: {str(e)}")
            raise
    
    def sanitize_input(self, data: Any) -> Any:
        """
        Sanitize user input to prevent XSS and injection attacks
        """
        if isinstance(data, str):
            # HTML escape and strip dangerous tags/attributes
            cleaned = bleach.clean(
                data,
                tags=[],  # No HTML tags allowed
                attributes={},
                protocols=['http', 'https'],
                strip=True
            )
            # Additional sanitization
            cleaned = html.escape(cleaned)
            # Remove potential SQL injection patterns
            cleaned = re.sub(r'[\'";\-\-]', '', cleaned)
            return cleaned
        elif isinstance(data, dict):
            return {k: self.sanitize_input(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_input(item) for item in data]
        return data
    
    def validate_api_key(self, api_key: str) -> bool:
        """
        Validate API key against environment variable
        """
        try:
            valid_key = os.getenv('GEMINI_API_KEY')
            if not valid_key:
                logger.error("API key not found in environment variables")
                return False
            
            # Use constant-time comparison to prevent timing attacks
            return hmac.compare_digest(api_key, valid_key)
            
        except Exception as e:
            logger.error(f"API key validation error: {str(e)}")
            return False
    
    def check_rate_limit(self, ip: str, rule_name: str = 'default') -> bool:
        """
        Check if request is within rate limits
        """
        with self.rate_limit_lock:
            try:
                # Check if IP is blocked
                if ip in self.blocked_ips:
                    block_end = self.blocked_ips[ip]
                    if time.time() < block_end:
                        return False
                    else:
                        del self.blocked_ips[ip]
                
                rule = self.rules.get(rule_name, self.rules['default'])
                now = time.time()
                
                # Initialize or clean old requests
                if ip not in self.rate_limits:
                    self.rate_limits[ip] = []
                
                # Remove requests outside the window
                self.rate_limits[ip] = [
                    timestamp for timestamp in self.rate_limits[ip]
                    if now - timestamp <= rule.window
                ]
                
                # Check if limit exceeded
                if len(self.rate_limits[ip]) >= rule.requests:
                    # Block the IP
                    self.blocked_ips[ip] = now + rule.block_duration
                    self._log_security_event(
                        'rate_limit_exceeded',
                        ip,
                        severity='warning',
                        details=f'Rate limit exceeded for {rule_name}'
                    )
                    return False
                
                # Add new request
                self.rate_limits[ip].append(now)
                return True
                
            except Exception as e:
                logger.error(f"Rate limit check error: {str(e)}")
                return False
    
    def _log_security_event(
        self,
        event_type: str,
        ip_address: Optional[str] = None,
        user_id: Optional[str] = None,
        details: Optional[str] = None,
        severity: str = 'info'
    ):
        """
        Log security events to database
        """
        try:
            conn = sqlite3.connect('security.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO security_events 
                (event_type, ip_address, user_id, details, severity)
                VALUES (?, ?, ?, ?, ?)
            ''', (event_type, ip_address, user_id, details, severity))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Security event logging error: {str(e)}")
    
    def get_client_ip(self) -> str:
        """
        Get client IP address from request
        """
        if request.headers.get('X-Forwarded-For'):
            return request.headers.get('X-Forwarded-For').split(',')[0]
        return request.remote_addr
    
    def require_api_key(self, f):
        """
        Decorator to require valid API key
        """
        @wraps(f)
        def decorated_function(*args, **kwargs):
            api_key = request.headers.get('X-API-Key')
            if not api_key or not self.validate_api_key(api_key):
                self._log_security_event(
                    'invalid_api_key',
                    self.get_client_ip(),
                    severity='warning'
                )
                return jsonify({'error': 'Invalid API key'}), 401
            return f(*args, **kwargs)
        return decorated_function
    
    def rate_limit(self, rule_name: str = 'default'):
        """
        Decorator to apply rate limiting
        """
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                ip = self.get_client_ip()
                if not self.check_rate_limit(ip, rule_name):
                    return jsonify({
                        'error': 'Rate limit exceeded',
                        'retry_after': self.blocked_ips.get(ip, 0) - time.time()
                    }), 429
                return f(*args, **kwargs)
            return decorated_function
        return decorator
    
    def sanitize_request(self, f):
        """
        Decorator to sanitize request data
        """
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if request.is_json:
                request.json = self.sanitize_input(request.json)
            if request.form:
                request.form = self.sanitize_input(request.form)
            if request.args:
                request.args = self.sanitize_input(request.args)
            return f(*args, **kwargs)
        return decorated_function
    
    def handle_error(self, error: Exception):
        """
        Handle and log errors
        """
        try:
            # Log error with full traceback
            logger.exception(f"Application error: {str(error)}")
            
            # Log security event
            self._log_security_event(
                'application_error',
                self.get_client_ip(),
                details=str(error),
                severity='error'
            )
            
            # Return safe error response
            return jsonify({
                'error': 'An internal error occurred',
                'error_id': hashlib.md5(str(time.time()).encode()).hexdigest()
            }), 500
            
        except Exception as e:
            logger.critical(f"Error handler failed: {str(e)}")
            return jsonify({'error': 'Critical system error'}), 500
    
    def cleanup_old_data(self):
        """
        Cleanup old security logs and blocked IPs
        """
        try:
            conn = sqlite3.connect('security.db')
            cursor = conn.cursor()
            
            # Remove old security events (keep last 30 days)
            cursor.execute('''
                DELETE FROM security_events
                WHERE timestamp < datetime('now', '-30 days')
            ''')
            
            # Remove expired IP blocks
            cursor.execute('''
                DELETE FROM blocked_ips
                WHERE blocked_until < datetime('now')
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Data cleanup error: {str(e)}")
