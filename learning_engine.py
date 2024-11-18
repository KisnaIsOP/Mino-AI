import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, deque
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Experience:
    """Experience tuple for reinforcement learning"""
    state: Dict[str, Any]
    action: str
    reward: float
    next_state: Dict[str, Any]
    metadata: Dict[str, Any]

class ReplayBuffer:
    """Experience replay buffer for reinforcement learning"""
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, experience: Experience):
        """Add experience to buffer"""
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch of experiences"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)

class QNetwork(nn.Module):
    """Neural network for Q-learning"""
    def __init__(self, state_size: int, action_size: int):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class AdaptiveLearningEngine:
    def __init__(self, db_path: str = 'learning.db'):
        self.db_path = db_path
        self.replay_buffer = ReplayBuffer()
        self.batch_size = 32
        self.gamma = 0.99  # discount factor
        self.tau = 0.001   # soft update parameter
        
        # Initialize databases
        self._init_db()
        
        # Load or initialize models
        self.response_model = self._init_response_model()
        self.user_preference_model = self._init_preference_model()
        
        # Initialize state scalers
        self.state_scaler = StandardScaler()
        
        # Track performance metrics
        self.performance_metrics = defaultdict(list)
        
        logger.info("Adaptive Learning Engine initialized")
    
    def _init_db(self):
        """Initialize learning database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create learning experiences table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_experiences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    state TEXT,
                    action TEXT,
                    reward FLOAT,
                    next_state TEXT,
                    metadata TEXT
                )
            ''')
            
            # Create user preferences table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    preferences TEXT,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metric_type TEXT,
                    value FLOAT,
                    metadata TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Database initialization error: {str(e)}")
            raise
    
    def _init_response_model(self) -> RandomForestClassifier:
        """Initialize or load response model"""
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    
    def _init_preference_model(self) -> Dict[str, Any]:
        """Initialize user preference model"""
        return {
            'response_length': {'short': 0.3, 'medium': 0.5, 'long': 0.2},
            'tone': {'formal': 0.5, 'casual': 0.3, 'technical': 0.2},
            'detail_level': {'basic': 0.3, 'detailed': 0.5, 'comprehensive': 0.2}
        }
    
    def record_experience(
        self,
        state: Dict[str, Any],
        action: str,
        reward: float,
        next_state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record learning experience"""
        try:
            # Create experience object
            experience = Experience(state, action, reward, next_state, metadata or {})
            
            # Add to replay buffer
            self.replay_buffer.push(experience)
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO learning_experiences
                (state, action, reward, next_state, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                json.dumps(state),
                action,
                reward,
                json.dumps(next_state),
                json.dumps(metadata)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error recording experience: {str(e)}")
    
    def update_user_preferences(
        self,
        user_id: str,
        interaction_data: Dict[str, Any]
    ):
        """Update user preference model"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get existing preferences
            cursor.execute(
                'SELECT preferences FROM user_preferences WHERE user_id = ?',
                (user_id,)
            )
            result = cursor.fetchone()
            
            if result:
                current_prefs = json.loads(result[0])
            else:
                current_prefs = self._init_preference_model()
            
            # Update preferences based on interaction
            updated_prefs = self._update_preferences(
                current_prefs,
                interaction_data
            )
            
            # Store updated preferences
            cursor.execute('''
                INSERT OR REPLACE INTO user_preferences
                (user_id, preferences, last_updated)
                VALUES (?, ?, datetime('now'))
            ''', (
                user_id,
                json.dumps(updated_prefs)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating user preferences: {str(e)}")
    
    def _update_preferences(
        self,
        current_prefs: Dict[str, Any],
        interaction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update preference weights based on interaction"""
        updated_prefs = current_prefs.copy()
        
        # Learning rate for preference updates
        alpha = 0.1
        
        for category, weights in updated_prefs.items():
            if category in interaction_data:
                chosen_option = interaction_data[category]
                # Increase weight for chosen option
                for option in weights:
                    if option == chosen_option:
                        weights[option] = weights[option] * (1 - alpha) + alpha
                    else:
                        weights[option] = weights[option] * (1 - alpha)
                
                # Normalize weights
                total = sum(weights.values())
                for option in weights:
                    weights[option] /= total
        
        return updated_prefs
    
    def train_models(self):
        """Train models using collected experiences"""
        try:
            if len(self.replay_buffer) < self.batch_size:
                return
            
            # Sample experiences
            experiences = self.replay_buffer.sample(self.batch_size)
            
            # Prepare training data
            states = []
            actions = []
            rewards = []
            next_states = []
            
            for exp in experiences:
                states.append(self._flatten_state(exp.state))
                actions.append(exp.action)
                rewards.append(exp.reward)
                next_states.append(self._flatten_state(exp.next_state))
            
            # Convert to numpy arrays
            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            next_states = np.array(next_states)
            
            # Scale states
            states_scaled = self.state_scaler.fit_transform(states)
            next_states_scaled = self.state_scaler.transform(next_states)
            
            # Train response model
            self.response_model.fit(states_scaled, actions)
            
            # Calculate and log performance metrics
            self._log_training_metrics(states_scaled, actions, rewards)
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
    
    def _flatten_state(self, state: Dict[str, Any]) -> List[float]:
        """Convert state dictionary to flat vector"""
        flat_state = []
        
        # Extract numerical features
        for key, value in state.items():
            if isinstance(value, (int, float)):
                flat_state.append(float(value))
            elif isinstance(value, dict):
                flat_state.extend(value.values())
            elif isinstance(value, list):
                flat_state.extend([float(x) for x in value if isinstance(x, (int, float))])
        
        return flat_state
    
    def _log_training_metrics(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray
    ):
        """Log training performance metrics"""
        try:
            # Calculate metrics
            avg_reward = np.mean(rewards)
            action_diversity = len(set(actions)) / len(actions)
            
            metrics = {
                'average_reward': avg_reward,
                'action_diversity': action_diversity
            }
            
            # Store metrics
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for metric_type, value in metrics.items():
                cursor.execute('''
                    INSERT INTO performance_metrics
                    (metric_type, value, metadata)
                    VALUES (?, ?, ?)
                ''', (
                    metric_type,
                    float(value),
                    json.dumps({'batch_size': self.batch_size})
                ))
            
            conn.commit()
            conn.close()
            
            # Update tracking
            for metric_type, value in metrics.items():
                self.performance_metrics[metric_type].append(value)
            
        except Exception as e:
            logger.error(f"Error logging metrics: {str(e)}")
    
    def get_performance_summary(
        self,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get summary of learning performance"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent metrics
            cursor.execute('''
                SELECT metric_type, value
                FROM performance_metrics
                WHERE timestamp > datetime('now', ?)
                ORDER BY timestamp DESC
            ''', (f'-{days} days',))
            
            results = cursor.fetchall()
            conn.close()
            
            # Process metrics
            metrics_by_type = defaultdict(list)
            for metric_type, value in results:
                metrics_by_type[metric_type].append(value)
            
            # Calculate summaries
            summary = {}
            for metric_type, values in metrics_by_type.items():
                summary[metric_type] = {
                    'current': values[0] if values else None,
                    'average': np.mean(values) if values else None,
                    'trend': np.polyfit(range(len(values)), values, 1)[0]
                    if len(values) > 1 else None
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {str(e)}")
            return {}
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up old learning data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Remove old experiences
            cursor.execute('''
                DELETE FROM learning_experiences
                WHERE timestamp < datetime('now', ?)
            ''', (f'-{days} days',))
            
            # Remove old metrics
            cursor.execute('''
                DELETE FROM performance_metrics
                WHERE timestamp < datetime('now', ?)
            ''', (f'-{days} days',))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error cleaning up data: {str(e)}")

# Initialize learning engine
learning_engine = AdaptiveLearningEngine()
