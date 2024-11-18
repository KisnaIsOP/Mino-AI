from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pickle
import os

@dataclass
class InteractionMetrics:
    message_length: int
    response_length: int
    interaction_duration: float
    satisfaction_score: float
    timestamp: datetime

@dataclass
class UserPreferences:
    response_style: str = "balanced"  # concise, detailed, balanced
    technical_level: str = "intermediate"  # beginner, intermediate, advanced
    tone_preference: str = "neutral"  # formal, casual, neutral
    language_complexity: float = 0.5  # 0.0 to 1.0
    detail_level: float = 0.5  # 0.0 to 1.0
    example_preference: bool = True
    code_snippet_preference: bool = True

@dataclass
class UserProfile:
    user_id: str
    preferences: UserPreferences
    interaction_history: List[InteractionMetrics]
    topics_of_interest: Dict[str, float]
    expertise_areas: Dict[str, float]
    learning_style: str
    engagement_patterns: Dict[str, Any]
    last_active: datetime
    personality_traits: Dict[str, float]
    feedback_history: List[Dict[str, Any]]

class UserProfileManager:
    def __init__(self, storage_path: str = "user_profiles/"):
        self.storage_path = storage_path
        self.profiles: Dict[str, UserProfile] = {}
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.interaction_clusters = KMeans(n_clusters=5)
        self.logger = logging.getLogger(__name__)
        
        os.makedirs(storage_path, exist_ok=True)
        self._load_profiles()

    def _load_profiles(self):
        """Load user profiles from storage."""
        try:
            for filename in os.listdir(self.storage_path):
                if filename.endswith('.json'):
                    with open(os.path.join(self.storage_path, filename), 'r') as f:
                        data = json.load(f)
                        user_id = filename[:-5]  # Remove .json
                        self.profiles[user_id] = self._deserialize_profile(data)
        except Exception as e:
            self.logger.error(f"Error loading profiles: {str(e)}")

    def _save_profile(self, profile: UserProfile):
        """Save user profile to storage."""
        try:
            filename = os.path.join(self.storage_path, f"{profile.user_id}.json")
            with open(filename, 'w') as f:
                json.dump(self._serialize_profile(profile), f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving profile: {str(e)}")

    def _serialize_profile(self, profile: UserProfile) -> Dict:
        """Convert UserProfile to JSON-serializable dict."""
        data = asdict(profile)
        data['last_active'] = data['last_active'].isoformat()
        for metric in data['interaction_history']:
            metric['timestamp'] = metric['timestamp'].isoformat()
        return data

    def _deserialize_profile(self, data: Dict) -> UserProfile:
        """Convert JSON data back to UserProfile."""
        data['last_active'] = datetime.fromisoformat(data['last_active'])
        for metric in data['interaction_history']:
            metric['timestamp'] = datetime.fromisoformat(metric['timestamp'])
        return UserProfile(**data)

    def get_or_create_profile(self, user_id: str) -> UserProfile:
        """Get existing profile or create new one."""
        if user_id not in self.profiles:
            self.profiles[user_id] = UserProfile(
                user_id=user_id,
                preferences=UserPreferences(),
                interaction_history=[],
                topics_of_interest={},
                expertise_areas={},
                learning_style="visual",
                engagement_patterns={
                    "peak_activity_hours": [],
                    "session_duration_avg": 0,
                    "interaction_frequency": "medium"
                },
                last_active=datetime.now(),
                personality_traits={
                    "openness": 0.5,
                    "conscientiousness": 0.5,
                    "extraversion": 0.5,
                    "agreeableness": 0.5,
                    "neuroticism": 0.5
                },
                feedback_history=[]
            )
            self._save_profile(self.profiles[user_id])
        return self.profiles[user_id]

    def update_interaction_metrics(self, user_id: str, metrics: InteractionMetrics):
        """Update user profile with new interaction metrics."""
        profile = self.get_or_create_profile(user_id)
        profile.interaction_history.append(metrics)
        profile.last_active = datetime.now()
        
        # Update engagement patterns
        self._update_engagement_patterns(profile)
        self._save_profile(profile)

    def _update_engagement_patterns(self, profile: UserProfile):
        """Update user engagement patterns based on interaction history."""
        if not profile.interaction_history:
            return

        recent_interactions = [
            m for m in profile.interaction_history
            if m.timestamp > datetime.now() - timedelta(days=30)
        ]
        
        if recent_interactions:
            # Calculate peak activity hours
            hours = [m.timestamp.hour for m in recent_interactions]
            peak_hours = [h for h in range(24) if hours.count(h) > len(hours)/24]
            profile.engagement_patterns["peak_activity_hours"] = peak_hours
            
            # Calculate average session duration
            durations = [m.interaction_duration for m in recent_interactions]
            profile.engagement_patterns["session_duration_avg"] = np.mean(durations)
            
            # Determine interaction frequency
            days_active = len(set(m.timestamp.date() for m in recent_interactions))
            freq = days_active / 30
            profile.engagement_patterns["interaction_frequency"] = (
                "high" if freq > 0.7 else "medium" if freq > 0.3 else "low"
            )

    def analyze_user_preferences(self, user_id: str, message: str, response: str, 
                               satisfaction_score: float):
        """Analyze user preferences based on interactions."""
        profile = self.get_or_create_profile(user_id)
        
        # Update interaction metrics
        metrics = InteractionMetrics(
            message_length=len(message),
            response_length=len(response),
            interaction_duration=0.0,  # To be calculated
            satisfaction_score=satisfaction_score,
            timestamp=datetime.now()
        )
        self.update_interaction_metrics(user_id, metrics)
        
        # Analyze preferred response style
        if len(profile.interaction_history) >= 5:
            satisfaction_by_length = defaultdict(list)
            for m in profile.interaction_history[-10:]:
                length_category = "concise" if m.response_length < 100 else \
                                "detailed" if m.response_length > 300 else "balanced"
                satisfaction_by_length[length_category].append(m.satisfaction_score)
            
            # Update preferred response style
            best_style = max(satisfaction_by_length.items(), 
                           key=lambda x: np.mean(x[1]) if x[1] else 0)
            profile.preferences.response_style = best_style[0]
        
        self._save_profile(profile)

    def get_personalization_params(self, user_id: str) -> Dict[str, Any]:
        """Get personalization parameters for response generation."""
        profile = self.get_or_create_profile(user_id)
        
        return {
            "response_style": profile.preferences.response_style,
            "technical_level": profile.preferences.technical_level,
            "tone": profile.preferences.tone_preference,
            "complexity": profile.preferences.language_complexity,
            "detail_level": profile.preferences.detail_level,
            "include_examples": profile.preferences.example_preference,
            "include_code": profile.preferences.code_snippet_preference,
            "expertise_areas": profile.expertise_areas,
            "learning_style": profile.learning_style,
            "personality_traits": profile.personality_traits
        }

    def update_expertise(self, user_id: str, topic: str, level: float):
        """Update user's expertise level in a specific topic."""
        profile = self.get_or_create_profile(user_id)
        
        # Exponential moving average for smooth updates
        alpha = 0.3
        current_level = profile.expertise_areas.get(topic, 0.0)
        profile.expertise_areas[topic] = alpha * level + (1 - alpha) * current_level
        
        self._save_profile(profile)

    def get_learning_recommendations(self, user_id: str) -> List[Dict[str, Any]]:
        """Generate personalized learning recommendations."""
        profile = self.get_or_create_profile(user_id)
        
        recommendations = []
        for topic, level in profile.expertise_areas.items():
            if level < 0.7:  # Room for improvement
                recommendations.append({
                    "topic": topic,
                    "current_level": level,
                    "recommended_resources": self._get_resources_for_level(level),
                    "estimated_time": self._estimate_learning_time(level, 0.7)
                })
        
        return sorted(recommendations, key=lambda x: x['current_level'])

    def _get_resources_for_level(self, current_level: float) -> List[str]:
        """Get appropriate learning resources based on expertise level."""
        if current_level < 0.3:
            return ["Beginner tutorials", "Basic documentation", "Practice exercises"]
        elif current_level < 0.6:
            return ["Intermediate guides", "Project examples", "Code reviews"]
        else:
            return ["Advanced documentation", "Research papers", "Expert workshops"]

    def _estimate_learning_time(self, current_level: float, target_level: float) -> int:
        """Estimate hours needed to reach target level."""
        return int((target_level - current_level) * 100)  # Simple linear estimate

    def get_interaction_cluster(self, user_id: str) -> int:
        """Get the interaction cluster for a user based on their behavior."""
        profile = self.get_or_create_profile(user_id)
        
        if not profile.interaction_history:
            return 0
        
        # Extract features for clustering
        features = np.array([
            [m.message_length, m.response_length, m.satisfaction_score]
            for m in profile.interaction_history[-10:]  # Last 10 interactions
        ])
        
        # Normalize features
        features = (features - features.mean(axis=0)) / features.std(axis=0)
        
        # Predict cluster
        cluster = self.interaction_clusters.predict(features.mean(axis=0).reshape(1, -1))
        return int(cluster[0])

    def update_feedback(self, user_id: str, feedback: Dict[str, Any]):
        """Update user profile with new feedback."""
        profile = self.get_or_create_profile(user_id)
        
        feedback["timestamp"] = datetime.now()
        profile.feedback_history.append(feedback)
        
        # Update preferences based on feedback
        if "response_style" in feedback:
            profile.preferences.response_style = feedback["response_style"]
        if "technical_level" in feedback:
            profile.preferences.technical_level = feedback["technical_level"]
        if "tone" in feedback:
            profile.preferences.tone_preference = feedback["tone"]
        
        self._save_profile(profile)
