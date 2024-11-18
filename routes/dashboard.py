from flask import Blueprint, render_template
from datetime import datetime, timedelta
import random  # For demo data, replace with actual data from your system

dashboard = Blueprint('dashboard', __name__)

def generate_demo_data():
    """Generate demo data for the dashboard. Replace with actual data from your system."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    dates = []
    positive = []
    neutral = []
    negative = []
    
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date.isoformat())
        positive.append(random.uniform(0.4, 0.8))
        neutral.append(random.uniform(0.2, 0.5))
        negative.append(random.uniform(0.1, 0.3))
        current_date += timedelta(days=1)
    
    return {
        'dates': dates,
        'positive': positive,
        'neutral': neutral,
        'negative': negative
    }

@dashboard.route('/dashboard')
def show_dashboard():
    # Mock user data - replace with actual user data from your system
    user = {
        'name': 'John Doe',
        'preferences': {
            'response_style': 'Balanced',
            'technical_level': 'Intermediate',
            'preferred_topics': ['AI', 'Machine Learning', 'Data Science']
        }
    }
    
    # Mock statistics - replace with actual stats from your system
    stats = {
        'total_conversations': 128,
        'conversation_increase': 15,
        'avg_sentiment': '8.5',
        'sentiment_increase': 12,
        'topics_count': 25,
        'top_topic': 'Machine Learning'
    }
    
    # Generate emotion timeline data
    emotion_timeline = generate_demo_data()
    
    return render_template(
        'dashboard.html',
        user=user,
        stats=stats,
        emotion_timeline=emotion_timeline
    )
