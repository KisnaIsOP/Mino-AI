from flask_socketio import SocketIO, emit
from flask import request
from analysis import AnalysisAI
from supervisor import SupervisorAI
from tone_analyzer import ToneAnalyzer
import json
import asyncio
from datetime import datetime

socketio = SocketIO()
analysis_ai = AnalysisAI()
supervisor_ai = SupervisorAI()
tone_analyzer = ToneAnalyzer()

@socketio.on('connect')
def handle_connect():
    """Handle new client connections"""
    print(f"Client connected: {request.sid}")
    emit('system_message', {
        'message': 'Connected to Mino AI',
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnections"""
    print(f"Client disconnected: {request.sid}")

@socketio.on('user_message')
def handle_message(data):
    """Handle incoming user messages"""
    message = data.get('message', '')
    
    # Emit typing indicator
    emit('bot_typing', {'status': True})
    
    try:
        # Analyze message sentiment and emotion
        sentiment = analysis_ai.analyze_sentiment(message)
        emotion = analysis_ai.analyze_emotion(message)
        
        # Get message context and intent
        context = analysis_ai.analyze_context(message)
        intent = analysis_ai.analyze_intent(message)
        
        # Analyze tone and get suggestions
        tone_analysis = tone_analyzer.analyze_tone(message)
        
        # Process message through MinoAI
        response = process_message(message, sentiment, emotion, context, intent)
        
        # Generate suggestions based on context and emotion
        suggestions = generate_suggestions(context, emotion)
        
        # Get system performance metrics
        metrics = supervisor_ai.get_performance_metrics()
        
        # Emit response with all analysis data
        emit('bot_response', {
            'response': response,
            'sentiment': sentiment,
            'emotion': emotion,
            'tone_analysis': tone_analysis,
            'suggestions': suggestions,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
        
        # Emit tone feedback separately for UI updates
        emit('tone_feedback', {
            'analysis': tone_analysis,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        emit('error', {
            'message': f"Error processing message: {str(e)}",
            'timestamp': datetime.now().isoformat()
        })
    
    finally:
        # Stop typing indicator
        emit('bot_typing', {'status': False})

def process_message(message, sentiment, emotion, context, intent):
    """Process the message using MinoAI system"""
    # Add your existing MinoAI processing logic here
    return "Processed response"

def generate_suggestions(context, emotion):
    """Generate contextual suggestions based on conversation context and emotion"""
    suggestions = []
    
    if emotion.get('type') == 'negative':
        suggestions.append({
            'type': 'emotional_support',
            'text': "I notice you might be feeling frustrated. Would you like to explore what's bothering you?"
        })
    
    if context.get('topic') in ['technical', 'problem_solving']:
        suggestions.append({
            'type': 'technical_assistance',
            'text': "Would you like me to break this down into smaller steps?"
        })
    
    return suggestions

def init_app(app):
    """Initialize SocketIO with the Flask app"""
    socketio.init_app(app, 
                     cors_allowed_origins="*",
                     async_mode='eventlet',
                     ping_timeout=10,
                     ping_interval=5)
    return socketio
