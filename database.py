import sqlite3
from datetime import datetime
import json

class Database:
    def __init__(self):
        self.conn = sqlite3.connect('mino_ai.db', check_same_thread=False)
        self.create_tables()
    
    def create_tables(self):
        cursor = self.conn.cursor()
        
        # Conversations table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            timestamp DATETIME,
            message TEXT,
            response TEXT,
            analysis TEXT,
            supervision TEXT,
            feedback_rating INTEGER,
            feedback_text TEXT
        )''')
        
        # User preferences table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id TEXT PRIMARY KEY,
            theme TEXT DEFAULT 'light',
            analysis_visible BOOLEAN DEFAULT true,
            supervision_visible BOOLEAN DEFAULT true,
            created_at DATETIME,
            updated_at DATETIME
        )''')
        
        # System metrics table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            response_time FLOAT,
            api_calls INTEGER,
            error_count INTEGER,
            memory_usage FLOAT
        )''')
        
        self.conn.commit()
    
    def save_conversation(self, user_id, message, response, analysis, supervision):
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO conversations (user_id, timestamp, message, response, analysis, supervision)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, datetime.now(), message, response, analysis, supervision))
        self.conn.commit()
    
    def get_conversation_history(self, user_id, limit=10):
        cursor = self.conn.cursor()
        cursor.execute('''
        SELECT message, response, analysis, supervision, timestamp
        FROM conversations
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
        ''', (user_id, limit))
        return cursor.fetchall()
    
    def save_user_preferences(self, user_id, preferences):
        cursor = self.conn.cursor()
        now = datetime.now()
        cursor.execute('''
        INSERT OR REPLACE INTO user_preferences 
        (user_id, theme, analysis_visible, supervision_visible, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, preferences.get('theme', 'light'),
              preferences.get('analysis_visible', True),
              preferences.get('supervision_visible', True),
              now, now))
        self.conn.commit()
    
    def get_user_preferences(self, user_id):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM user_preferences WHERE user_id = ?', (user_id,))
        result = cursor.fetchone()
        if result:
            return {
                'theme': result[1],
                'analysis_visible': bool(result[2]),
                'supervision_visible': bool(result[3])
            }
        return None
    
    def log_system_metrics(self, response_time, api_calls, error_count, memory_usage):
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO system_metrics 
        (timestamp, response_time, api_calls, error_count, memory_usage)
        VALUES (?, ?, ?, ?, ?)
        ''', (datetime.now(), response_time, api_calls, error_count, memory_usage))
        self.conn.commit()
    
    def get_system_metrics(self, hours=24):
        cursor = self.conn.cursor()
        cursor.execute('''
        SELECT * FROM system_metrics
        WHERE timestamp >= datetime('now', '-' || ? || ' hours')
        ORDER BY timestamp DESC
        ''', (hours,))
        return cursor.fetchall()
    
    def save_feedback(self, conversation_id, rating, feedback_text):
        cursor = self.conn.cursor()
        cursor.execute('''
        UPDATE conversations
        SET feedback_rating = ?, feedback_text = ?
        WHERE id = ?
        ''', (rating, feedback_text, conversation_id))
        self.conn.commit()
    
    def close(self):
        self.conn.close()
