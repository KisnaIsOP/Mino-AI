import sqlite3
from datetime import datetime
import json
import logging
import os

class Database:
    def __init__(self, db_path="chat_history.db"):
        self.db_path = db_path
        self.init_db()
    
    def get_connection(self):
        """Create and return a database connection with row factory"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_db(self):
        """Initialize database tables"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Conversations table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                message TEXT NOT NULL,
                response TEXT NOT NULL,
                analysis TEXT,
                supervision TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # System metrics table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                response_time FLOAT,
                api_calls INTEGER,
                error_count INTEGER,
                memory_usage FLOAT,
                cpu_usage FLOAT
            )
            """)
            
            # User sessions table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE NOT NULL,
                last_active DATETIME,
                context_data TEXT,
                preferences TEXT
            )
            """)
            
            # Error logs table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS error_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                error_type TEXT NOT NULL,
                error_message TEXT,
                stack_trace TEXT,
                user_id TEXT,
                request_data TEXT
            )
            """)
            
            conn.commit()
            
        except Exception as e:
            logging.error(f"Database initialization error: {str(e)}")
            raise
        finally:
            conn.close()
    
    def save_conversation(self, user_id, message, response, analysis, supervision):
        """Save a conversation interaction"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
            INSERT INTO conversations (
                user_id, message, response, analysis, supervision, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                message,
                response,
                json.dumps(analysis),
                json.dumps(supervision),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            
        except Exception as e:
            logging.error(f"Error saving conversation: {str(e)}")
            raise
        finally:
            conn.close()
    
    def fetch_chat_history(self, user_id, limit=50):
        """Fetch chat history for a user"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
            SELECT message, response, analysis, supervision, timestamp
            FROM conversations
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """, (user_id, limit))
            
            rows = cursor.fetchall()
            
            # Convert rows to list of dicts
            history = []
            for row in rows:
                history.append({
                    'message': row['message'],
                    'response': row['response'],
                    'analysis': json.loads(row['analysis']) if row['analysis'] else None,
                    'supervision': json.loads(row['supervision']) if row['supervision'] else None,
                    'timestamp': row['timestamp']
                })
            
            return history
            
        except Exception as e:
            logging.error(f"Error fetching chat history: {str(e)}")
            raise
        finally:
            conn.close()
    
    def clear_chat_history(self, user_id):
        """Clear chat history for a user"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
            DELETE FROM conversations
            WHERE user_id = ?
            """, (user_id,))
            
            conn.commit()
            
        except Exception as e:
            logging.error(f"Error clearing chat history: {str(e)}")
            raise
        finally:
            conn.close()
    
    def log_system_metrics(self, timestamp, response_time, api_calls, error_count, memory_usage, cpu_usage=0.0):
        """Log system performance metrics"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
            INSERT INTO system_metrics (
                timestamp, response_time, api_calls, error_count, memory_usage, cpu_usage
            ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                timestamp,
                response_time,
                api_calls,
                error_count,
                memory_usage,
                cpu_usage
            ))
            
            conn.commit()
            
        except Exception as e:
            logging.error(f"Error logging system metrics: {str(e)}")
            raise
        finally:
            conn.close()
    
    def log_error(self, error_type, error_message, stack_trace=None, user_id=None, request_data=None):
        """Log error information"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
            INSERT INTO error_logs (
                timestamp, error_type, error_message, stack_trace, user_id, request_data
            ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                error_type,
                error_message,
                stack_trace,
                user_id,
                json.dumps(request_data) if request_data else None
            ))
            
            conn.commit()
            
        except Exception as e:
            logging.error(f"Error logging error: {str(e)}")
            raise
        finally:
            conn.close()
    
    def update_user_session(self, user_id, context_data=None, preferences=None):
        """Update user session data"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
            INSERT INTO user_sessions (
                user_id, last_active, context_data, preferences
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                last_active = excluded.last_active,
                context_data = COALESCE(excluded.context_data, user_sessions.context_data),
                preferences = COALESCE(excluded.preferences, user_sessions.preferences)
            """, (
                user_id,
                datetime.now().isoformat(),
                json.dumps(context_data) if context_data else None,
                json.dumps(preferences) if preferences else None
            ))
            
            conn.commit()
            
        except Exception as e:
            logging.error(f"Error updating user session: {str(e)}")
            raise
        finally:
            conn.close()
    
    def get_user_session(self, user_id):
        """Get user session data"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
            SELECT last_active, context_data, preferences
            FROM user_sessions
            WHERE user_id = ?
            """, (user_id,))
            
            row = cursor.fetchone()
            
            if row:
                return {
                    'last_active': row['last_active'],
                    'context_data': json.loads(row['context_data']) if row['context_data'] else None,
                    'preferences': json.loads(row['preferences']) if row['preferences'] else None
                }
            return None
            
        except Exception as e:
            logging.error(f"Error getting user session: {str(e)}")
            raise
        finally:
            conn.close()
    
    def get_system_metrics(self, hours=24):
        """Get system metrics for the last n hours"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
            SELECT *
            FROM system_metrics
            WHERE timestamp >= datetime('now', '-' || ? || ' hours')
            ORDER BY timestamp DESC
            """, (hours,))
            
            rows = cursor.fetchall()
            
            metrics = []
            for row in rows:
                metrics.append(dict(row))
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error getting system metrics: {str(e)}")
            raise
        finally:
            conn.close()
    
    def cleanup_old_data(self, days=30):
        """Clean up old data from the database"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Clean up old conversations
            cursor.execute("""
            DELETE FROM conversations
            WHERE timestamp < datetime('now', '-' || ? || ' days')
            """, (days,))
            
            # Clean up old system metrics
            cursor.execute("""
            DELETE FROM system_metrics
            WHERE timestamp < datetime('now', '-' || ? || ' days')
            """, (days,))
            
            # Clean up old error logs
            cursor.execute("""
            DELETE FROM error_logs
            WHERE timestamp < datetime('now', '-' || ? || ' days')
            """, (days,))
            
            # Clean up inactive user sessions
            cursor.execute("""
            DELETE FROM user_sessions
            WHERE last_active < datetime('now', '-' || ? || ' days')
            """, (days,))
            
            conn.commit()
            
        except Exception as e:
            logging.error(f"Error cleaning up old data: {str(e)}")
            raise
        finally:
            conn.close()

# Create database instance
db = Database()
