# Mino AI Quick Start Guide

## Project Continuation Code
MINO_CHECKPOINT_ODYSSEY_V1

## Quick Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Copy `.env.template` to `.env` and fill in your values:
```bash
cp .env.template .env
```

3. Generate SSL certificates:
```bash
python generate_cert.py
```

4. Start the server:
```bash
python app.py
```

## Key Features

### 1. Tetra AI System
The heart of Mino AI is its Tetra system - four specialized AI agents working together:
- Analyst AI: Processes data and recognizes patterns
- Emotional AI: Handles sentiment and emotional intelligence
- Memory AI: Manages context and user history
- Response AI: Generates natural language responses

Benefits:
- Enhanced accuracy through multiple perspectives
- Specialized processing for different tasks
- Real-time collaboration between AIs
- Robust error handling
- Scalable architecture

### 2. Real-time Chat with Emotional Intelligence
- Open `http://localhost:5000` in your browser
- Start chatting to see the Tetra system in action:
  - Watch emotional analysis from Emotional AI
  - See pattern recognition from Analyst AI
  - Experience contextual responses from Memory AI
  - Get natural responses from Response AI

### 3. Tone Analysis
- Watch real-time tone feedback
- Get suggestions for improvement
- Try alternative phrasings
- See emotional patterns

### 4. User Profiling
- System learns from interactions
- Personalizes responses
- Adapts to your style
- Maintains conversation context

## Development Tips
1. Check `PROJECT_DOCUMENTATION.md` for full details
2. Use `MINO_CHECKPOINT_ODYSSEY_V1` in new Cascade sessions
3. Follow the modular architecture
4. Keep security in mind

## Common Commands
- Start server: `python app.py`
- Run tests: `python -m pytest tests/`
- Generate certs: `python generate_cert.py`
- Clear cache: `python clear_cache.py`
- Monitor Tetra: `python monitor_tetra.py`

## Need Help?
1. Check `PROJECT_DOCUMENTATION.md`
2. Review logs in `logs/`
3. Check the `.env` configuration
4. Verify SSL certificates in `certs/`
5. Monitor Tetra logs in `logs/tetra/`

Remember to use `MINO_CHECKPOINT_ODYSSEY_V1` when starting new development sessions!
