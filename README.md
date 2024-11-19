# Mino AI - Your Advanced AI Companion

## Overview
Mino AI is an intelligent chatbot that combines emotional intelligence with advanced natural language processing capabilities. Created by Kisna Raghuvanshi (16 years old), it's designed to provide engaging, context-aware conversations while understanding and responding to emotional nuances.

## Features
- 🧠 Advanced AI-powered conversations using Google's Generative AI
- 💭 Context-aware responses
- 🎯 Emotion detection and sentiment analysis
- 🔒 Secure and private conversations
- ☁️ Cloud-hosted solution

## Live Demo
Visit [Mino AI on Render](https://mino-ai.onrender.com) to try it out!

## Tech Stack
- Python 3.9
- Flask
- Google Generative AI
- Gunicorn
- Gevent

## Dependencies
```
Flask==2.3.3
Flask-CORS==4.0.0
python-dotenv==1.0.0
google-generativeai==0.8.3
gunicorn==21.2.0
gevent==24.2.1
```

## Environment Variables
- `GOOGLE_API_KEY`: Your Google API key for Generative AI
- `PORT`: Server port (default: 10000)
- `PYTHON_VERSION`: 3.9.0

## Local Development
1. Clone the repository:
```bash
git clone https://github.com/KisnaIsOP/Mino-AI.git
cd Mino-AI
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a .env file with your environment variables:
```
GOOGLE_API_KEY=your_api_key_here
```

5. Run the development server:
```bash
python app.py
```

## Deployment
The application is deployed on Render. For deployment:

1. Push your changes to GitHub
2. Connect your GitHub repository to Render
3. Configure the environment variables
4. Deploy!

## Creator
**Kisna Raghuvanshi**
- Age: 16
- Role: Developer of Mino AI

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing
Contributions, issues, and feature requests are welcome! Feel free to check issues page.
