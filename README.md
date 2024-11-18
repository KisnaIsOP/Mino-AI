# Mino AI

A minimalist AI chat interface powered by Google's Gemini Pro model.

## Features

- Clean, modern user interface
- Real-time AI responses
- Dark/Light theme toggle
- Comprehensive logging system
- Interactive console control

## Setup

1. Install Python 3.8 or higher
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create a `.env` file and add your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

## Running the Application

1. Run the control console:
```bash
log.bat
```

2. Use the console menu to:
   - Start/Stop the server
   - View logs
   - Update dependencies
   - Monitor server status

## Development

The project structure is organized as follows:

```
MinoAI/
├── app.py              # Main Flask application
├── log.bat            # Control console
├── requirements.txt   # Python dependencies
├── .env              # Environment variables
├── static/           # Static assets
│   ├── css/         # Stylesheets
│   └── js/          # JavaScript files
├── templates/        # HTML templates
└── logs/            # Application logs
```

## Technology Stack

- Backend: Flask
- AI: Google Generative AI (Gemini Pro)
- Frontend: Vanilla JavaScript
- Styling: Modern CSS with variables
- Logging: Python's built-in logging

## License

MIT License
