from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import os
import google.generativeai as genai
from dotenv import load_dotenv
from simple_analysis import AdvancedAnalyzer

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set!")

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Initialize analyzer
try:
    analyzer = AdvancedAnalyzer()
    print("✓ Message analyzer initialized successfully")
except Exception as e:
    print(f"! Warning: Could not initialize analyzer: {str(e)}")
    analyzer = None

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mino AI Chat</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/11.1.0/marked.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
        <style>
            :root {
                --primary-color: #4CAF50;
                --primary-hover: #45a049;
                --bg-color: #f5f7fa;
                --text-color: #333;
                --chat-bg: #fff;
                --user-msg-bg: #d1e7dd;
                --ai-msg-bg: #e9ecef;
                --border-color: #ddd;
            }

            .dark-mode {
                --bg-color: #1e1e2f;
                --text-color: #fff;
                --chat-bg: #2d2d44;
                --user-msg-bg: #2e7d32;
                --ai-msg-bg: #37374f;
                --border-color: #444;
            }

            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Inter', sans-serif;
                margin: 0;
                padding: 20px;
                background: var(--bg-color);
                color: var(--text-color);
                transition: background-color 0.3s, color 0.3s;
            }

            .container {
                max-width: 800px;
                margin: 0 auto;
                background: var(--chat-bg);
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                display: flex;
                flex-direction: column;
                height: 85vh;
            }

            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
                padding-bottom: 15px;
                border-bottom: 2px solid var(--border-color);
            }

            .theme-toggle {
                background: none;
                border: none;
                color: var(--text-color);
                cursor: pointer;
                font-size: 1.2rem;
                padding: 5px;
                border-radius: 50%;
                transition: background-color 0.3s;
            }

            .theme-toggle:hover {
                background: rgba(128, 128, 128, 0.1);
            }

            #chat-box {
                flex-grow: 1;
                overflow-y: auto;
                margin-bottom: 20px;
                padding: 15px;
                border: 1px solid var(--border-color);
                border-radius: 10px;
                display: flex;
                flex-direction: column;
                gap: 15px;
            }

            .message {
                max-width: 80%;
                padding: 12px 16px;
                border-radius: 15px;
                animation: fadeInUp 0.3s ease-out;
                position: relative;
                line-height: 1.5;
            }

            .user-message {
                background: var(--user-msg-bg);
                margin-left: auto;
                border-bottom-right-radius: 5px;
            }

            .ai-message {
                background: var(--ai-msg-bg);
                margin-right: auto;
                border-bottom-left-radius: 5px;
            }

            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .input-container {
                display: flex;
                gap: 10px;
                padding: 10px;
                background: var(--chat-bg);
                border: 1px solid var(--border-color);
                border-radius: 10px;
            }

            #user-input {
                flex-grow: 1;
                padding: 12px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                background: var(--bg-color);
                color: var(--text-color);
                transition: background-color 0.3s;
            }

            #user-input:focus {
                outline: none;
                box-shadow: 0 0 0 2px var(--primary-color);
            }

            button {
                padding: 12px 24px;
                background: var(--primary-color);
                color: white;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 500;
                display: flex;
                align-items: center;
                gap: 8px;
                transition: all 0.3s ease;
            }

            button:hover {
                background: var(--primary-hover);
                transform: translateY(-1px);
            }

            button:active {
                transform: translateY(0);
            }

            .timestamp {
                font-size: 0.75rem;
                color: #666;
                margin-top: 5px;
            }

            #typing-indicator {
                display: none;
                align-items: center;
                gap: 8px;
                color: #666;
                font-style: italic;
                padding: 8px 12px;
                background: var(--ai-msg-bg);
                border-radius: 15px;
                margin-bottom: 10px;
                width: fit-content;
            }

            .typing-animation {
                display: flex;
                gap: 4px;
            }

            .dot {
                width: 4px;
                height: 4px;
                background: #666;
                border-radius: 50%;
                animation: bounce 1.4s infinite ease-in-out;
            }

            .dot:nth-child(1) { animation-delay: -0.32s; }
            .dot:nth-child(2) { animation-delay: -0.16s; }

            @keyframes bounce {
                0%, 80%, 100% { transform: translateY(0); }
                40% { transform: translateY(-6px); }
            }

            @media (max-width: 768px) {
                body {
                    padding: 10px;
                }
                
                .container {
                    height: 90vh;
                }

                .message {
                    max-width: 90%;
                }

                button {
                    padding: 12px;
                }

                button span {
                    display: none;
                }
            }

            .message-content {
                line-height: 1.6;
            }

            .message-content p {
                margin: 0.5em 0;
            }

            .message-content code {
                background: rgba(0, 0, 0, 0.05);
                padding: 0.2em 0.4em;
                border-radius: 3px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 0.9em;
            }

            .dark-mode .message-content code {
                background: rgba(255, 255, 255, 0.1);
            }

            .message-content pre {
                background: rgba(0, 0, 0, 0.05);
                padding: 1em;
                border-radius: 5px;
                overflow-x: auto;
                margin: 0.5em 0;
            }

            .dark-mode .message-content pre {
                background: rgba(255, 255, 255, 0.1);
            }

            .message-content pre code {
                background: none;
                padding: 0;
                border-radius: 0;
            }

            .message-content blockquote {
                border-left: 4px solid var(--primary-color);
                margin: 0.5em 0;
                padding-left: 1em;
                color: #666;
            }

            .dark-mode .message-content blockquote {
                color: #aaa;
            }

            .message-content ul, .message-content ol {
                margin: 0.5em 0;
                padding-left: 1.5em;
            }

            .message-content table {
                border-collapse: collapse;
                margin: 0.5em 0;
                width: 100%;
            }

            .message-content th, .message-content td {
                border: 1px solid var(--border-color);
                padding: 0.4em 0.6em;
            }

            .message-content img {
                max-width: 100%;
                border-radius: 5px;
                margin: 0.5em 0;
            }

            .message-content a {
                color: var(--primary-color);
                text-decoration: none;
            }

            .message-content a:hover {
                text-decoration: underline;
            }

            /* Emoji support */
            .message-content .emoji {
                font-family: "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
            }
            
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Mino AI Chat</h1>
                <button class="theme-toggle" onclick="toggleTheme()">
                    <i class="fas fa-moon"></i>
                </button>
            </div>
            <div id="chat-box"></div>
            <div id="typing-indicator">
                <span>AI is thinking</span>
                <div class="typing-animation">
                    <div class="dot"></div>
                    <div class="dot"></div>
                    <div class="dot"></div>
                </div>
            </div>
            <div class="input-container">
                <input type="text" id="user-input" placeholder="Type your message...">
                <button onclick="sendMessage()">
                    <i class="fas fa-paper-plane"></i>
                    <span>Send</span>
                </button>
            </div>
        </div>
        <script>
            const chatBox = document.getElementById('chat-box');
            const userInput = document.getElementById('user-input');
            const typingIndicator = document.getElementById('typing-indicator');

            // Theme handling
            function toggleTheme() {
                document.body.classList.toggle('dark-mode');
                const themeIcon = document.querySelector('.theme-toggle i');
                if (document.body.classList.contains('dark-mode')) {
                    themeIcon.classList.remove('fa-moon');
                    themeIcon.classList.add('fa-sun');
                    localStorage.setItem('theme', 'dark');
                } else {
                    themeIcon.classList.remove('fa-sun');
                    themeIcon.classList.add('fa-moon');
                    localStorage.setItem('theme', 'light');
                }
            }

            // Load saved theme
            if (localStorage.getItem('theme') === 'dark') {
                document.body.classList.add('dark-mode');
                const themeIcon = document.querySelector('.theme-toggle i');
                themeIcon.classList.remove('fa-moon');
                themeIcon.classList.add('fa-sun');
            }

            // Configure marked options
            marked.setOptions({
                gfm: true, // GitHub Flavored Markdown
                breaks: true, // Convert line breaks to <br>
                highlight: function(code, lang) {
                    if (lang && hljs.getLanguage(lang)) {
                        return hljs.highlight(code, { language: lang }).value;
                    }
                    return hljs.highlightAuto(code).value;
                }
            });

            function addMessage(content, isUser = false) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${isUser ? 'user-message' : 'ai-message'}`;
                
                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';
                
                // Parse markdown only for AI messages
                if (isUser) {
                    contentDiv.textContent = content;
                } else {
                    // Process emojis in content
                    content = content.replace(/:\w+:/g, match => {
                        const emoji = match.replace(/:/g, '');
                        return `<span class="emoji">${emoji}</span>`;
                    });
                    
                    // Convert markdown to HTML
                    contentDiv.innerHTML = marked.parse(content);
                    
                    // Highlight code blocks
                    contentDiv.querySelectorAll('pre code').forEach((block) => {
                        hljs.highlightBlock(block);
                    });
                }
                
                messageDiv.appendChild(contentDiv);
                
                const timestamp = document.createElement('div');
                timestamp.className = 'timestamp';
                timestamp.textContent = new Date().toLocaleTimeString();
                messageDiv.appendChild(timestamp);
                
                chatBox.appendChild(messageDiv);
                chatBox.scrollTop = chatBox.scrollHeight;
            }

            async function sendMessage() {
                const message = userInput.value.trim();
                if (!message) return;
                
                // Disable input and button while processing
                userInput.disabled = true;
                const sendButton = document.querySelector('button');
                sendButton.disabled = true;
                
                addMessage(message, true);
                userInput.value = '';
                typingIndicator.style.display = 'flex';
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ message })
                    });
                    
                    const data = await response.json();
                    typingIndicator.style.display = 'none';
                    
                    if (data.error) {
                        addMessage('Sorry, there was an error processing your message.');
                    } else {
                        addMessage(data.response);
                    }
                } catch (error) {
                    typingIndicator.style.display = 'none';
                    addMessage('Sorry, there was an error processing your message.');
                } finally {
                    // Re-enable input and button
                    userInput.disabled = false;
                    sendButton.disabled = false;
                    userInput.focus();
                }
            }

            userInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });

            // Focus input on load
            userInput.focus();
        </script>
    </body>
    </html>
    '''

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400

        # Analyze message if analyzer is available
        analysis = None
        if analyzer:
            try:
                analysis = analyzer.analyze_message(message)
            except Exception as e:
                print(f"Warning: Message analysis failed: {str(e)}")

        # Generate base response using Gemini
        response = model.generate_content(message)
        response_text = response.text

        # Enhance response if analyzer is available
        if analyzer and analysis:
            try:
                response_text = analyzer.enhance_response(message, response_text, analysis)
            except Exception as e:
                print(f"Warning: Response enhancement failed: {str(e)}")
        
        return jsonify({
            'response': response_text,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7007))
    host = os.environ.get('HOST', '127.0.0.1')
    print(f"\n=== Mino AI Server starting on port {port} ===")
    app.run(host=host, port=port, debug=False)
