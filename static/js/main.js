// DOM Elements
const chatMessages = document.getElementById('chat-messages');
const messageInput = document.getElementById('message-input');
const typingIndicator = document.getElementById('typing-indicator');
const aiStatus = document.getElementById('ai-status');
const sendButton = document.getElementById('send-button');

// State
let isProcessing = false;

// Auto-resize textarea as user types
messageInput.addEventListener('input', () => {
    messageInput.style.height = 'auto';
    messageInput.style.height = messageInput.scrollHeight + 'px';
});

// Message handling
async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || isProcessing) return;

    isProcessing = true;
    updateUIState(true);
    
    // Add user message
    addMessage(message, 'user');
    
    // Clear input and reset height
    messageInput.value = '';
    messageInput.style.height = 'auto';
    
    try {
        // Show typing indicator
        typingIndicator.style.display = 'flex';
        aiStatus.querySelector('.status-text').textContent = 'Thinking...';
        
        // Send to backend and get AI response
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message })
        });
        
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        
        const data = await response.json();
        
        if (data.status === 'error') {
            throw new Error(data.message || 'Unknown error occurred');
        }
        
        // Hide typing indicator
        typingIndicator.style.display = 'none';
        aiStatus.querySelector('.status-text').textContent = 'Ready';
        
        // Add AI response
        addMessage(data.response, 'ai');
        
    } catch (error) {
        console.error('Error:', error);
        typingIndicator.style.display = 'none';
        aiStatus.querySelector('.status-text').textContent = 'Error';
        addMessage('Sorry, there was an error processing your request. Please try again.', 'ai error');
    } finally {
        isProcessing = false;
        updateUIState(false);
    }
}

function addMessage(text, type) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', type);
    messageDiv.textContent = text;
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

function updateUIState(processing) {
    sendButton.disabled = processing;
    messageInput.disabled = processing;
    aiStatus.querySelector('.status-dot').style.opacity = processing ? '0.5' : '1';
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Event Listeners
sendButton.addEventListener('click', sendMessage);

messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Initial setup
messageInput.focus();
