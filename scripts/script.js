// Daily quotes for reflection
const quotes = [
    "Breathe in peace, breathe out tension.",
    "Every moment is a fresh beginning.",
    "In the midst of movement and chaos, keep stillness inside of you.",
    "The present moment is filled with joy and happiness. If you are attentive, you will see it.",
    "Peace comes from within. Do not seek it without.",
];

// DOM Elements
const dayNightToggle = document.getElementById('day-night-toggle');
const breathingCircle = document.querySelector('.breathing-circle');
const sphere = document.querySelector('.sphere');
const journalText = document.getElementById('journal-text');
const quoteContainer = document.querySelector('.quote-container');

// Initialize the application
function init() {
    setupDayNightMode();
    setupBreathingExercise();
    setupSphere();
    loadJournal();
    displayRandomQuote();
    setupSoundControls();
}

// Day/Night Mode
function setupDayNightMode() {
    const currentHour = new Date().getHours();
    if (currentHour >= 18 || currentHour < 6) {
        document.body.classList.add('night-mode');
    }

    dayNightToggle.addEventListener('click', () => {
        document.body.classList.toggle('night-mode');
    });
}

// Breathing Exercise
function setupBreathingExercise() {
    let isBreathing = false;

    breathingCircle.addEventListener('click', () => {
        if (!isBreathing) {
            startBreathingExercise();
            isBreathing = true;
        } else {
            stopBreathingExercise();
            isBreathing = false;
        }
    });
}

function startBreathingExercise() {
    const breatheInterval = setInterval(() => {
        breathingCircle.classList.add('expand');
        setTimeout(() => {
            breathingCircle.classList.remove('expand');
        }, 4000);
    }, 8000);

    breathingCircle.dataset.intervalId = breatheInterval;
}

function stopBreathingExercise() {
    clearInterval(Number(breathingCircle.dataset.intervalId));
    breathingCircle.classList.remove('expand');
}

// Interactive Sphere
function setupSphere() {
    sphere.addEventListener('mousemove', (e) => {
        const rect = sphere.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;
        
        const angleX = (y - centerY) / centerY * 30;
        const angleY = (x - centerX) / centerX * 30;
        
        sphere.style.transform = `rotateX(${angleX}deg) rotateY(${angleY}deg)`;
    });

    sphere.addEventListener('mouseleave', () => {
        sphere.style.transform = 'rotateX(0) rotateY(0)';
    });
}

// Journal Functionality
function saveJournal() {
    localStorage.setItem('mindful-journal', journalText.value);
    showMessage('Journal saved successfully!');
}

function loadJournal() {
    const savedJournal = localStorage.getItem('mindful-journal');
    if (savedJournal) {
        journalText.value = savedJournal;
    }
}

// Sound Controls
function setupSoundControls() {
    const soundButtons = document.querySelectorAll('.sound-btn');
    soundButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            btn.classList.toggle('active');
        });
    });
}

function toggleSound(soundId) {
    const audio = document.getElementById(soundId);
    const button = document.querySelector(`[onclick="toggleSound('${soundId}')"]`);
    
    if (audio.paused) {
        audio.play();
        button.classList.add('active');
    } else {
        audio.pause();
        button.classList.remove('active');
    }
}

// Quote Display
function displayRandomQuote() {
    const randomIndex = Math.floor(Math.random() * quotes.length);
    quoteContainer.textContent = quotes[randomIndex];

    // Change quote every 24 hours
    setInterval(() => {
        const newIndex = Math.floor(Math.random() * quotes.length);
        quoteContainer.textContent = quotes[newIndex];
    }, 24 * 60 * 60 * 1000);
}

// Utility Functions
function showMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.textContent = message;
    messageDiv.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: var(--primary-color);
        color: white;
        padding: 1rem 2rem;
        border-radius: 5px;
        animation: fadeOut 3s forwards;
    `;
    document.body.appendChild(messageDiv);
    setTimeout(() => messageDiv.remove(), 3000);
}

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', init);
