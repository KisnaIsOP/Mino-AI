const audioInput = document.getElementById('audioUpload');
const canvas = document.getElementById('visualizer');
const ctx = canvas.getContext('2d');
const visualStyleSelect = document.getElementById('visualStyle');

// Music Player Elements
const playPauseBtn = document.querySelector('.play-pause');
const prevBtn = document.querySelector('.prev');
const nextBtn = document.querySelector('.next');
const shuffleBtn = document.querySelector('.shuffle');
const repeatBtn = document.querySelector('.repeat');
const volumeSlider = document.querySelector('.volume-slider');
const volumeIcon = document.querySelector('.volume-control i');
const progressBar = document.querySelector('.progress');
const progressArea = document.querySelector('.progress-bar');
const currentTimeSpan = document.querySelector('.current');
const durationSpan = document.querySelector('.duration');
const trackName = document.querySelector('.track-name');
const artistName = document.querySelector('.artist-name');
const playlist = document.querySelector('.playlist');

let audioContext;
let analyser;
let source;
let animationId;
let audioElement = null;
let tracks = [];
let currentTrackIndex = 0;
let isRepeat = false;
let isShuffle = false;

// Initialize audio context when user interacts with the page
document.addEventListener('click', initializeAudioContext, { once: true });

function initializeAudioContext() {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
}

audioInput.addEventListener('change', handleAudioUpload);

function handleAudioUpload() {
    const files = Array.from(this.files);
    if (!files.length) return;

    // Reset playlist
    tracks = files.map((file, index) => ({
        file,
        name: file.name.replace(/\.[^/.]+$/, ""),
        artist: 'Local File',
        duration: 0
    }));

    // Clear playlist UI
    playlist.innerHTML = '';

    // Create playlist items
    tracks.forEach((track, index) => {
        const item = createPlaylistItem(track, index);
        playlist.appendChild(item);
    });

    currentTrackIndex = 0;
    playTrack(currentTrackIndex);
}

function createPlaylistItem(track, index) {
    const div = document.createElement('div');
    div.className = 'playlist-item' + (index === currentTrackIndex ? ' active' : '');
    div.innerHTML = `
        <span class="playlist-item-number">${index + 1}</span>
        <div class="playlist-item-info">
            <div class="playlist-item-name">${track.name}</div>
            <div class="playlist-item-duration">${formatTime(track.duration)}</div>
        </div>
    `;
    div.addEventListener('click', () => {
        currentTrackIndex = index;
        playTrack(currentTrackIndex);
    });
    return div;
}

function updatePlaylistUI() {
    const items = playlist.querySelectorAll('.playlist-item');
    items.forEach((item, index) => {
        item.className = 'playlist-item' + (index === currentTrackIndex ? ' active' : '');
    });
}

function playTrack(index) {
    if (!tracks[index]) return;

    // Stop current audio and visualization
    if (animationId) {
        cancelAnimationFrame(animationId);
    }
    if (audioElement) {
        audioElement.pause();
        audioElement.src = '';
    }

    // Create new audio element
    audioElement = new Audio();
    audioElement.src = URL.createObjectURL(tracks[index].file);

    // Update UI
    updateTrackInfo();
    updatePlayPauseButton();
    updatePlaylistUI();

    // Set up audio element event listeners
    setupAudioEventListeners();

    // Connect to Web Audio API
    setupAudioNodes();

    // Start playing
    audioElement.play();
}

function setupAudioEventListeners() {
    audioElement.addEventListener('loadedmetadata', () => {
        tracks[currentTrackIndex].duration = audioElement.duration;
        updateTrackInfo();
    });

    audioElement.addEventListener('timeupdate', updateProgress);
    audioElement.addEventListener('ended', handleTrackEnd);
    audioElement.addEventListener('play', () => updatePlayPauseButton(true));
    audioElement.addEventListener('pause', () => updatePlayPauseButton(false));
}

function setupAudioNodes() {
    source = audioContext.createMediaElementSource(audioElement);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    source.connect(analyser);
    analyser.connect(audioContext.destination);

    draw(dataArray, bufferLength);
}

function updateTrackInfo() {
    const track = tracks[currentTrackIndex];
    trackName.textContent = track.name;
    artistName.textContent = track.artist;
    durationSpan.textContent = formatTime(track.duration);
}

function updateProgress() {
    const progress = (audioElement.currentTime / audioElement.duration) * 100;
    progressBar.style.width = `${progress}%`;
    currentTimeSpan.textContent = formatTime(audioElement.currentTime);
}

function handleTrackEnd() {
    if (isRepeat) {
        audioElement.currentTime = 0;
        audioElement.play();
    } else if (isShuffle) {
        playRandomTrack();
    } else {
        playNextTrack();
    }
}

function formatTime(seconds) {
    if (!seconds || isNaN(seconds)) return '0:00';
    const minutes = Math.floor(seconds / 60);
    seconds = Math.floor(seconds % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

function updatePlayPauseButton(isPlaying) {
    playPauseBtn.innerHTML = `<i class="fas fa-${isPlaying ? 'pause' : 'play'}"></i>`;
}

function playNextTrack() {
    if (isShuffle) {
        playRandomTrack();
    } else {
        currentTrackIndex = (currentTrackIndex + 1) % tracks.length;
        playTrack(currentTrackIndex);
    }
}

function playPreviousTrack() {
    if (audioElement && audioElement.currentTime > 3) {
        audioElement.currentTime = 0;
    } else {
        currentTrackIndex = (currentTrackIndex - 1 + tracks.length) % tracks.length;
        playTrack(currentTrackIndex);
    }
}

function playRandomTrack() {
    const newIndex = Math.floor(Math.random() * tracks.length);
    currentTrackIndex = newIndex;
    playTrack(currentTrackIndex);
}

// Event Listeners
playPauseBtn.addEventListener('click', () => {
    if (!audioElement) return;
    if (audioElement.paused) {
        audioElement.play();
    } else {
        audioElement.pause();
    }
});

prevBtn.addEventListener('click', playPreviousTrack);
nextBtn.addEventListener('click', playNextTrack);

shuffleBtn.addEventListener('click', () => {
    isShuffle = !isShuffle;
    shuffleBtn.style.color = isShuffle ? '#1DB954' : '#b3b3b3';
});

repeatBtn.addEventListener('click', () => {
    isRepeat = !isRepeat;
    repeatBtn.style.color = isRepeat ? '#1DB954' : '#b3b3b3';
});

progressArea.addEventListener('click', (e) => {
    if (!audioElement) return;
    const progressWidth = progressArea.clientWidth;
    const clickedWidth = e.offsetX;
    audioElement.currentTime = (clickedWidth / progressWidth) * audioElement.duration;
});

volumeSlider.addEventListener('input', (e) => {
    if (audioElement) {
        audioElement.volume = e.target.value;
        updateVolumeIcon(e.target.value);
    }
});

function updateVolumeIcon(value) {
    if (value === 0) {
        volumeIcon.className = 'fas fa-volume-mute';
    } else if (value < 0.5) {
        volumeIcon.className = 'fas fa-volume-down';
    } else {
        volumeIcon.className = 'fas fa-volume-up';
    }
}

// Secret Interactive Features
let easterEggMode = false;
let colorMode = 'default';
let lastKeypressTime = 0;
let secretCode = '';

document.addEventListener('keydown', (e) => {
    const currentTime = new Date().getTime();
    if (currentTime - lastKeypressTime > 1000) {
        secretCode = '';
    }
    lastKeypressTime = currentTime;
    secretCode += e.key.toLowerCase();

    // Check for secret combinations
    checkSecretCombinations();

    // Visualization controls
    switch(e.key) {
        case ' ':  // Space bar
            if (audioElement.paused) {
                audioElement.play();
            } else {
                audioElement.pause();
            }
            break;
        case 'ArrowRight':
            const visualStyles = ['bars', 'circles', 'wave', 'dnaHelix', 'circularWave'];
            let currentStyleIndex = visualStyles.indexOf(visualStyleSelect.value);
            currentStyleIndex = (currentStyleIndex + 1) % visualStyles.length;
            visualStyleSelect.value = visualStyles[currentStyleIndex];
            break;
        case 'ArrowLeft':
            const visualStyles2 = ['bars', 'circles', 'wave', 'dnaHelix', 'circularWave'];
            let currentStyleIndex2 = visualStyles2.indexOf(visualStyleSelect.value);
            currentStyleIndex2 = (currentStyleIndex2 - 1 + visualStyles2.length) % visualStyles2.length;
            visualStyleSelect.value = visualStyles2[currentStyleIndex2];
            break;
        case 'ArrowUp':
            audioElement.volume = Math.min(1, audioElement.volume + 0.1);
            updateVolumeIcon(audioElement.volume);
            break;
        case 'ArrowDown':
            audioElement.volume = Math.max(0, audioElement.volume - 0.1);
            updateVolumeIcon(audioElement.volume);
            break;
    }
});

// Mood and Color Management
const moodColors = {
    calm: ['#1E88E5', '#0D47A1'],      // Cool blues
    neutral: ['#7E57C2', '#4527A0'],    // Purple tones
    energetic: ['#FF4081', '#F50057']   // Hot pinks
};

function getMoodColors(averageFrequency) {
    if (averageFrequency < 80) return moodColors.calm;
    if (averageFrequency < 150) return moodColors.neutral;
    return moodColors.energetic;
}

// Simple Particle System
class Particle {
    constructor(x, y, color) {
        this.x = x;
        this.y = y;
        this.color = color;
        this.size = Math.random() * 3 + 1;
        this.speedX = Math.random() * 2 - 1;
        this.speedY = -Math.random() * 3;
        this.life = 1;
    }

    update() {
        this.x += this.speedX;
        this.y += this.speedY;
        this.speedY += 0.1;
        this.life -= 0.02;
        this.size = Math.max(0.1, this.size - 0.05);
    }

    draw(ctx) {
        ctx.fillStyle = `rgba(${this.color}, ${this.life})`;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
    }
}

let particles = [];

function createParticles(x, y, color) {
    for (let i = 0; i < 3; i++) {
        particles.push(new Particle(x, y, color));
    }
}

// 3D Perspective Settings
let perspective = 800;
let rotationX = 0;
let rotationY = 0;
let isRotating = false;

// Convert 2D point to 3D with perspective
function to3D(x, y, z) {
    const scale = perspective / (perspective + z);
    return {
        x: canvas.width/2 + (x - canvas.width/2) * scale,
        y: canvas.height/2 + (y - canvas.height/2) * scale,
        scale: scale
    };
}

// Enhanced bar visualization with 3D
function drawBars3D(dataArray, bufferLength, moodColors) {
    const barWidth = (canvas.width / bufferLength) * 2.5;
    const barDepth = 50; // Depth of 3D bars
    
    // Sort bars by z-index for proper rendering
    const bars = [];
    for (let i = 0; i < bufferLength; i++) {
        const x = i * (barWidth + 1) - canvas.width/2;
        const barHeight = dataArray[i] * 1.5;
        const z = Math.sin(i * 0.1 + rotationY) * 100;
        bars.push({ x, height: barHeight, z, index: i });
    }
    bars.sort((a, b) => b.z - a.z);

    // Draw bars with 3D perspective
    bars.forEach(bar => {
        const barHeight = bar.height;
        const x = bar.x;
        const z = bar.z;
        
        // Calculate 3D points for the bar
        const frontBottom = to3D(x, canvas.height, z);
        const frontTop = to3D(x, canvas.height - barHeight, z);
        const backBottom = to3D(x, canvas.height, z + barDepth);
        const backTop = to3D(x, canvas.height - barHeight, z + barDepth);
        
        // Get color based on height
        const colorIndex = Math.min(Math.floor(barHeight / 150), 1);
        const barColor = moodColors[colorIndex];
        const darkColor = barColor.replace('rgb', 'rgba').replace(')', ',0.7)');
        
        // Draw front face
        ctx.fillStyle = barColor;
        ctx.fillRect(
            frontBottom.x, 
            frontTop.y, 
            barWidth * frontBottom.scale, 
            (frontBottom.y - frontTop.y)
        );
        
        // Draw top face if visible
        if (rotationX < 0) {
            ctx.fillStyle = darkColor;
            ctx.beginPath();
            ctx.moveTo(frontTop.x, frontTop.y);
            ctx.lineTo(frontTop.x + barWidth * frontTop.scale, frontTop.y);
            ctx.lineTo(backTop.x + barWidth * backTop.scale, backTop.y);
            ctx.lineTo(backTop.x, backTop.y);
            ctx.fill();
        }
        
        // Draw side face
        ctx.fillStyle = darkColor;
        ctx.beginPath();
        ctx.moveTo(frontTop.x + barWidth * frontTop.scale, frontTop.y);
        ctx.lineTo(backTop.x + barWidth * backTop.scale, backTop.y);
        ctx.lineTo(backBottom.x + barWidth * backBottom.scale, backBottom.y);
        ctx.lineTo(frontBottom.x + barWidth * frontBottom.scale, frontBottom.y);
        ctx.fill();
        
        // Create particles for high bars
        if (barHeight > 150 && Math.random() > 0.9) {
            const particleColor = barColor.replace('#', '').match(/.{2}/g)
                .map(hex => parseInt(hex, 16)).join(', ');
            createParticles(frontTop.x + barWidth * frontTop.scale/2, frontTop.y, particleColor);
        }
    });
}

// Waveform overlay
function drawWaveformOverlay(dataArray, bufferLength) {
    const timeData = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteTimeDomainData(timeData);
    
    ctx.save();
    ctx.globalAlpha = 0.3;
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1;
    
    // Draw waveform
    ctx.beginPath();
    const sliceWidth = canvas.width / bufferLength;
    let x = 0;
    
    for (let i = 0; i < bufferLength; i++) {
        const v = timeData[i] / 128.0;
        const y = v * canvas.height/4 + canvas.height/8;
        
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
        x += sliceWidth;
    }
    
    ctx.stroke();
    ctx.restore();
}

// Update keyboard controls for 3D rotation
document.addEventListener('keydown', (e) => {
    switch(e.key) {
        case 'w':
            rotationX = Math.max(-45, rotationX - 5);
            break;
        case 's':
            rotationX = Math.min(45, rotationX + 5);
            break;
        case 'a':
            rotationY -= 5;
            break;
        case 'd':
            rotationY += 5;
            break;
        case 'r':
            isRotating = !isRotating;
            break;
    }
});

function draw(dataArray, bufferLength) {
    animationId = requestAnimationFrame(() => draw(dataArray, bufferLength));
    analyser.getByteFrequencyData(dataArray);
    
    ctx.fillStyle = 'rgb(0, 0, 0)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const visualStyle = visualStyleSelect.value;
    const average = dataArray.reduce((a, b) => a + b) / bufferLength;
    const currentMoodColors = getMoodColors(average);
    
    if (colorMode === 'disco') {
        ctx.globalAlpha = 0.8;
    } else if (colorMode === 'matrix') {
        ctx.globalAlpha = 0.9;
        ctx.fillStyle = 'rgba(0, 255, 0, 0.1)';
    } else if (colorMode === 'rainbow') {
        ctx.globalAlpha = 1;
    }

    // Update rotation if auto-rotate is enabled
    if (isRotating) {
        rotationY += 0.5;
    }

    switch(visualStyle) {
        case 'bars':
            drawBars3D(dataArray, bufferLength, currentMoodColors);
            break;
        case 'circles':
            drawCircles(dataArray, bufferLength);
            break;
        case 'wave':
            drawWave(dataArray, bufferLength);
            break;
        case 'dnaHelix':
            drawDNAHelix(dataArray, bufferLength);
            break;
        case 'circularWave':
            drawCircularWave(dataArray, bufferLength, currentMoodColors);
            break;
    }

    // Draw waveform overlay
    drawWaveformOverlay(dataArray, bufferLength);

    // Update and draw particles
    particles = particles.filter(particle => particle.life > 0);
    particles.forEach(particle => {
        particle.update();
        particle.draw(ctx);
    });

    ctx.globalAlpha = 1;
}

function drawCircles(dataArray, bufferLength) {
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    
    for (let i = 0; i < bufferLength; i++) {
        const radius = dataArray[i] * 0.5;
        const hue = (i / bufferLength) * 360;
        
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
        ctx.strokeStyle = `hsl(${hue}, 100%, 50%)`;
        ctx.lineWidth = 2;
        ctx.stroke();
    }
}

function drawWave(dataArray, bufferLength) {
    ctx.beginPath();
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgb(0, 255, 0)';

    const sliceWidth = canvas.width / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = v * canvas.height / 2;

        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }

        x += sliceWidth;
    }

    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
}

function drawDNAHelix(dataArray, bufferLength) {
    const time = Date.now() * 0.001; // Current time in seconds
    const centerY = canvas.height / 2;
    const amplitude = 100; // Height of the wave
    const frequency = 0.02; // How tight the helix is
    const speed = 2; // Speed of rotation
    const spacing = canvas.width / (bufferLength / 2); // Space between points
    
    // Draw connecting lines (base pairs)
    ctx.lineWidth = 2;
    
    for (let i = 0; i < bufferLength / 2; i++) {
        const x = i * spacing;
        const intensity = dataArray[i] / 255; // Normalize the frequency data
        
        // Calculate y positions for both strands
        const y1 = centerY + Math.sin(x * frequency + time * speed) * amplitude * (0.5 + intensity * 0.5);
        const y2 = centerY + Math.sin(x * frequency + time * speed + Math.PI) * amplitude * (0.5 + intensity * 0.5);
        
        // Draw base pair connections with gradient based on audio intensity
        const gradient = ctx.createLinearGradient(
            x, y1,
            x, y2
        );
        gradient.addColorStop(0, `hsla(${intensity * 360}, 100%, 50%, 0.8)`);
        gradient.addColorStop(1, `hsla(${(intensity * 360 + 180) % 360}, 100%, 50%, 0.8)`);
        
        ctx.beginPath();
        ctx.strokeStyle = gradient;
        ctx.moveTo(x, y1);
        ctx.lineTo(x, y2);
        ctx.stroke();
        
        // Draw nucleotide points
        const radius = 3 + intensity * 5; // Point size based on frequency
        
        // Upper strand point
        ctx.beginPath();
        ctx.fillStyle = `hsl(${intensity * 360}, 100%, 50%)`;
        ctx.arc(x, y1, radius, 0, Math.PI * 2);
        ctx.fill();
        
        // Lower strand point
        ctx.beginPath();
        ctx.fillStyle = `hsl(${(intensity * 360 + 180) % 360}, 100%, 50%)`;
        ctx.arc(x, y2, radius, 0, Math.PI * 2);
        ctx.fill();
    }
    
    // Add glow effect
    ctx.shadowBlur = 15;
    ctx.shadowColor = 'rgba(255, 255, 255, 0.5)';
}

function drawCircularWave(dataArray, bufferLength, moodColors) {
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const maxRadius = Math.min(centerX, centerY) - 50;
    const minRadius = maxRadius * 0.3;
    const average = dataArray.reduce((a, b) => a + b) / bufferLength;
    
    // Draw outer circle
    ctx.beginPath();
    ctx.arc(centerX, centerY, maxRadius, 0, Math.PI * 2);
    ctx.strokeStyle = moodColors[0];
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw circular wave
    ctx.beginPath();
    for (let i = 0; i < bufferLength; i++) {
        const value = dataArray[i];
        const angle = (i * Math.PI * 2) / bufferLength;
        const radiusOffset = (value / 255) * (maxRadius - minRadius) * 0.5;
        const radius = maxRadius - radiusOffset;
        
        const x = centerX + Math.cos(angle) * radius;
        const y = centerY + Math.sin(angle) * radius;
        
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }

        // Add particles on peaks
        if (value > 200 && Math.random() > 0.9) {
            const particleColor = moodColors[1].replace('#', '').match(/.{2}/g)
                .map(hex => parseInt(hex, 16)).join(', ');
            createParticles(x, y, particleColor);
        }
    }
    ctx.closePath();
    
    // Create gradient
    const gradient = ctx.createRadialGradient(
        centerX, centerY, minRadius,
        centerX, centerY, maxRadius
    );
    gradient.addColorStop(0, moodColors[0]);
    gradient.addColorStop(1, moodColors[1]);
    
    ctx.strokeStyle = gradient;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw center circle
    ctx.beginPath();
    ctx.arc(centerX, centerY, minRadius, 0, Math.PI * 2);
    ctx.fillStyle = moodColors[0];
    ctx.fill();

    // Add rotation effect
    ctx.save();
    ctx.translate(centerX, centerY);
    ctx.rotate(Date.now() * 0.001);
    ctx.translate(-centerX, -centerY);

    // Draw connecting lines
    for (let i = 0; i < bufferLength; i += 20) {
        const value = dataArray[i];
        if (value > 100) {
            const angle = (i * Math.PI * 2) / bufferLength;
            ctx.beginPath();
            ctx.moveTo(
                centerX + Math.cos(angle) * minRadius,
                centerY + Math.sin(angle) * minRadius
            );
            ctx.lineTo(
                centerX + Math.cos(angle) * maxRadius,
                centerY + Math.sin(angle) * maxRadius
            );
            ctx.strokeStyle = `rgba(${moodColors[1].replace('#', '').match(/.{2}/g)
                .map(hex => parseInt(hex, 16)).join(', ')}, 0.3)`;
            ctx.lineWidth = 1;
            ctx.stroke();
        }
    }
    ctx.restore();
}

function checkSecretCombinations() {
    if (secretCode.includes('disco')) {
        toggleDiscoMode();
        secretCode = '';
    } else if (secretCode.includes('matrix')) {
        toggleMatrixMode();
        secretCode = '';
    } else if (secretCode.includes('rainbow')) {
        toggleRainbowMode();
        secretCode = '';
    }
}

function toggleDiscoMode() {
    easterEggMode = !easterEggMode;
    colorMode = easterEggMode ? 'disco' : 'default';
    document.body.style.transition = 'all 0.3s';
    if (easterEggMode) {
        startDiscoColors();
    } else {
        stopDiscoColors();
    }
}

function toggleMatrixMode() {
    easterEggMode = !easterEggMode;
    colorMode = easterEggMode ? 'matrix' : 'default';
    if (easterEggMode) {
        startMatrixRain();
    } else {
        stopMatrixRain();
    }
}

function toggleRainbowMode() {
    easterEggMode = !easterEggMode;
    colorMode = easterEggMode ? 'rainbow' : 'default';
    if (easterEggMode) {
        startRainbowWave();
    } else {
        stopRainbowWave();
    }
}

let discoInterval;
function startDiscoColors() {
    const musicPlayer = document.querySelector('.music-player');
    discoInterval = setInterval(() => {
        const hue = Math.random() * 360;
        musicPlayer.style.background = `linear-gradient(45deg, 
            hsl(${hue}, 70%, 20%), 
            hsl(${(hue + 60) % 360}, 70%, 20%))`;
        document.body.style.backgroundColor = `hsl(${(hue + 180) % 360}, 50%, 10%)`;
    }, 500);
}

function stopDiscoColors() {
    clearInterval(discoInterval);
    const musicPlayer = document.querySelector('.music-player');
    musicPlayer.style.background = 'linear-gradient(to bottom, #282828, #181818)';
    document.body.style.backgroundColor = '#1a1a1a';
}

let matrixCanvas;
function startMatrixRain() {
    matrixCanvas = document.createElement('canvas');
    matrixCanvas.style.position = 'fixed';
    matrixCanvas.style.top = '0';
    matrixCanvas.style.left = '0';
    matrixCanvas.style.width = '100%';
    matrixCanvas.style.height = '100%';
    matrixCanvas.style.zIndex = '-1';
    matrixCanvas.style.opacity = '0.3';
    document.body.prepend(matrixCanvas);

    const context = matrixCanvas.getContext('2d');
    matrixCanvas.width = window.innerWidth;
    matrixCanvas.height = window.innerHeight;

    const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@#$%^&*()';
    const fontSize = 10;
    const columns = matrixCanvas.width / fontSize;
    const drops = Array(Math.floor(columns)).fill(1);

    function drawMatrix() {
        context.fillStyle = 'rgba(0, 0, 0, 0.05)';
        context.fillRect(0, 0, matrixCanvas.width, matrixCanvas.height);
        context.fillStyle = '#0F0';
        context.font = `${fontSize}px monospace`;

        for (let i = 0; i < drops.length; i++) {
            const text = characters[Math.floor(Math.random() * characters.length)];
            context.fillText(text, i * fontSize, drops[i] * fontSize);
            if (drops[i] * fontSize > matrixCanvas.height && Math.random() > 0.975) {
                drops[i] = 0;
            }
            drops[i]++;
        }
        matrixCanvas.matrixAnimation = requestAnimationFrame(drawMatrix);
    }
    drawMatrix();
}

function stopMatrixRain() {
    if (matrixCanvas) {
        cancelAnimationFrame(matrixCanvas.matrixAnimation);
        matrixCanvas.remove();
    }
}

let rainbowInterval;
function startRainbowWave() {
    let hue = 0;
    const visualizerCanvas = document.getElementById('visualizer');
    rainbowInterval = setInterval(() => {
        hue = (hue + 1) % 360;
        visualizerCanvas.style.boxShadow = `0 0 30px hsl(${hue}, 70%, 50%)`;
    }, 50);
}

function stopRainbowWave() {
    clearInterval(rainbowInterval);
    const visualizerCanvas = document.getElementById('visualizer');
    visualizerCanvas.style.boxShadow = '0 0 20px rgba(0, 0, 0, 0.5)';
}
