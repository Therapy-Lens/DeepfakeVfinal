/* Background Canvas Animation */
const canvas = document.getElementById('bg-canvas');
const ctx = canvas.getContext('2d');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

window.addEventListener('resize', () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
});

const particles = [];
const particleCount = 60;

for (let i = 0; i < particleCount; i++) {
    particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        size: Math.random() * 4 + 4,
        speed: Math.random() * 0.15 + 0.05,
        opacity: Math.random() * 0.4 + 0.2,
        opacityDir: Math.random() > 0.5 ? 1 : -1
    });
}

function animateParticles() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (let p of particles) {
        ctx.fillStyle = `rgba(146, 188, 234, ${p.opacity})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fill();

        p.y -= p.speed;
        p.x += Math.sin(p.y * 0.01) * 0.2;
        
        if (p.y < -10) {
            p.y = canvas.height + 10;
            p.x = Math.random() * canvas.width;
        }

        p.opacity += p.opacityDir * 0.001;
        if (p.opacity >= 0.7 || p.opacity <= 0.15) {
            p.opacityDir *= -1;
        }
    }
    requestAnimationFrame(animateParticles);
}
animateParticles();

/* File Handling */
const uploadBox = document.getElementById('upload-box');
const fileInput = document.getElementById('file-input');
const previewGrid = document.getElementById('preview-grid');
const videoPreview = document.getElementById('video-preview');
const resultSection = document.getElementById('result-section');
const resultText = document.getElementById('result-text');
const confidenceBar = document.getElementById('confidence-bar');

uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('dragover');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('dragover');
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
    let files = Array.from(e.dataTransfer.files);
    processFiles(files);
});

fileInput.addEventListener('change', (e) => {
    let files = Array.from(e.target.files);
    processFiles(files);
});

async function processFiles(files) {
    let videoFile = files.find(f => f.type.startsWith('video/'));
    let validFiles = [];

    if (videoFile) {
        validFiles = [videoFile];
    } else {
        validFiles = files.filter(f => f.type.startsWith('image/'));
    }

    if (validFiles.length > 0) {
        showPreviews(validFiles);
        
        // Disable uploading while processing
        fileInput.disabled = true;
        const uploadBtn = document.querySelector('.upload-btn');
        if(uploadBtn) uploadBtn.disabled = true;
        
        if (typeof sendToBackend === 'function') {
            resultSection.style.display = 'block';
            confidenceBar.style.width = '0%';
            resultText.textContent = 'Uploading...';
            resultText.style.color = 'white';
            
            try {
                const data = await sendToBackend(validFiles);
                showResult(data);
            } catch (err) {
                console.error("Backend Error:", err);
                resultText.textContent = 'UPLOAD FAILED';
                resultText.style.color = '#FF4C4C';
            } finally {
                fileInput.disabled = false;
                if(uploadBtn) uploadBtn.disabled = false;
            }
        } else {
            showResult();
            fileInput.disabled = false;
            if(uploadBtn) uploadBtn.disabled = false;
        }
    }
}

async function sendToBackend(files) {
    const formData = new FormData();
    files.forEach(file => {
        formData.append("files", file);
    });

    const response = await fetch("http://localhost:5000/upload", {
        method: "POST",
        body: formData
    });

    if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();
    console.log("API response:", data);
    return data;
}

function showPreviews(files) {
    previewGrid.innerHTML = '';
    videoPreview.innerHTML = '';
    previewGrid.style.display = 'none';
    videoPreview.style.display = 'none';
    resultSection.style.display = 'none';

    for (let file of files) {
        if (file.type.startsWith('image/')) {
            const img = document.createElement('img');
            img.src = URL.createObjectURL(file);
            previewGrid.appendChild(img);
            previewGrid.style.display = 'grid';
        } else if (file.type.startsWith('video/')) {
            const vid = document.createElement('video');
            vid.src = URL.createObjectURL(file);
            vid.controls = true;
            videoPreview.appendChild(vid);
            videoPreview.style.display = 'block';
        }
    }
}

function showResult(data) {
    resultSection.style.display = 'block';
    
    if (data && data.prediction) {
        if (data.prediction.includes('REAL')) {
            resultText.textContent = `${data.prediction} (${data.confidence}%)`;
            resultText.style.color = '#4CAF50';
            confidenceBar.style.width = `${data.confidence}%`;
        } else if (data.prediction.includes('FAKE')) {
            resultText.textContent = `${data.prediction} (${data.confidence}%)`;
            resultText.style.color = '#FF4C4C';
            confidenceBar.style.width = `${data.confidence}%`;
        } else {
            resultText.textContent = `UNCERTAIN (${data.confidence}%)`;
            resultText.style.color = '#FFC107'; // Yellow
            confidenceBar.style.width = `${data.confidence}%`;
        }
    } else {
        confidenceBar.style.width = '0%';
        resultText.textContent = 'Processing...';
        resultText.style.color = 'white';
        setTimeout(() => {
            const isReal = Math.random() > 0.5;
            if (isReal) {
                resultText.textContent = 'REAL (82%)';
                resultText.style.color = '#4CAF50';
                confidenceBar.style.width = '82%';
            } else {
                resultText.textContent = 'FAKE (67%)';
                resultText.style.color = '#FF4C4C';
                confidenceBar.style.width = '67%';
            }
        }, 1000);
    }
}
