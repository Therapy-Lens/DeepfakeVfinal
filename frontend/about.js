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
