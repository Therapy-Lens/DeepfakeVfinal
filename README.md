# Deepfake Detection Platform

A full-stack, modular ecosystem designed for robust authentication of multimedia content. The platform integrates localized image and dynamic video inference architectures leveraging bleeding-edge deep-learning networks.

### 🛡️ Team: Therapy Lens

- **Anshul Khandar**
- **Aditya Ambare**
- **Rohit Hajare**
- **Manasvi Yeole**

---

### 📂 Repository Structure

```text
Deepfake-Detection/
│
├── frontend/                 # Client UI
│   ├── home.html             # Main dashboard
│   ├── home.js               # Client-side API mapping logic
│   ├── index.css             # Glassmorphism/CSS utility classes
│   └── about.js
│
├── backend/                  # Node.js Express Server
│   ├── server.js             # Entrypoint server instance
│   ├── routes/               # API route definitions (e.g., /upload)
│   ├── utils/                # PyTorch Inference Bridges
│   │   ├── predict.py        # Single image detection pipeline
│   │   └── predict_video.py  # Video classification pipeline
│   ├── model/                # Compiled .pt & .pth model weights
│   └── uploads/              # Local temporary upload buffers
│
└── ml/                       # Machine Learning Engineering
    ├── data/                 # Raw/Processed dataset storage
    ├── train_image_model.py  # Image classifier backbone training
    ├── train_video_model.py  # Hybrid LSTM feature training
    └── preprocess_videos.py  # Dataset transformation operations
```

---

### 🧠 Core Neural Architectures & Utilities

Our inference pipelines are configured leveraging specialized neural graphs engineered to cross-correlate distinct visual artifacts:

1.  **EfficientNet-B4:** Employed specifically regarding standalone Image classification natively mapping facial distortions dynamically.
2.  **ResNeXt50 + LSTM (Hybrid):** Engineered specifically iterating dynamic sequence behaviors traversing spatial dimensions and sequentially tracking temporals via sequence recurrent endpoints.
3.  **AASIST:** Dedicated Anti-Spoofing topological frameworks designed natively for robust identification integrations.
4.  **OpenCV / MTCNN / PyTorch Tools:** Embedded specifically for real-time localized facial bounding and image sequence extractions.

---

### 📊 Dataset Information

Trained precisely parsing comprehensive real-world benchmarks ensuring extreme threshold viability targeting compression degradation:

- **FaceForensics++:** Highly robust benchmark containing pristine variants and four different artificial manipulation permutations (Deepfakes, Face2Face, FaceSwap, NeuralTextures).
- **Celeb-DF:** Extensive large-scale deepfake video dataset providing distinctly refined generation behaviors simulating highly challenging real-world visual paradigms.
