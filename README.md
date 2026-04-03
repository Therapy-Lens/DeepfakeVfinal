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
│   ├── about.html            # About page interface
│   ├── about.js              # About page logic
│   ├── about.css             # About page styling
│   ├── home.html             # Main dashboard interface
│   ├── home.js               # Client-side API mapping logic
│   ├── home.css              # Main dashboard styling
│   ├── tech.html             # Technology details interface
│   ├── tech.js               # Technology details logic
│   ├── tech.css              # Technology details styling
│   └── link.js               # Cross-page navigation utilities
│
├── backend/                  # Node.js Express Server
│   ├── server.js             # Entrypoint server instance
│   ├── package.json          # Node dependencies (Express, Cors, Multer)
│   ├── routes/               # API routes
│   │   └── upload.js         # Video & Image processing API router
│   ├── utils/                # PyTorch Inference Bridges
│   │   ├── predict.py        # Single image detection pipeline
│   │   └── predict_video.py  # Video classification pipeline
│   ├── model/                # Compiled .pt & .pth model weights
│   │   ├── best_video_model.pt
│   │   └── image_model.pth
│   └── uploads/              # Local temporary upload buffers
│
└── ml/                       # Machine Learning Engineering
    ├── data/                 # Raw/Processed dataset storage
    ├── preprocess_videos.py  # Dataset transformation operations
    ├── train_image_model.py  # Image classifier early backbone training
    ├── train_image_model2.py # Image classifier fine-tuning pipeline
    ├── train_video_model.py  # Hybrid LSTM feature training
    ├── train_video_model2.py # High-performance hybrid video model training
    └── requirements.txt      # Python dependencies
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
