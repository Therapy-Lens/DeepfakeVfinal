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
├── .git/                      # Git repository metadata
├── .gitignore                 # Tracking exclusions
├── README.md                  # Project documentation
│
├── frontend/                  # Client-side UI (Vanilla HTML/JS/CSS)
│   ├── about.css
│   ├── about.html
│   ├── about.js
│   ├── home.css
│   ├── home.html
│   ├── home.js
│   ├── link.js
│   ├── tech.css
│   ├── tech.html
│   └── tech.js
│
├── backend/                   # Node.js Express Server
│   ├── .uploads/              # Active hidden temporary upload buffer
│   ├── model/                 # Production Model Weights
│   │   ├── __pycache__/
│   │   ├── audio_model.pt
│   │   ├── best_video_model.pt
│   │   └── image_model.pth
│   ├── node_modules/          # Node dependencies (express, multer, cors, ...) [Total: 87 folders]
│   ├── routes/
│   │   └── upload.js          # Master Upload & Script Routing Logic
│   ├── uploads/               # Legacy upload folder (Unused)
│   ├── utils/                 # Python Inference Bridges
│   │   ├── __pycache__/
│   │   ├── predict.py         # Image detection engine
│   │   ├── predict_audio.py   # Audio detection engine
│   │   └── predict_video.py   # Video detection engine
│   ├── package-lock.json
│   ├── package.json
│   └── server.js              # Backend Entrypoint
│
├── ml/                        # Machine Learning Operations (Training & Preprocessing)
│   ├── __pycache__/           # Compiled Python bytecode
│   ├── data/                  # Dataset Management
│   │   ├── audio/             # Raw audio samples
│   │   ├── images/            # Raw image samples
│   │   ├── processed_audio/   # Optimized audio tensors
│   │   ├── processed_videos/  # Extracted frame sequences
│   │   │   ├── fake/          # (Deepfakes_012_026.pt, ...) [Total: 992 files]
│   │   │   └── real/          # (faceforencics_000.pt, ...) [Total: 998 files]
│   │   └── videos/            # Raw video datasets (FaceForensics, Celeb-DF)
│   ├── preprocess_audios.py
│   ├── preprocess_videos.py
│   ├── requirements.txt       # Python dependency manifest
│   ├── train_audio_model.py
│   ├── train_fft_model.py     # Frequency-domain analysis training
│   ├── train_image_model.py
│   ├── train_image_model2.py
│   ├── train_video_model.py
│   └── train_video_model2.py
│
└── venv/                      # Python Virtual Environment (Isolated dependencies)
    ├── Include/
    ├── Lib/                   # Installed packages (torch, torchvision, etc.) [Total: thousands of files]
    ├── Scripts/               # Executables (python.exe, pip.exe)
    ├── share/
    └── pyvenv.cfg
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
