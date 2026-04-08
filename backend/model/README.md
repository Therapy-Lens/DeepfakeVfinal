# Production Model Weights

This directory contains the compiled neural network weights used by the inference engines.

### 📂 Contents
- **`image_model.pth`**: EfficientNet-B4 weights for static image classification.
- **`best_video_model.pt`**: ResNeXt50 + LSTM hybrid weights for temporal video analysis.
- **`audio_model.pt`**: AASIST/ResNet18 weights for synthetic voice and audio deepfake detection.

### 🧠 Description
These files represent the "brain" of the platform. They are the result of extensive training on the datasets mentioned in the `ml/data` directory. The scripts in `backend/utils/` load these weights at runtime to perform real-world predictions.

### ⚠️ Note on Git
Model weight files (`.pt`, `.pth`) are binary files that can exceed 100MB+ each. They are tracked via Git LFS or excluded from Git entirely using `.gitignore` to prevent repository bloat.
