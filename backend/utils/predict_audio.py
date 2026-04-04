import sys
import os
import json
import torch
import torch.nn as nn
import librosa
import numpy as np
import cv2
from torchvision import models

# --- GLOBAL CONFIGURATION (Matched with Training) ---
SR = 16000
N_MELS = 128
TARGET_SHAPE = (128, 128)

def get_model(model_path, device):
    """Reconstruct ResNet18 architecture and load trained weights."""
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    # Load Weights
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def preprocess_audio(file_path):
    """Load and transform raw audio into Mel Spectrogram tensor matching training logic."""
    # 1. Load Audio (16kHz, Mono)
    y, sr = librosa.load(file_path, sr=SR, mono=True)
    
    # Ensure minimum length (pad if too short)
    if len(y) < SR:
        y = np.pad(y, (0, SR - len(y)))

    # 2. Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    
    # 3. Log Scale (dB)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 4. Resize (128, 128)
    # OpenCV expects (width, height), and librosa output is (n_mels, time)
    resized_mel = cv2.resize(mel_db, TARGET_SHAPE, interpolation=cv2.INTER_AREA)
    
    # 5. Min-Max Normalization to [0, 1]
    mel_min = resized_mel.min()
    mel_max = resized_mel.max()
    if mel_max > mel_min:
        normalized_mel = (resized_mel - mel_min) / (mel_max - mel_min)
    else:
        normalized_mel = resized_mel - mel_min
        
    # 6. Convert to Tensor [1, 1, 128, 128]
    tensor = torch.FloatTensor(normalized_mel).unsqueeze(0).unsqueeze(0)
    return tensor

def predict_audio(audio_path, model, device):
    """Predict label (REAL/FAKE) for a given audio file."""
    try:
        input_tensor = preprocess_audio(audio_path).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()
        
        label = "REAL" if pred_idx == 0 else "FAKE"
        
        return {
            "prediction": label,
            "confidence": float(round(confidence * 100, 2))
        }
    except Exception as e:
        return {
            "error": str(e)
        }

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "No file path provided"}))
        return

    audio_path = sys.argv[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Locate model relative to script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(os.path.join(base_dir, "..", "model", "audio_model.pt"))
    
    if not os.path.exists(model_path):
        print(json.dumps({"success": False, "error": f"Model weights not found at {model_path}"}))
        return

    try:
        # Load model (Global load simulation for one-shot execution)
        model = get_model(model_path, device)
        
        # Inference
        result = predict_audio(audio_path, model, device)
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({"success": False, "error": f"Internal Error: {str(e)}"}))

if __name__ == "__main__":
    main()
