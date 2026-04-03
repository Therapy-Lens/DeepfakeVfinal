import torch
import cv2
import numpy as np
import sys
import json
from PIL import Image
import torchvision.transforms as transforms
import timm

import os

# CONFIG
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "best_video_model.pt")
MODEL_PATH = os.path.abspath(MODEL_PATH)

print(f"[DEBUG] Script Dir: {BASE_DIR}")
print(f"[DEBUG] Model Path: {MODEL_PATH}")
print(f"[DEBUG] Exists: {os.path.exists(MODEL_PATH)}")

NUM_FRAMES = 8

import torch.nn as nn
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights

class ResNeXtLSTM(nn.Module):
    def __init__(self, hidden_size=512):
        super(ResNeXtLSTM, self).__init__()
        resnext = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnext.children())[:-1])
        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)
        
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.feature_extractor(x)
        features = features.view(batch_size, seq_len, 2048)
        lstm_out, _ = self.lstm(features)
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)

# Load model
try:
    model = ResNeXtLSTM()
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print("[DEBUG] Model loaded successfully with state_dict", file=sys.stderr)
except Exception as e:
    print(f"Error loading model: {e}", file=sys.stderr)
    sys.exit(1)

# Standard Transformation matching Training Backbone expectations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return None

    indices = np.linspace(0, total_frames - 1, NUM_FRAMES).astype(int)

    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        parsed_tensor = transform(pil_img)
        frames.append(parsed_tensor)

    cap.release()

    if len(frames) == 0:
        return None

    while len(frames) < NUM_FRAMES:
        frames.append(frames[-1].clone())

    frames = frames[:NUM_FRAMES]

    return torch.stack(frames)

def predict(video_path):
    tensor = extract_frames(video_path)

    if tensor is None:
        return {"prediction": "UNCERTAIN", "confidence": 0}

    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    prob_fake = probs[1].item()
    prob_real = probs[0].item()

    confidence = max(prob_fake, prob_real) * 100

    if prob_real > 0.8:
        label = "REAL"
    elif prob_fake > 0.8:
        label = "FAKE"
    else:
        label = "UNCERTAIN"

    return {"prediction": label, "confidence": round(confidence, 2)}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        result = predict(video_path)
        # Using json.dumps to ensure output is valid JSON (double quotes)
        print(json.dumps(result))
    else:
        print(json.dumps({"prediction": "UNCERTAIN", "confidence": 0}))
