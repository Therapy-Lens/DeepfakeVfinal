import os
import sys
import json
import traceback

try:
    # STEP 1 — IMPORT SYSTEM PATH CORRECTLY
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BACKEND_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_ROOT, ".."))

    sys.path.append(BACKEND_ROOT)

    # STEP 2 — IMPORTS
    import torch
    import torchaudio.transforms as T
    import numpy as np
    import soundfile as sf
    
    print("Backend root:", BACKEND_ROOT)
    print("Python Path:", sys.path)
    print("Importing AASIST model...")
    from model.aasist import Model

    # STEP 3 — DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # STEP 4 — MODEL PATH (SAFE)
    MODEL_PATH = os.path.join(PROJECT_ROOT, "backend", "model", "best_aasist.pt")
    
    print("Model path:", MODEL_PATH)
    print("Exists:", os.path.exists(MODEL_PATH))

    # STEP 14 — VALIDATION CHECKS (IMPORTANT)
    if len(sys.argv) < 2:
        print(json.dumps({
            "prediction": "ERROR",
            "confidence": 0
        }))
        sys.exit(1)

    audio_path = sys.argv[1]
    
    print("Audio path:", audio_path)
    print("Exists:", os.path.exists(audio_path))

    if not os.path.exists(MODEL_PATH) or not os.path.exists(audio_path):
        print(json.dumps({
            "prediction": "ERROR",
            "confidence": 0
        }))
        sys.exit(1)

    # STEP 5 — LOAD MODEL (MATCH TRAINING EXACTLY)
    d_args = {
        "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64], [64, 64]],
        "first_conv": 1024,
        "gat_dims": [64, 32],
        "pool_ratios": [0.5, 0.7, 0.5],
        "temperatures": [2.0, 2.0, 100.0]
    }

    print("STEP 4: Model loading")
    model = Model(d_args)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    # STEP 6 — LOAD AUDIO
    print("STEP 1: Loading audio")
    waveform, sr = sf.read(audio_path)
    waveform = torch.from_numpy(waveform).float()

    # STEP 7 — PREPROCESS (MUST MATCH TRAINING)
    print("STEP 2: Preprocessing")
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=-1)

    waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-7)

    if sr != 16000:
        waveform = T.Resample(sr, 16000)(waveform)

    # STEP 8 — FIX LENGTH
    TARGET_LENGTH = 16000 * 8

    if waveform.shape[0] < TARGET_LENGTH:
        repeat_factor = TARGET_LENGTH // waveform.shape[0] + 1
        waveform = waveform.repeat(repeat_factor)

    waveform = waveform[:TARGET_LENGTH]

    # STEP 9 — LFCC (CRITICAL MATCH)
    print("STEP 3: LFCC")
    lfcc = T.LFCC(
        sample_rate=16000,
        n_lfcc=60,
        speckwargs={
            "n_fft": 512,
            "hop_length": 160,
            "win_length": 400
        }
    )

    feat = lfcc(waveform.unsqueeze(0))

    feat = torch.clamp(feat, -5, 5)
    feat = feat / (feat.std() + 1e-7)

    # STEP 10 — INFERENCE
    print("STEP 5: Inference")
    with torch.no_grad():
        _, logits = model(feat.to(DEVICE))
        probs = torch.softmax(logits, dim=1)

    # STEP 11 — OUTPUT DECISION
    fake_score = probs[0][1].item()

    if fake_score > 0.6:
        prediction = "FAKE"
    elif fake_score < 0.4:
        prediction = "REAL"
    else:
        prediction = "UNCERTAIN"

    # STEP 12 — PRINT JSON OUTPUT
    print(json.dumps({
        "prediction": prediction,
        "confidence": round(fake_score * 100, 2)
    }))

except Exception as e:
    # STEP 13 — ERROR HANDLING (MANDATORY)
    print("=== AUDIO ERROR DEBUG ===")
    print(str(e))
    traceback.print_exc()
    print(json.dumps({
        "prediction": "ERROR",
        "confidence": 0
    }))
