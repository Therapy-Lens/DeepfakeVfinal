import os
import torch
import torchaudio
import torchaudio.transforms as T
import random
import numpy as np
import soundfile as sf
import multiprocessing
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

# Add the project root to sys.path for local imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.models.aasist import Model

# =========================
# 🔒 GLOBAL STABILITY LOCK
# =========================
torch.backends.cudnn.benchmark = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# CONFIG
BASE_PATH = "ml/data" # Scanning root data dir correctly
BATCH_SIZE = 32 
LR = 1e-4
EPOCHS = 15
TARGET_LENGTH = 16000 * 8 # Still 8s for AASIST stability

# =========================
# 📂 DATASET (TRUE-LABEL SCANNER)
# =========================
# 1. Sources with absolute labels
FAKE_DIRS = ["OpenAI", "VALLE", "VoiceBox", "FlashSpeech", "PromptTTS2", "NaturalSpeech3", "seedtts_files", "xTTS"]
REAL_DIRS = ["real_samples", "UK", "USA"]
# 2. Mixed sources (labeled by 'real'/'fake' folder name)
MIXED_DIRS = ["audio", "for-2sec", "for-norm", "for-original", "for-rerec", "HAV-DF"]

def collect_verified_data():
    samples = []
    final_path = os.path.join(BASE_PATH, "final_audio")
    
    # 🌟 Priority: Use standardized dataset if sorting is complete
    if os.path.exists(final_path):
        print(f"✅ Scanning standardized dataset: {final_path}")
        for label_name in ["real", "fake"]:
            folder = os.path.join(final_path, label_name)
            if not os.path.exists(folder): continue
            label = 0 if label_name == "real" else 1
            for f in os.listdir(folder):
                if f.lower().endswith((".wav", ".mp3", ".flac")):
                    samples.append((os.path.join(folder, f), label))
        return samples

    print(f"Deep scanning {BASE_PATH} for forensic labeling...")
    # Track counts for balancing confirmation
    counts = {0: 0, 1: 0}

    for root, _, files in os.walk(BASE_PATH):
        # Determine the source folder (first directory after 'ml/data')
        rel_path = os.path.relpath(root, BASE_PATH)
        top_folder = rel_path.split(os.sep)[0]
        
        for f in files:
            if f.lower().endswith((".wav", ".mp3", ".flac")):
                path = os.path.join(root, f)
                label = -1
                
                # Check absolute fake sources
                if top_folder in FAKE_DIRS:
                    label = 1
                # Check absolute real sources
                elif top_folder in REAL_DIRS:
                    label = 0
                # Check mixed sources
                else:
                    if "fake" in root.lower() or "synthesized" in root.lower() or "spoof" in root.lower():
                        label = 1
                    elif "real" in root.lower() or "original" in root.lower() or "bonafide" in root.lower():
                        label = 0
                    else: continue # Skip ambiguous samples

                samples.append((path, label))
                counts[label] += 1
                
    print(f"✅ Scanning complete. Found: {counts[0]} REAL | {counts[1]} FAKE (Total: {len(samples)})")
    return samples

class AudioDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.sample_rate = 16000

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        try:
            waveform, sr = sf.read(path)
            waveform = torch.from_numpy(waveform).float()
            
            # Normalize and Mono
            if waveform.ndim > 1: waveform = waveform.mean(dim=-1)
            max_val = torch.max(torch.abs(waveform))
            if max_val > 1e-7: waveform /= max_val
            
            # Resample if needed
            if sr != self.sample_rate:
                waveform = T.Resample(sr, self.sample_rate)(waveform)
            
            # 🔥 Noise injection (reduces overfitting)
            if random.random() < 0.2:
                waveform = waveform + 0.003 * torch.randn_like(waveform)
            
            # Clamp to prevent numerical explosion
            waveform = torch.clamp(waveform, -1.0, 1.0)

            # Enforce AASIST-safe length
            if waveform.size(0) < TARGET_LENGTH:
                repeat = (TARGET_LENGTH // waveform.size(0)) + 1
                waveform = waveform.repeat(repeat)[:TARGET_LENGTH]
            else:
                waveform = waveform[:TARGET_LENGTH]
                
            return waveform, torch.tensor(label, dtype=torch.float32)
        except:
            return self.__getitem__((idx + 1) % len(self.data))

# -------------------------
# 🚀 MAIN TRAINING
# -------------------------
def main():
    print(f"🚀 FINAL ZERO-LABEL-ERROR AASIST TRAINING | Device: {DEVICE}")

    # 1. Load Data with corrected forensic labels
    data = collect_verified_data()
    if not data:
        print("❌ No data found."); return
    random.shuffle(data)

    # 2. Balanced Sampling
    labels = [d[1] for d in data]
    class_counts = np.bincount(labels)
    # Give weight based on class frequency to ensure 50/50 batches
    weights = 1. / (class_counts + 1e-6)
    sample_weights = torch.tensor([weights[l] for l in labels])
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # 3. Dataloader (🔒 ZERO-CRASH MODE)
    train_loader = DataLoader(
        AudioDataset(data),
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )

    # 4. Model Setup
    d_args = {"filts": [70, [1, 32], [32, 32], [32, 64], [64, 64], [64, 64]], "first_conv": 1024, "gat_dims": [64, 32], "pool_ratios": [0.5, 0.7, 0.5], "temperatures": [2.0, 2.0, 100.0]}
    model = Model(d_args).to(DEVICE)
    
    # Penalize FAKE mismatch significantly to prioritize security
    loss_weights = torch.tensor([1.0, 3.0]).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=0.1)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scaler = torch.amp.GradScaler('cuda')
    lfcc = T.LFCC(sample_rate=16000, n_lfcc=60, speckwargs={"n_fft": 512, "hop_length": 160, "win_length": 400}).to(DEVICE)

    best_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for waveforms, y in pbar:
            waveforms, y = waveforms.to(DEVICE, non_blocking=True), y.long().to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                feat = lfcc(waveforms)
                feat = torch.clamp(feat, -5, 5)
                feat = feat / (feat.std() + 1e-7)
                
                _, logits = model(feat)
                logits = torch.clamp(logits, -10, 10)
                loss = criterion(logits, y)
                
                # Confidence Penalty
                probs = torch.softmax(logits, dim=1)
                penalty = torch.mean(torch.abs(probs[:, 1] - 0.5))
                loss = loss + 0.05 * penalty

            if torch.isnan(loss) or torch.isinf(loss): continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"🔥 Epoch {epoch+1} Completed | Avg Loss: {avg_loss:.4f}")

        # Checkpoint
        os.makedirs("ml/models", exist_ok=True)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "ml/models/best_aasist.pt")
            print(f"⭐ New Best Model Saved!")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
