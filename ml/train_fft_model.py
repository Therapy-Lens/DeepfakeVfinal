import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from torch.optim import Adam
from functools import lru_cache
from tqdm import tqdm
import random

# =========================
# CONFIGURATION
# =========================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed_videos')
REAL_DIR = os.path.join(PROCESSED_DIR, 'real')
FAKE_DIR = os.path.join(PROCESSED_DIR, 'fake')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, '..', 'backend', 'model', 'fft_model.pth')

@lru_cache(maxsize=256)
def load_tensor(file_path):
    """LRU Cache avoids thrashing the disk when fetching individual frames out of order."""
    return torch.load(file_path, map_location='cpu', weights_only=True)

# =========================
# FFT TRANSFORM
# =========================
def compute_fft(image_tensor):
    """
    Transforms a [3, 224, 224] spatial frame into a [3, 224, 224] Frequency Magnitude map.
    """
    # 1. Convert to grayscale via mean across color channels
    gray = image_tensor.mean(dim=0).numpy()
    
    # 2. Apply FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    
    # 3. Magnitude
    magnitude = np.log(np.abs(fshift) + 1)
    
    # 4. Normalize to [0,1]
    mag_min = magnitude.min()
    mag_max = magnitude.max()
    if mag_max > mag_min:
        magnitude = (magnitude - mag_min) / (mag_max - mag_min)
    else:
        magnitude = np.zeros_like(magnitude)
        
    # 5. Convert to 3-channel
    magnitude_3c = np.stack([magnitude, magnitude, magnitude], axis=0)
    
    return torch.tensor(magnitude_3c, dtype=torch.float32)

# =========================
# DATASET
# =========================
class FFTVideoDataset(Dataset):
    def __init__(self, video_list):
        # video_list is a list of tuples containing (file_path, label)
        self.video_list = video_list

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        file_path, label = self.video_list[idx]
        
        # Load sequence tensor [NUM_FRAMES, 3, 224, 224]
        tensor_batch = load_tensor(file_path)
        
        # Randomly select ONE frame from that video
        num_frames = tensor_batch.size(0)
        random_frame_idx = random.randint(0, num_frames - 1)
        frame = tensor_batch[random_frame_idx]
        
        # Apply FFT Transform natively
        fft_frame = compute_fft(frame)
        
        return fft_frame, label

# =========================
# PREPARATION & SPLIT
# =========================
def main():
    print(f"Using Device: {DEVICE}")
    print("Scanning datasets...")

    video_list = []
    
    # Load Real
    if os.path.exists(REAL_DIR):
        real_files = [os.path.join(REAL_DIR, f) for f in os.listdir(REAL_DIR) if f.endswith('.pt')]
        for file in real_files:
            video_list.append((file, 0)) # 0 = REAL

    # Load Fake
    if os.path.exists(FAKE_DIR):
        fake_files = [os.path.join(FAKE_DIR, f) for f in os.listdir(FAKE_DIR) if f.endswith('.pt')]
        for file in fake_files:
            video_list.append((file, 1)) # 1 = FAKE

    total_videos = len(video_list)
    print(f"\n[DEBUG] Total videos loaded: {total_videos}")

    if total_videos == 0:
        print("❌ CRITICAL: No processed videos found!")
        return

    # Random split stratified purely by video
    real_videos = [v for v in video_list if v[1] == 0]
    fake_videos = [v for v in video_list if v[1] == 1]
    
    random.seed(42)
    random.shuffle(real_videos)
    random.shuffle(fake_videos)
    
    real_split = int(0.8 * len(real_videos))
    fake_split = int(0.8 * len(fake_videos))
    
    train_videos = real_videos[:real_split] + fake_videos[:fake_split]
    val_videos = real_videos[real_split:] + fake_videos[fake_split:]
    
    random.shuffle(train_videos)
    random.shuffle(val_videos)

    print(f"[DEBUG] Train videos count: {len(train_videos)}")
    print(f"[DEBUG] Validation videos count: {len(val_videos)}")
    print(f"[DEBUG] Total samples per epoch: {len(train_videos)}\n")

    train_dataset = FFTVideoDataset(train_videos)
    val_dataset = FFTVideoDataset(val_videos)

    # Minimal workers to prevent disk access collisions with LRU Cache threads natively
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # =========================
    # MODEL ARCHITECTURE
    # =========================
    model = resnet18(weights=None) # Start fresh or define weights natively
    model.fc = nn.Linear(512, 2)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # =========================
    # TRAINING LOOP
    # =========================
    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- EPOCH {epoch}/{EPOCHS} ---")
        
        # TRAIN
        model.train()
        train_loss = 0.0
        
        pbar_train = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        for inputs, labels in pbar_train:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            pbar_train.set_postfix({'loss': loss.item()})
            
        avg_train_loss = train_loss / len(train_dataset)
        
        # VALIDATE
        model.eval()
        val_loss = 0.0
        correct = 0
        
        pbar_val = tqdm(val_loader, desc=f"Validating Epoch {epoch}")
        with torch.no_grad():
            for inputs, labels in pbar_val:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data)
                
        avg_val_loss = val_loss / len(val_dataset)
        val_acc = correct.double() / len(val_dataset)
        
        print(f"\n[DEBUG] Epoch {epoch} Results:")
        print(f"-> Train Loss: {avg_train_loss:.4f}")
        print(f"-> Val Loss:   {avg_val_loss:.4f}")
        print(f"-> Val Acc:    {(val_acc * 100):.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"🌟 Best model saved to: {MODEL_SAVE_PATH}")

    print("\n✅ Training Complete!")

if __name__ == "__main__":
    main()
