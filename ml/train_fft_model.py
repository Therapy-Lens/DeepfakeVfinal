import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from functools import lru_cache
from tqdm import tqdm
import random
from collections import defaultdict

# =========================
# CONFIGURATION
# =========================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed_videos')
REAL_DIR = os.path.join(PROCESSED_DIR, 'real')
FAKE_DIR = os.path.join(PROCESSED_DIR, 'fake')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, '..', 'backend', 'model', 'fft_model.pth')

@lru_cache(maxsize=1024)
def load_tensor(file_path):
    return torch.load(file_path, map_location='cpu', weights_only=True)

# =========================
# FFT TRANSFORM
# =========================
def compute_fft(image_tensor):
    gray = image_tensor.mean(dim=0).numpy()
    
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    
    magnitude = np.log(np.abs(fshift) + 1)
    
    mag_min = magnitude.min()
    mag_max = magnitude.max()
    if mag_max > mag_min:
        magnitude = (magnitude - mag_min) / (mag_max - mag_min)
    else:
        magnitude = np.zeros_like(magnitude)
        
    magnitude_3c = np.stack([magnitude, magnitude, magnitude], axis=0)
    return torch.tensor(magnitude_3c, dtype=torch.float32)

# =========================
# AUGMENTATIONS
# =========================
def apply_augmentations(frames):
    # frames shape: [4, 3, 224, 224] natively loaded spatial tensors
    if random.random() > 0.5:
        frames = TF.hflip(frames)
        
    jitter = T.ColorJitter(brightness=0.1)
    frames = jitter(frames)
    
    # Tiny Gaussian perturbation
    noise = torch.randn_like(frames) * 0.05
    frames = frames + noise
    
    return frames

# =========================
# DATASET
# =========================
class FFTVideoDataset(Dataset):
    def __init__(self, video_list, is_train=False):
        self.video_list = video_list
        self.is_train = is_train

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        file_path, label = self.video_list[idx]
        tensor_batch = load_tensor(file_path)
        
        num_frames = tensor_batch.size(0)
        target_frames = 4
        
        selected_indices = []
        if num_frames <= target_frames:
            selected_indices = list(range(num_frames))
            while len(selected_indices) < target_frames:
                selected_indices.append(random.choice(selected_indices))
            selected_indices.sort()
        else:
            chunk_size = num_frames // target_frames
            for i in range(target_frames):
                start = i * chunk_size
                end = start + chunk_size - 1
                if i == target_frames - 1:
                    end = num_frames - 1 # guarantee coverage of tail frame bound
                selected_indices.append(random.randint(start, end)) # Sub-chunk Jitter Mapping!
                
        frames = tensor_batch[selected_indices]
        
        if self.is_train:
            frames = apply_augmentations(frames)
            
        fft_frames = torch.stack([compute_fft(f) for f in frames])
        return fft_frames, label

# =========================
# PREPARATION & GROUP SPLIT (LEAKAGE PREVENTION)
# =========================
def main():
    print(f"Using Device: {DEVICE}")
    print("Scanning datasets and extracting group IDs...")

    video_list = []
    
    if os.path.exists(REAL_DIR):
        real_files = [os.path.join(REAL_DIR, f) for f in os.listdir(REAL_DIR) if f.endswith('.pt')]
        for file in real_files:
            video_list.append((file, 0))

    if os.path.exists(FAKE_DIR):
        fake_files = [os.path.join(FAKE_DIR, f) for f in os.listdir(FAKE_DIR) if f.endswith('.pt')]
        for file in fake_files:
            video_list.append((file, 1))

    total_videos = len(video_list)
    if total_videos == 0:
        print("❌ CRITICAL: No processed videos found!")
        return

    # 🚨 STRICT GROUP-BASED SPLITTING 
    # Prevents base videos propagating into both Train and Validation organically!
    groups = defaultdict(list)
    for file_path, label in video_list:
        basename = os.path.basename(file_path).replace('.pt', '')
        group_id = basename.split('_')[-1] # Grabs explicit video sequence root ID safely
        groups[group_id].append((file_path, label))
        
    group_ids = list(groups.keys())
    random.seed(42)
    random.shuffle(group_ids)
    
    split_idx = int(0.8 * len(group_ids))
    train_groups = group_ids[:split_idx]
    val_groups = group_ids[split_idx:]
    
    train_videos = []
    for gid in train_groups:
        train_videos.extend(groups[gid])
        
    val_videos = []
    for gid in val_groups:
        val_videos.extend(groups[gid])
        
    random.shuffle(train_videos)
    random.shuffle(val_videos)

    print(f"\n[DEBUG] Total videos loaded: {total_videos}")
    print(f"[DEBUG] Total unique ID Groups defined: {len(group_ids)}")
    print(f"[DEBUG] Train groups separated: {len(train_groups)}")
    print(f"[DEBUG] Validation groups separated: {len(val_groups)}")
    print(f"[DEBUG] Total train samples distribution: {len(train_videos)}")
    print(f"[DEBUG] Total validation samples distribution: {len(val_videos)}\n")

    train_dataset = FFTVideoDataset(train_videos, is_train=True)
    val_dataset = FFTVideoDataset(val_videos, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # =========================
    # MODEL ARCHITECTURE
    # =========================
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, 2)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == 'cuda')

    best_val_loss = float('inf')
    early_stop_counter = 0
    PATIENCE = 3

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # =========================
    # TRAINING LOOP
    # =========================
    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- EPOCH {epoch}/{EPOCHS} ---")
        
        # TRAIN
        model.train()
        train_loss = 0.0
        
        pbar_train = tqdm(train_loader, desc=f"Training")
        for inputs, labels in pbar_train:
            # Inputs shape natively yields [B, 4, 3, 224, 224]
            B, num_frames, C, H, W = inputs.shape
            
            # Sub-flattening strictly collapsing spatial batches into uniform flat arrays
            inputs = inputs.view(B * num_frames, C, H, W).to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=DEVICE.type, enabled=DEVICE.type == 'cuda'):
                outputs = model(inputs) # Evaluates uniquely outputting [B*4, 2]
                outputs = outputs.view(B, num_frames, 2).mean(dim=1) # Condenses temporally yielding [B, 2] predictions!
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * B
            pbar_train.set_postfix({'loss': loss.item()})
            
        avg_train_loss = train_loss / len(train_dataset)
        scheduler.step()
        
        # VALIDATE
        model.eval()
        val_loss = 0.0
        correct = 0
        
        pbar_val = tqdm(val_loader, desc=f"Validating")
        with torch.no_grad():
            for inputs, labels in pbar_val:
                B, num_frames, C, H, W = inputs.shape
                inputs = inputs.view(B * num_frames, C, H, W).to(DEVICE)
                labels = labels.to(DEVICE)
                
                with torch.amp.autocast(device_type=DEVICE.type, enabled=DEVICE.type == 'cuda'):
                    outputs = model(inputs)
                    outputs = outputs.view(B, num_frames, 2).mean(dim=1)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item() * B
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels)
                
        avg_val_loss = val_loss / len(val_dataset)
        val_acc = correct.double() / len(val_dataset)
        
        print(f"-> Train Loss: {avg_train_loss:.4f}")
        print(f"-> Val Loss:   {avg_val_loss:.4f}")
        print(f"-> Val Acc:    {(val_acc * 100):.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"🌟 Best model saved!")
        else:
            early_stop_counter += 1
            if early_stop_counter >= PATIENCE:
                print(f"🛑 Early stopping triggered after {PATIENCE} epochs without loss improvement.")
                break

    print("\n✅ Training Complete!")

if __name__ == "__main__":
    main()
