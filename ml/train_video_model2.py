import os
import cv2
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
import torchvision.transforms as transforms
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading
import time
import psutil
import numpy as np

# Performance tuning
cv2.setNumThreads(0)

# 1. Dataset Class for Raw Videos
class RawVideoDataset(Dataset):
    def __init__(self, video_files, video_labels, target_frames=8, transform=None):
        self.video_files = video_files
        self.labels = video_labels
        self.target_frames = target_frames
        self.transform = transform

    def __len__(self):
        return len(self.video_files)

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return None

        # Evenly space frame indices (optimized jump-based extraction)
        indices = np.linspace(0, total_frames - 1, self.target_frames).astype(int)
        
        frames = []
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR (OpenCV) to RGB (Model expects)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()

        if not frames:
            return None
            
        while len(frames) < self.target_frames:
            frames.append(frames[-1].clone())
        
        return torch.stack(frames)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = self.labels[idx]
        try:
            frames = self.extract_frames(video_path)
            if frames is None:
                return torch.zeros((self.target_frames, 3, 224, 224)), -1
            return frames, label
        except Exception:
            return torch.zeros((self.target_frames, 3, 224, 224)), -1

# 2. Model Architecture
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

# 3. Hybrid Pipeline Components (High worker count for i7-13650HX)
executor = ThreadPoolExecutor(max_workers=16)

def producer(dataset, indices, batch_size, queue, stop_event):
    """Produces batches of data into the prefetch queue."""
    for i in range(0, len(indices), batch_size):
        if stop_event.is_set():
            break
        
        batch_idx = indices[i : i + batch_size]
        
        # Async loading using ThreadPoolExecutor
        futures = [executor.submit(dataset.__getitem__, idx) for idx in batch_idx]
        batch_data = [f.result() for f in futures]
        
        # Collate and filter
        valid_data = [item for item in batch_data if item[1] != -1]
        if not valid_data:
            continue
            
        tensors, labels = zip(*valid_data)
        batch = (torch.stack(tensors), torch.tensor(labels, dtype=torch.long))
        
        queue.put(batch)
    
    queue.put(None) # Signal end of data

def main():
    # 4. OPTIMIZATIONS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    
    print(f"Using device: {device}")
    
    script_dir = Path(__file__).resolve().parent
    video_root = script_dir / "data" / "videos"
    
    # Dataset scan as before
    real_paths = [video_root / "celeb" / "Celeb-real", video_root / "celeb" / "YouTube-real", video_root / "faceforencics" / "real"]
    fake_paths = [video_root / "celeb" / "Celeb-synthesis", video_root / "faceforencics" / "fake" / "Deepfakes", video_root / "faceforencics" / "fake" / "Face2Face", video_root / "faceforencics" / "fake" / "FaceSwap", video_root / "faceforencics" / "fake" / "NeuralTextures"]
    
    real_files, fake_files = [], []
    video_ext = ('.mp4', '.avi', '.mov')
    for path in real_paths:
        if path.exists(): real_files.extend([f for f in path.rglob("*") if f.suffix.lower() in video_ext])
    for path in fake_paths:
        if path.exists(): fake_files.extend([f for f in path.rglob("*") if f.suffix.lower() in video_ext])
            
    # Balance
    min_len = min(len(real_files), len(fake_files))
    real_files = random.sample(real_files, min_len)
    fake_files = random.sample(fake_files, min_len)
    
    all_files = real_files + fake_files
    all_labels = [0]*min_len + [1]*min_len
    
    combined = list(zip(all_files, all_labels))
    random.shuffle(combined)
    all_files, all_labels = zip(*combined)
    
    # Split
    split_idx = int(0.8 * len(all_files))
    train_files, val_files = all_files[:split_idx], all_files[split_idx:]
    train_labels, val_labels = all_labels[:split_idx], all_labels[split_idx:]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = RawVideoDataset(train_files, train_labels, target_frames=8, transform=transform)
    val_dataset = RawVideoDataset(val_files, val_labels, target_frames=8, transform=transform)
    
    # Model Setup
    model = ResNeXtLSTM().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    batch_size = 8
    epochs = 15
    patience = 3
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    model_save_path = script_dir / "models" / "video" / "best_video_model.pt"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    
    prefetch_queue = Queue(maxsize=20)
    stop_event = threading.Event()
    
    print("\nStarting high-performance hybrid training...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_indices = list(range(len(train_dataset)))
        random.shuffle(train_indices)
        
        # Start producer thread
        while not prefetch_queue.empty(): prefetch_queue.get()
        producer_thread = threading.Thread(target=producer, args=(train_dataset, train_indices, batch_size, prefetch_queue, stop_event))
        producer_thread.start()
        
        total_train_loss = 0.0
        pbar = tqdm(total=len(train_indices)//batch_size + 1, desc=f"Epoch {epoch}/{epochs} [Train]")
        
        while True:
            batch = prefetch_queue.get()
            if batch is None: break
            
            inputs, labels = batch
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item() * inputs.size(0)
            
            # Monitoring
            cpu_usage = psutil.cpu_percent()
            queue_size = prefetch_queue.qsize()
            pbar.set_postfix(cpu=f"{cpu_usage}%", q=queue_size)
            pbar.update(1)
            
        pbar.close()
        producer_thread.join()
        avg_train_loss = total_train_loss / max(1, len(train_dataset))
        
        # Validation (Synchronous for stability)
        model.eval()
        total_val_loss, correct, total_val_samples = 0.0, 0, 0
        val_indices = list(range(len(val_dataset)))
        
        while not prefetch_queue.empty(): prefetch_queue.get()
        v_producer_thread = threading.Thread(target=producer, args=(val_dataset, val_indices, batch_size, prefetch_queue, stop_event))
        v_producer_thread.start()
        
        v_pbar = tqdm(total=len(val_indices)//batch_size + 1, desc=f"Epoch {epoch}/{epochs} [Val]")
        with torch.no_grad():
            while True:
                batch = prefetch_queue.get()
                if batch is None: break
                
                inputs, labels = batch
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total_val_samples += labels.size(0)
                v_pbar.update(1)
        v_pbar.close()
        v_producer_thread.join()
        
        avg_val_loss = total_val_loss / max(1, len(val_dataset))
        val_acc = correct / max(1, total_val_samples)
        
        print(f"\nEpoch {epoch}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | Val Acc={val_acc*100:.2f}%")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            print("Model saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggers")
                stop_event.set()
                break
                
    print(f"\nTraining completed. Best model saved at: {model_save_path}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        executor.shutdown(wait=False)