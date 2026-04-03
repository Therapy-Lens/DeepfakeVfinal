import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnext50_32x4d, ResNeXt50_32x4d_Weights
import torchvision.transforms as transforms
from torch.optim import AdamW
from tqdm import tqdm

class VideoTensorDataset(Dataset):
    def __init__(self, data_dir, max_videos=2000):
        self.files = []
        self.labels = []
        
        real_dir = os.path.join(data_dir, 'real')
        fake_dir = os.path.join(data_dir, 'fake')
        
        all_videos = []
        if os.path.exists(real_dir):
            for f in glob.glob(os.path.join(real_dir, '*.pt')):
                all_videos.append((f, 0)) # real = 0
                
        if os.path.exists(fake_dir):
            for f in glob.glob(os.path.join(fake_dir, '*.pt')):
                all_videos.append((f, 1)) # fake = 1
                
        # Limit dataset size to safely avoid OOM and speed up experiment
        all_videos = all_videos[:max_videos]
        
        for path, label in all_videos:
            self.files.append(path)
            self.labels.append(label)
            
        # ImageNet Normalization values
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
                                              
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]
        
        try:
            tensor = torch.load(path, weights_only=True)
            # Expecting tensor of shape [20, 3, 224, 224]
            # Normalize frame sequence using ImageNet stats
            for i in range(tensor.size(0)):
                tensor[i] = self.normalize(tensor[i])
            return tensor, label
        except Exception as e:
            # Corrupted wrapper -> caught by collate_fn below
            return torch.zeros((20, 3, 224, 224)), -1

def custom_collate(batch):
    # Filter out any corrupted files (where label is -1)
    batch = [item for item in batch if item[1] != -1]
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([])
    tensors, labels = zip(*batch)
    return torch.stack(tensors), torch.tensor(labels, dtype=torch.long)


class ResNeXtLSTM(nn.Module):
    def __init__(self, hidden_size=512):
        super(ResNeXtLSTM, self).__init__()
        
        # 1. Feature Extractor -> outputs [2048] per frame
        resnext = resnext50_32x4d(weights=ResNeXt50_32x4d_Weights.IMAGENET1K_V1)
        # Exclude the final classification fully connected layer
        self.feature_extractor = nn.Sequential(*list(resnext.children())[:-1])
        
        # 2. LSTM analyzing sequence of frames sequentially
        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_size, num_layers=1, batch_first=True)
        
        # 3. Final target classification layer (Real=0 vs Fake=1)
        self.fc = nn.Linear(hidden_size, 2)
        
    def forward(self, x):
        # x shape: [batch_size, 20, 3, 224, 224]
        batch_size, seq_len, c, h, w = x.size()
        
        # Flatten time and batch limits for feature extraction
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.feature_extractor(x)  # shape: [batch*20, 2048, 1, 1]
        features = features.view(batch_size, seq_len, 2048)  # -> [batch_size, 20, 2048]
        
        # Feed entire sequence mapping into LSTM
        lstm_out, _ = self.lstm(features)
        
        # We only care about the very last sequence representation
        last_timestep_out = lstm_out[:, -1, :]  # shape: [batch_size, hidden_size]
        
        # Classification projection
        return self.fc(last_timestep_out)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data', 'processed_videos')
    
    # Save best model logic -> ml/../backend/model/video_model.pth
    model_save_dir = os.path.abspath(os.path.join(base_dir, '..', 'backend', 'model'))
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, 'video_model.pth')
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading datasets...")
    full_dataset = VideoTensorDataset(data_dir, max_videos=2000)
    
    if len(full_dataset) == 0:
        print("No .pt data found. Make sure preprocess_videos.py has extracted tensors first!")
        return
        
    # Standard Train/Validation split of 80/20
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Total videos: {len(full_dataset)} | Train: {train_size} | Validation: {val_size}")
    
    # Dataloaders - very low batch to avoid OUT OF MEMORY (OOM)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=custom_collate
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=custom_collate
    )
    
    # Network Initialization & Rules
    model = ResNeXtLSTM().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4) # AdamW
    criterion = nn.CrossEntropyLoss()
    
    epochs = 15
    best_val_acc = 0.0
    patience = 3
    epochs_no_improve = 0
    
    print("\nStarting Training Loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        # Training Routine
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            if len(inputs) == 0:
                continue
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            
        avg_train_loss = total_loss / max(1, train_size)
        
        # Evaluation Routine
        model.eval()
        correct = 0
        total_val_samples = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                if len(inputs) == 0:
                    continue
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total_val_samples += labels.size(0)
                
        val_acc = correct / max(1, total_val_samples)
        
        # Logging & Model Tracking Requirement
        print(f"\nEpoch {epoch}:")
        print(f"Loss = {avg_train_loss:.4f}")
        print(f"Validation Accuracy = {val_acc * 100:.2f}%\n")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"--> Best model saved to: {model_save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                # Early Stopping Logic 
                print(f"Early stopping triggered! No validation accuracy improvement for {patience} consecutive epochs.")
                break
                
    print("\nTraining completed.")

if __name__ == '__main__':
    main()
