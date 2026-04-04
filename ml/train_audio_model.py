import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed_audio")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "..", "backend", "model", "audio_model.pt")
BATCH_SIZE = 32
LR = 1e-4
EPOCH_COUNT = 15

# Ensure models directory exists
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

class AudioDataset(Dataset):
    """Custom Dataset for loading preprocessed Mel Spectrogram .pt tensors."""
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []
        
        # Load Real
        real_dir = os.path.join(root_dir, "real")
        for f in os.listdir(real_dir):
            if f.endswith(".pt"):
                self.samples.append((os.path.join(real_dir, f), 0))
                
        # Load Fake
        fake_dir = os.path.join(root_dir, "fake")
        for f in os.listdir(fake_dir):
            if f.endswith(".pt"):
                self.samples.append((os.path.join(fake_dir, f), 1))
                
        print(f"Dataset: Loaded {len(self.samples)} total samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        tensor = torch.load(path, weights_only=True)
        return tensor, label

def get_model():
    """Load ResNet18, modify input layer for 1-channel, and adjust head for 2 classes."""
    # Use weights='DEFAULT' for latest pretrained weights (ImageNet)
    model = models.resnet18(weights='DEFAULT')
    
    # Standard ResNet expects 3 channels. Modify conv1 to accept 1 channel (Mel Spectrogram)
    # Original: self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Modify the fully connected layer for binary classification (Real vs Fake)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    return model

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: Using {device}")

    # 1. Dataset & Split
    full_dataset = AudioDataset(DATA_DIR)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Model, Loss, Optimizer
    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0
    
    # 3. Training Loop
    for epoch in range(EPOCH_COUNT):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"\nEpoch {epoch+1}/{EPOCH_COUNT}")
        pbar = tqdm(train_loader, desc="Training")
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f"{running_loss/len(train_loader):.4f}", 'acc': f"{100.*correct/total:.2f}%"})

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        epoch_acc = 100. * val_correct / val_total
        print(f"Validation -> Loss: {val_loss/len(val_loader):.4f}, Acc: {epoch_acc:.2f}%")

        if epoch_acc > best_acc:
            print(f"Accuracy improved from {best_acc:.2f}% to {epoch_acc:.2f}%. Saving Best Model.")
            best_acc = epoch_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"\n" + "="*30)
    print("TRAINING COMPLETE")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print("="*30)

if __name__ == "__main__":
    train()
