import os
import glob
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt

class TransformWrapper(Dataset):
    """
    To securely apply distinct Train and Validation transforms to a `random_split` dataset.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __len__(self):
        return len(self.subset)
        
    def __getitem__(self, idx):
        # Translate subset index back to global index
        real_idx = self.subset.indices[idx]
        path = self.subset.dataset.files[real_idx]
        label = self.subset.dataset.labels[real_idx]
        
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception:
            # Corrupt image flag
            return torch.zeros((3, 224, 224)), -1


class ImageDatasetCollector(Dataset):
    """Just collects paths and labels securely, without opening images yet."""
    def __init__(self, data_dir, max_images=10000):
        import random
        self.files = []
        self.labels = []
        
        real_dir = os.path.join(data_dir, 'Real')
        fake_dir = os.path.join(data_dir, 'Fake')
        
        real_images = []
        fake_images = []
        extensions = ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG')
        
        if os.path.exists(real_dir):
            for ext in extensions:
                real_images.extend(glob.glob(os.path.join(real_dir, ext)))
                
        if os.path.exists(fake_dir):
            for ext in extensions:
                fake_images.extend(glob.glob(os.path.join(fake_dir, ext)))
                
        # 1. Reduce duplicate consecutive frames: sort then pick every 5th frame
        real_images = sorted(real_images)[::5]
        fake_images = sorted(fake_images)[::5]
        
        # 2. Balance dataset: 50/50 exactly.
        target_per_class = max_images // 2
        
        # 3. Shuffle before truncating to guarantee good variety
        random.shuffle(real_images)
        random.shuffle(fake_images)
        
        real_images = real_images[:target_per_class]
        fake_images = fake_images[:target_per_class]
        
        all_images = [(f, 0) for f in real_images] + [(f, 1) for f in fake_images]
        
        # 4. Final shuffle to thoroughly mix real and fake labels natively
        random.shuffle(all_images)
        
        for path, label in all_images:
            self.files.append(path)
            self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pass


def custom_collate(batch):
    # Filter out corrupted
    batch = [item for item in batch if item[1] != -1]
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([])
    images, labels = zip(*batch)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data', 'images')
    
    # Save paths
    model_save_dir = os.path.abspath(os.path.join(base_dir, '..', 'backend', 'model'))
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, 'image_model.pth')
    plot_save_path = os.path.join(base_dir, 'image_training_curves.png')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Training Transforms -> Data Augmentation built-in
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation Transforms -> Deterministic
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Gathering dataset files...")
    full_dataset_paths = ImageDatasetCollector(data_dir, max_images=10000)
    
    if len(full_dataset_paths) == 0:
        print(f"No images found in {data_dir}. Ensure Real/ and Fake/ exist with images.")
        return
        
    train_size = int(0.8 * len(full_dataset_paths))
    val_size = len(full_dataset_paths) - train_size
    train_subset, val_subset = random_split(full_dataset_paths, [train_size, val_size])
    
    # Map Transforms
    train_dataset = TransformWrapper(train_subset, transform=train_transform)
    val_dataset = TransformWrapper(val_subset, transform=val_transform)
    
    print(f"Total Images: {len(full_dataset_paths)} | Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                              num_workers=2, pin_memory=True if device.type=='cuda' else False,
                              collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False,
                            num_workers=2, pin_memory=True if device.type=='cuda' else False,
                            collate_fn=custom_collate)
                            
    print("Loading EfficientNet-B4 from TIMM...")
    try:
        model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=2)
    except Exception as e:
        print("Failed to load timm model. Make sure to run `pip install timm`")
        raise e
        
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=1e-4)
    epochs = 10
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    patience = 3
    epochs_no_improve = 0
    
    history_train_loss = []
    history_val_acc = []
    
    print("\nStarting Training Loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        # --- TRAINING ---
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
            
        scheduler.step()
        avg_train_loss = total_loss / max(1, len(train_dataset))
        history_train_loss.append(avg_train_loss)
        
        # --- VALIDATION ---
        model.eval()
        correct = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                if len(inputs) == 0:
                    continue
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total_val += labels.size(0)
                
        val_acc = correct / max(1, total_val)
        history_val_acc.append(val_acc)
        
        # Logging Outputs
        print(f"\nEpoch {epoch}:")
        print(f"Loss = {avg_train_loss:.4f}")
        print(f"Val Acc = {val_acc * 100:.2f}%\n")
        
        # Early Stopping Logic & Model Tracking
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"--> Best model saved to: {model_save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered! No validation accuracy improvement for {patience} consecutive epochs.")
                break
                
    print("\nTraining completed.")
    
    # Plotting System
    try:
        plt.figure(figsize=(10, 4))
        
        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(history_train_loss) + 1), history_train_loss, 'b-', label='Train Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(history_val_acc) + 1), [acc * 100 for acc in history_val_acc], 'g-', label='Val Accuracy')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        
        plt.tight_layout()
        plt.savefig(plot_save_path)
        print(f"Training curves successfully plotted and saved to: {plot_save_path}")
    except Exception as e:
        print(f"Failed to plot training curves (matplotlib might be missing): {e}")

if __name__ == '__main__':
    main()
