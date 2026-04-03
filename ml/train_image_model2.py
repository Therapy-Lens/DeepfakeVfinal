import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing
from torch.amp import autocast, GradScaler

def balance_and_sample(dataset, max_per_class=20000):
    class_indices = {0: [], 1: []}

    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)

    random.shuffle(class_indices[0])
    random.shuffle(class_indices[1])

    selected_indices = class_indices[0][:max_per_class] + class_indices[1][:max_per_class]
    random.shuffle(selected_indices)

    return Subset(dataset, selected_indices)

def main():
    # ---------------- CONFIG ----------------
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 64
    EPOCHS = 5
    LR = 1e-5
    IMG_SIZE = 192

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'images', 'ds2')

    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VAL_DIR   = os.path.join(DATA_DIR, 'valid')
    TEST_DIR  = os.path.join(DATA_DIR, 'test')

    MODEL_SAVE_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'backend', 'model', 'image_model_finetuned.pth'))
    PRETRAINED_LOAD_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'backend', 'model', 'image_model.pth'))

    print(f"Using device: {DEVICE}")

    # ---------------- TRANSFORMS ----------------
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # ---------------- DATASETS ----------------
    print("Loading datasets...")

    train_dataset_full = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset   = datasets.ImageFolder(VAL_DIR,   transform=val_transform)
    test_dataset  = datasets.ImageFolder(TEST_DIR,  transform=val_transform)

    print("Class mapping:", train_dataset_full.class_to_idx)

    # Balance and sample the training dataset to max 40,000 images (20k real, 20k fake)
    train_dataset = balance_and_sample(train_dataset_full, max_per_class=20000)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # ---------------- MODEL ----------------
    print("Loading EfficientNet-B4...")
    model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=2)
    
    try:
        model.load_state_dict(torch.load(PRETRAINED_LOAD_PATH, map_location=DEVICE))
        print("Loaded existing trained model (fine-tuning)")
    except Exception as e:
        print(f"ERROR loading existing model at {PRETRAINED_LOAD_PATH}: {e}")
        return
        
    # Freeze ALL layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze ONLY classifier
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    model = model.to(DEVICE)

    # Optimizer ONLY for trainable params
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=1e-4
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize Mixed Precision Scaler
    scaler = GradScaler()

    best_val_acc = 0
    patience = 3
    epochs_no_improve = 0

    history_loss = []
    history_val_acc = []

    # ---------------- TRAINING ----------------
    print("\nStarting Training...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            
            with autocast("cuda"):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        history_loss.append(avg_loss)

        # -------- VALIDATION --------
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

                with autocast("cuda"):
                    outputs = model(imgs)
                    
                preds = outputs.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        history_val_acc.append(val_acc)

        scheduler.step()

        print(f"\nEpoch {epoch+1}: Loss={avg_loss:.4f} | Val Acc={val_acc*100:.2f}%")

        # -------- SAVE BEST --------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"--> Saved best model to: {MODEL_SAVE_PATH}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered!")
                break

    # ---------------- TEST EVALUATION ----------------
    print("\nEvaluating on TEST set...")

    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    correct, total = 0, 0

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Testing"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            with autocast("cuda"):
                outputs = model(imgs)
                
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_acc = correct / total
    print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")

    # ---------------- PLOTS ----------------
    try:
        plt.figure(figsize=(10,4))

        plt.subplot(1,2,1)
        plt.plot(history_loss)
        plt.title("Training Loss")

        plt.subplot(1,2,2)
        plt.plot([x*100 for x in history_val_acc])
        plt.title("Validation Accuracy")

        plt.tight_layout()
        plt.savefig("image_training_curves_finetuned.png")
    except Exception as e:
        print("Failed to save plot:", e)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()