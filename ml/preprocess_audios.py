import os
import random
import librosa
import numpy as np
import torch
import cv2
from tqdm import tqdm
import shutil

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "data", "audio")
PROCESSED_ROOT = os.path.join(BASE_DIR, "data", "processed_audio")
SR = 16000
N_MELS = 128
TARGET_SHAPE = (128, 128)

def setup_directories():
    """Ensure output directories exist and are clean."""
    if os.path.exists(PROCESSED_ROOT):
        print(f"Cleaning existing processed data at {PROCESSED_ROOT}...")
        shutil.rmtree(PROCESSED_ROOT)
    
    os.makedirs(os.path.join(PROCESSED_ROOT, "real"), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_ROOT, "fake"), exist_ok=True)

def load_file_list():
    """Scan directories and return balanced lists of REAL and FAKE audio files."""
    real_dir = os.path.join(DATA_ROOT, "real_samples")
    
    # Collect all real files
    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) 
                  if f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac'))]
    
    # Collect all fake files from other subdirectories
    fake_files = []
    for folder in os.listdir(DATA_ROOT):
        folder_path = os.path.join(DATA_ROOT, folder)
        if os.path.isdir(folder_path) and folder != "real_samples" and folder != "processed_audio":
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                     if f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac'))]
            fake_files.extend(files)
    
    print(f"Stats: Found {len(real_files)} Real files.")
    print(f"Stats: Found {len(fake_files)} Total Fake files across subdirectories.")

    # BALANCING
    random.shuffle(fake_files)
    target_count = len(real_files)
    balanced_fake_files = fake_files[:target_count]
    
    print(f"Balancing: Sampled {len(balanced_fake_files)} Fake files to match Real count.")
    
    return real_files, balanced_fake_files

def process_audio(file_path):
    """Load, resample, convert to Mel Spectrogram, resize, and normalize."""
    try:
        # 1. Load Audio (16kHz, Mono)
        y, sr = librosa.load(file_path, sr=SR, mono=True)
        
        # Ensure minimum length (pad if too short)
        if len(y) < SR:
            y = np.pad(y, (0, SR - len(y)))

        # 2. Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        
        # 3. Log Scale (dB)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 4. Resize (128, 128)
        # librosa melspec shape is [n_mels, time]
        resized_mel = cv2.resize(mel_db, TARGET_SHAPE, interpolation=cv2.INTER_AREA)
        
        # 5. Min-Max Normalization to [0, 1]
        mel_min = resized_mel.min()
        mel_max = resized_mel.max()
        if mel_max > mel_min:
            normalized_mel = (resized_mel - mel_min) / (mel_max - mel_min)
        else:
            normalized_mel = resized_mel - mel_min
            
        # 6. Convert to Tensor [1, 128, 128]
        tensor = torch.FloatTensor(normalized_mel).unsqueeze(0)
        return tensor

    except Exception as e:
        # print(f"\nError processing {file_path}: {e}")
        return None

def main():
    setup_directories()
    real_files, fake_files = load_file_list()
    
    # Process Real
    print("\nProcessing REAL audio samples...")
    success_real = 0
    for idx, f in enumerate(tqdm(real_files)):
        tensor = process_audio(f)
        if tensor is not None:
            save_path = os.path.join(PROCESSED_ROOT, "real", f"real_{idx:04d}.pt")
            torch.save(tensor, save_path)
            success_real += 1
            
    # Process Fake
    print("\nProcessing FAKE audio samples...")
    success_fake = 0
    for idx, f in enumerate(tqdm(fake_files)):
        tensor = process_audio(f)
        if tensor is not None:
            save_path = os.path.join(PROCESSED_ROOT, "fake", f"fake_{idx:04d}.pt")
            torch.save(tensor, save_path)
            success_fake += 1

    print("\n" + "="*30)
    print("PREPROCESSING COMPLETE")
    print(f"Real Processed: {success_real}")
    print(f"Fake Processed: {success_fake}")
    print(f"Total Tensors Saved: {success_real + success_fake}")
    print(f"Data saved to: {PROCESSED_ROOT}")
    print("="*30)

if __name__ == "__main__":
    main()
