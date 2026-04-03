import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from facenet_pytorch import MTCNN

# =========================
# CONFIG
# =========================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# AUTO-DETECT DATASET PATH (handles typo safely)
possible_paths = [
    os.path.join(BASE_DIR, 'data', 'videos', 'faceforencics'),
    os.path.join(BASE_DIR, 'data', 'videos', 'faceforensics')
]

VIDEO_ROOT = None
for p in possible_paths:
    if os.path.exists(p):
        VIDEO_ROOT = p
        break

if VIDEO_ROOT is None:
    raise Exception("❌ Could not find faceforensics dataset folder")

OUTPUT_ROOT = os.path.join(BASE_DIR, 'data', 'processed_videos')

REAL_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, 'real')
FAKE_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, 'fake')

NUM_FRAMES = 8
FRAME_SIZE = 224

FAKE_TYPES = [
    'Deepfakes',
    'Face2Face',
    'FaceShifter',
    'FaceSwap',
    'NeuralTextures'
]

# =========================
# INIT
# =========================
print(f"Using dataset path: {VIDEO_ROOT}")
print(f"Using device: {DEVICE}")

mtcnn = MTCNN(keep_all=False, device=DEVICE)

os.makedirs(REAL_OUTPUT_DIR, exist_ok=True)
os.makedirs(FAKE_OUTPUT_DIR, exist_ok=True)

# =========================
# COUNT EXISTING REAL
# =========================
existing_real = [f for f in os.listdir(REAL_OUTPUT_DIR) if f.endswith('.pt')]
TARGET_FAKE_COUNT = len(existing_real)

if TARGET_FAKE_COUNT == 0:
    raise Exception("❌ No REAL samples found. Cannot balance dataset.")

print(f"\nExisting REAL samples: {TARGET_FAKE_COUNT}")
print("We will process SAME number of FAKE samples.\n")

# =========================
# FRAME EXTRACTION
# =========================
def extract_frames(video_dir):
    extensions = ['.png', '.jpg', '.jpeg']
    try:
        all_files = os.listdir(video_dir)
    except Exception:
        return None
        
    frame_files = [f for f in all_files if any(f.lower().endswith(ext) for ext in extensions)]
    frame_files.sort()
    
    total_frames = len(frame_files)

    if total_frames < NUM_FRAMES:
        return None

    indices = np.linspace(0, total_frames - 1, NUM_FRAMES).astype(int)
    frames = []

    for idx in indices:
        img_path = os.path.join(video_dir, frame_files[idx])
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face = mtcnn(frame)
        if face is None:
            return None

        # Process MTCNN tensor accurately
        face = face.permute(1, 2, 0).cpu().numpy()
        face = cv2.resize(face, (FRAME_SIZE, FRAME_SIZE))
        face = np.transpose(face, (2, 0, 1))  # (C,H,W)

        frames.append(face)

    if len(frames) != NUM_FRAMES:
        return None

    return torch.tensor(np.array(frames), dtype=torch.float32)

# =========================
# COLLECT FAKE VIDEOS
# =========================
fake_root = os.path.join(VIDEO_ROOT, 'fake')

if not os.path.exists(fake_root):
    raise Exception("❌ Fake folder not found")

fake_video_groups = []

print("Scanning fake folders...\n")

for fake_type in FAKE_TYPES:
    type_path = os.path.join(fake_root, fake_type)

    if not os.path.exists(type_path):
        print(f"⚠️ Missing: {fake_type}")
        continue

    videos = [
        os.path.join(type_path, v)
        for v in os.listdir(type_path)
        if os.path.isdir(os.path.join(type_path, v))
    ]

    print(f"{fake_type}: {len(videos)} videos found")
    fake_video_groups.append((fake_type, videos))

if len(fake_video_groups) == 0:
    raise Exception("❌ No fake video folders found")

# =========================
# BALANCED SAMPLING
# =========================
per_type_limit = TARGET_FAKE_COUNT // len(fake_video_groups)

print(f"\nSampling ~{per_type_limit} videos from each type...\n")

selected_videos = []

for fake_type, videos in fake_video_groups:
    np.random.shuffle(videos)
    selected = videos[:per_type_limit]

    print(f"{fake_type}: {len(selected)} selected")
    selected_videos.extend([(fake_type, v) for v in selected])

# Trim extra (safety)
selected_videos = selected_videos[:TARGET_FAKE_COUNT]

print(f"\nTotal FAKE videos selected: {len(selected_videos)}\n")

# =========================
# PROCESS FAKE VIDEOS
# =========================
processed = 0
skipped = 0

for fake_type, video_path in tqdm(selected_videos, desc="Processing FAKE videos"):
    filename = f"{fake_type}_{os.path.basename(video_path)}.pt"
    save_path = os.path.join(FAKE_OUTPUT_DIR, filename)

    if os.path.exists(save_path):
        skipped += 1
        continue

    tensor = extract_frames(video_path)

    if tensor is None:
        skipped += 1
        continue

    torch.save(tensor, save_path)
    processed += 1

# =========================
# FINAL REPORT
# =========================
real_count = len([f for f in os.listdir(REAL_OUTPUT_DIR) if f.endswith('.pt')])
fake_count = len([f for f in os.listdir(FAKE_OUTPUT_DIR) if f.endswith('.pt')])

print("\n=========================")
print("PROCESS COMPLETE")
print(f"Processed FAKE: {processed}")
print(f"Skipped: {skipped}")
print(f"\nFINAL BALANCE:")
print(f"REAL: {real_count}")
print(f"FAKE: {fake_count}")
print("=========================")