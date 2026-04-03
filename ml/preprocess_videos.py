import os
import glob
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm

def get_frame_paths(video_dir):
    """Get all image frame paths from the directory."""
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    frames = []
    for ext in extensions:
        frames.extend(glob.glob(os.path.join(video_dir, ext)))
    return sorted(frames)

def process_video(video_dir, mtcnn, target_frames=20):
    """
    Reads frames, samples them evenly, detects faces, 
    and returns a tensor of shape [20, 3, 224, 224].
    """
    frame_paths = get_frame_paths(video_dir)
    if not frame_paths:
        return None
        
    # Sample 20 evenly spaced frames
    if len(frame_paths) >= target_frames:
        indices = torch.linspace(0, len(frame_paths)-1, target_frames).long().tolist()
        sampled_paths = [frame_paths[i] for i in indices]
    else:
        # Not enough frames, read all for now
        sampled_paths = frame_paths
        
    processed_frames = []
    for path in sampled_paths:
        try:
            img = Image.open(path).convert('RGB')
            # Detect face and get tensor of shape [3, 224, 224]
            face_tensor = mtcnn(img) 
            if face_tensor is not None:
                processed_frames.append(face_tensor)
        except Exception:
            # Skip corrupted frames safely
            continue
            
    if not processed_frames:
        return None
        
    # Ensure exactly `target_frames` frames
    # If fewer -> repeat last frame
    while len(processed_frames) < target_frames:
        processed_frames.append(processed_frames[-1].clone())
        
    # If more -> trim
    if len(processed_frames) > target_frames:
        processed_frames = processed_frames[:target_frames]
        
    # Stack into [20, 3, 224, 224]
    video_tensor = torch.stack(processed_frames)
    return video_tensor

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_root_dir = os.path.join(base_dir, 'data', 'videos')
    out_dir = os.path.join(base_dir, 'data', 'processed_videos')
    
    # User requested to force GPU
    device = 'cuda'
    print(f"Using device: {device}")
    
    # Initialize MTCNN for face detection
    mtcnn = MTCNN(image_size=224, margin=20, device=device)
    
    MAX_CONSECUTIVE_FAILURES = 50
    
    consecutive_failures = 0
    success_count = 0
    total_processed = 0
    
    video_tasks = []
    
    for dataset in ['faceforencics', 'celeb']:
        root_dir = os.path.join(video_root_dir, dataset)
        
        # 1. Process Real videos
        real_dir = os.path.join(root_dir, 'real')
        if os.path.exists(real_dir):
            for seq in sorted(os.listdir(real_dir)):
                seq_path = os.path.join(real_dir, seq)
                if os.path.isdir(seq_path):
                    out_path = os.path.join(out_dir, 'real', f"{dataset}_{seq}.pt")
                    video_tasks.append((seq_path, out_path))
                    
        # 2. Process Fake videos
        fake_dir = os.path.join(root_dir, 'fake')
        if os.path.exists(fake_dir):
            for item in sorted(os.listdir(fake_dir)):
                item_path = os.path.join(fake_dir, item)
                if os.path.isdir(item_path):
                    # Check if this item is a method folder (e.g. Deepfakes) or a direct video sequence
                    has_subdirs = any(os.path.isdir(os.path.join(item_path, sub)) for sub in os.listdir(item_path))
                    if has_subdirs:
                        for seq in sorted(os.listdir(item_path)):
                            seq_path = os.path.join(item_path, seq)
                            if os.path.isdir(seq_path):
                                out_path = os.path.join(out_dir, 'fake', f"{dataset}_{item}_{seq}.pt")
                                video_tasks.append((seq_path, out_path))
                    else:
                        out_path = os.path.join(out_dir, 'fake', f"{dataset}_{item}.pt")
                        video_tasks.append((item_path, out_path))
                        
    print(f"Found {len(video_tasks)} video directories to process.")
    
    os.makedirs(os.path.join(out_dir, 'real'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'fake'), exist_ok=True)
    
    for in_path, out_path in tqdm(video_tasks, desc="Processing videos"):
        # Removed MAX_VIDEOS limit check to process all discovered videos
        try:
            tensor = process_video(in_path, mtcnn)
            if tensor is not None:
                torch.save(tensor, out_path)
                success_count += 1
                consecutive_failures = 0
            else:
                consecutive_failures += 1
        except Exception as e:
            consecutive_failures += 1
            
        total_processed += 1
        
        # Early Stopping Condition for complete crash protections
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            print("\nWARNING: Too many failures sequentially, stopping preprocessing to avoid errors")
            break
            
    print(f"\nPreprocessing complete. Successfully processed {success_count}/{len(video_tasks)} videos.")

if __name__ == '__main__':
    main()
