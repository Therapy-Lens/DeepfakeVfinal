import sys
import os
import json
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import timm
from facenet_pytorch import MTCNN

def get_model(model_path, device):
    model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def main():
    print("DEBUG RUNNING: STARTING PREDICTION PIPELINE", file=sys.stderr)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    mtcnn = MTCNN(
        image_size=224,
        margin=20,
        keep_all=True,
        device=device
    )
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "..", "model", "image_model.pth")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}", file=sys.stderr)
        return
        
    file_size = os.path.getsize(model_path) / (1024 * 1024)
    file_time = os.path.getmtime(model_path)
    import datetime
    time_str = datetime.datetime.fromtimestamp(file_time).strftime('%Y-%m-%d %H:%M:%S')
    print(f"Model File: {model_path} (Size: {file_size:.2f} MB, Last Modified: {time_str})", file=sys.stderr)
    
    try:
        model = get_model(model_path, device)
        print("Model loaded successfully", file=sys.stderr)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 6. Sanity Test logic
    if len(sys.argv) > 1 and sys.argv[1] == "--sanity-test":
        fake_dir = os.path.join(base_dir, "..", "..", "ml", "data", "images", "Fake")
        if not os.path.exists(fake_dir):
            print(f"Sanity test Fake dir missing: {fake_dir}", file=sys.stderr)
            return
            
        import glob
        fake_images = glob.glob(os.path.join(fake_dir, "*.png")) + glob.glob(os.path.join(fake_dir, "*.jpg"))
        if len(fake_images) == 0:
            print("No fake images found for sanity test", file=sys.stderr)
            return
            
        image_path = fake_images[0]
        print(f"SANITY TEST: Testing on KNOWN FAKE image -> {image_path}", file=sys.stderr)
    elif len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print(json.dumps({"error": "No file passed"}))
        return

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(json.dumps({"error": f"Failed to open image: {e}"}))
        return
        
    # Convert to numpy for MTCNN
    image_np = np.array(image)
    try:
        faces = mtcnn(image_np)
    except Exception as e:
        print("MTCNN error:", e, file=sys.stderr)
        faces = None

    print("Face detected:", faces is not None, file=sys.stderr)

    if faces is not None:
        # If multiple faces, select largest
        if len(faces.shape) == 4:
            areas = [f.shape[1] * f.shape[2] for f in faces]
            face = faces[areas.index(max(areas))]
        else:
            face = faces

        input_tensor = face.unsqueeze(0).to(device)

    else:
        print("No face detected, using full image fallback", file=sys.stderr)
        
        # fallback to original pipeline
        input_tensor = transform(image).unsqueeze(0).to(device)

    print(f"Image shape: {input_tensor.shape}", file=sys.stderr)
    print(f"Tensor Min: {input_tensor.min().item():.4f}, Max: {input_tensor.max().item():.4f}", file=sys.stderr)

    # ---- Forward pass ----
    with torch.no_grad():
        outputs = model(input_tensor)

    # ---- Temperature scaling ----
    temperature = 3.0  # tune between 2.0–4.0
    outputs = outputs / temperature

    # ---- Softmax ----
    probs = torch.softmax(outputs, dim=1)[0]

    # ---- Prediction ----
    pred = probs.argmax().item()
    confidence = probs[pred].item() * 100

    # ---- Label mapping ----
    label = "REAL" if pred == 0 else "FAKE"

    # ---- Smart decision layer ----
    if confidence < 60:
        label = "UNCERTAIN"
    elif confidence < 80:
        label = "LIKELY " + label

    confidence = round(confidence, 1)

    print("Temperature:", temperature, file=sys.stderr)
    print("Probabilities:", probs.tolist(), file=sys.stderr)
    print("Final:", label, confidence, file=sys.stderr)

    result = {
        "prediction": label,
        "confidence": confidence
    }
    
    print(f"Final JSON Output: {result}", file=sys.stderr)
    print(json.dumps(result))

if __name__ == "__main__":
    main()
