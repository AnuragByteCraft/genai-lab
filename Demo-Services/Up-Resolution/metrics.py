import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from skimage import io, img_as_float
from skimage.metrics import niqe
from piq import brisque
from PIL import Image
import argparse

def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        i += 1
    cap.release()
    return frame_paths

def compute_niqe(image_path):
    try:
        img = img_as_float(io.imread(image_path))
        return niqe(img)
    except Exception as e:
        print(f"NIQE failed on {image_path}: {e}")
        return None

def compute_brisque(image_path):
    try:
        transform = transforms.Compose([transforms.ToTensor()])
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        return brisque(img_tensor, data_range=1.0).item()
    except Exception as e:
        print(f"BRISQUE failed on {image_path}: {e}")
        return None

def compute_video_quality_metrics(video_path):
    frame_dir = "frames_temp"
    print(f"\nExtracting frames from {video_path}...")
    frame_paths = extract_frames(video_path, frame_dir)
    
    niqe_scores = []
    brisque_scores = []

    print("Computing quality metrics...")
    for path in frame_paths:
        niqe_score = compute_niqe(path)
        brisque_score = compute_brisque(path)
        if niqe_score is not None:
            niqe_scores.append(niqe_score)
        if brisque_score is not None:
            brisque_scores.append(brisque_score)

    # Cleanup
    for path in frame_paths:
        os.remove(path)
    os.rmdir(frame_dir)

    print("\n--- Quality Report ---")
    print(f"Total Frames: {len(frame_paths)}")
    print(f"Average NIQE: {np.mean(niqe_scores):.4f}" if niqe_scores else "NIQE failed on all frames")
    print(f"Average BRISQUE: {np.mean(brisque_scores):.4f}" if brisque_scores else "BRISQUE failed on all frames")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Aakrosh-2x.mov")
    args = parser.parse_args()
    compute_video_quality_metrics(args.video_path)