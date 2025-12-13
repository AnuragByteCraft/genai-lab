import cv2
import os

def extract_frames(video_path, output_base_folder, start_sec=100, end_sec=105):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(output_base_folder, video_name)
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"🎞️ Video: {video_name} | Duration: {duration:.2f}s | FPS: {fps:.2f}")

    if start_sec >= duration:
        print(f"⚠️ Start time {start_sec}s exceeds video duration. Skipping {video_name}.")
        return
    if end_sec > duration:
        end_sec = duration
        print(f"⚠️ End time adjusted to {end_sec:.2f}s (max duration)")

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # 🔥 Accurate seeking
    current_frame = start_frame

    while cap.isOpened() and current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ Failed to read frame {current_frame}")
            break

        frame_filename = os.path.join(output_folder, f"frame_{current_frame}.jpg")
        cv2.imwrite(frame_filename, frame)
        current_frame += 1

    cap.release()
    print(f"✅ Extracted frames for '{video_name}' from {start_sec}s to {end_sec}s")

def process_multiple_videos(video_paths, output_base_folder="output_frames", start_sec=100, end_sec=105):
    for video_path in video_paths:
        extract_frames(video_path, output_base_folder, start_sec, end_sec)

# Example usage
video_files = [
    "MOKSHA.mp4",
    "Moksha-2x.mp4",
    "Moksha-3x.mp4",
    "Moksha-4x.mp4"
    # etc.
]

process_multiple_videos(video_files, start_sec=100, end_sec=105)
