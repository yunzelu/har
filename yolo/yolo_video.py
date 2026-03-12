import cv2
import time
import sys
import os
import datetime
import csv
import argparse
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = 'yolo/yolov8n-pose_openvino_model/'
IMGSZ = (352, 640)

def process_video(input_path, output_path):
    """
    Processes a video to extract pose keypoints and saves them to a CSV file.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to the output CSV file.
    """
    # --- SETUP CHECKS ---
    if not os.path.exists(input_path):
        print(f"[ERROR] Input video not found: {input_path}")
        sys.exit(1)

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[INFO] Created output directory: {output_dir}")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: Library missing. Run: pip install ultralytics")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"[CRITICAL ERROR] Model folder not found: {MODEL_PATH}")
        print("Please ensure the 'yolov8n-pose_openvino_model' folder is in the 'yolo' directory.")
        sys.exit(1)

    # --- INITIALIZATION ---
    print(f"[INFO] Loading model from: {MODEL_PATH}...")
    model = YOLO(MODEL_PATH, task='pose')
    print("[INFO] Model ready.")

    print(f"[INFO] Opening video: {input_path}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {input_path}")
        sys.exit(1)
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Video properties: {total_frames} frames @ {video_fps:.2f} FPS")


    with open(output_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Header: Timestamp, ID, Keypoints...
        header = ["Timestamp", "ID"] + [f"KP{i}_{c}" for i in range(17) for c in ["X", "Y", "C"]]
        csv_writer.writerow(header)
        print(f"[INFO] Started writing to: {output_path}")

        frame_num = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame_num += 1
            # Using frame number and FPS to calculate timestamp
            # This is more reliable for video files than datetime.now()
            current_time_sec = frame_num / video_fps
            timestamp = f"{int(current_time_sec // 3600):02}:{int((current_time_sec % 3600) // 60):02}:{current_time_sec % 60:06.3f}"


            # --- INFERENCE ---
            results = model.track(
                frame,
                persist=True,
                verbose=False,
                imgsz=IMGSZ,
                conf=0.5
            )
            
            # --- SAVE DATA ---
            if results[0].boxes.id is not None:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                kpts = results[0].keypoints.data.cpu().numpy()

                for i, p_id in enumerate(ids):
                    row = [timestamp, p_id]
                    row.extend(kpts[i].flatten())
                    csv_writer.writerow(row)
            
            # Print progress
            if frame_num % 100 == 0:
                print(f"  Processed frame {frame_num}/{total_frames}")

    cap.release()
    print(f"[INFO] Processing complete. Output saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Pose Estimation from Video")
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to the input video file."
    )
    parser.add_argument(
        "--output", 
        type=str,
        default=None,
        help="Path to the output CSV file. Defaults to 'Pose_Data/<video_name>.csv'"
    )

    args = parser.parse_args()
    
    # Determine output path if not specified
    if args.output is None:
        video_name = os.path.splitext(os.path.basename(args.input))[0]
        output_dir = "Pose_Data"
        # Ensure the directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        args.output = os.path.join(output_dir, f"{video_name}_pose_data.csv")


    process_video(args.input, args.output)
