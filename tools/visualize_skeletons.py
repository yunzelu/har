
import cv2
import json
import zipfile
import tempfile
import argparse
from pathlib import Path
import numpy as np

# Skeleton mapping for YOLOv8-pose (COCO keypoints)
# This defines which keypoints to connect to form the skeleton.
SKELETON_MAP = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

# A list of colors to draw different skeletons if multiple people are detected
SKELETON_COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (0, 255, 255),  # Cyan
    (255, 0, 255),  # Magenta
]

def draw_skeleton(image, person_data, color, min_score_threshold=0.5):
    """
    Draws a single skeleton on the image.

    Args:
        image (np.ndarray): The image to draw on.
        person_data (dict): A dictionary containing 'keypoints' and 'scores'.
        color (tuple): The BGR color tuple to use for drawing.
        min_score_threshold (float): Keypoints with a score below this will not be drawn.
    """
    keypoints = person_data['keypoints']
    scores = person_data['scores']

    # Draw keypoint circles
    for i, (x, y) in enumerate(keypoints):
        if scores[i] > min_score_threshold:
            cv2.circle(image, (int(x), int(y)), 3, color, -1)

    # Draw skeleton lines
    for joint1_idx, joint2_idx in SKELETON_MAP:
        if scores[joint1_idx] > min_score_threshold and scores[joint2_idx] > min_score_threshold:
            x1, y1 = keypoints[joint1_idx]
            x2, y2 = keypoints[joint2_idx]
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

def visualize_skeletons(json_path: Path):
    """
    Main visualization function.
    """
    if not json_path.exists():
        print(f"Error: JSON file not found at {json_path}")
        return

    # --- 1. Infer paths and load data ---
    try:
        # Example json_path: .../keypoints_har_up/Subject1/Activity6/Trial1/File.json
        # We need to find: .../HAR_UP_Dataset/Subject1/Activity6/Trial1/File.zip
        parts = json_path.parts
        # Find the 'keypoints_har_up' index
        base_index = parts.index('keypoints_har_up')
        relative_path = Path(*parts[base_index + 1:])
        
        project_root = Path(*parts[:base_index])
        zip_file_name = json_path.with_suffix('.zip').name
        
        zip_path = project_root / 'HAR_UP_Dataset' / relative_path.parent / zip_file_name
        
        if not zip_path.exists():
            print(f"Error: Could not find the corresponding image zip file at {zip_path}")
            return
            
        with open(json_path, 'r') as f:
            all_frames_data = json.load(f)
            
    except (ValueError, IndexError) as e:
        print(f"Error: Could not determine the image zip path from the JSON path.")
        print("Please ensure your directory structure is correct (e.g., '/keypoints_har_up/' and '/HAR_UP_Dataset/' in the path).")
        return
        

    # --- 2. Unzip images to a temporary directory ---
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        
        image_files = sorted([p for p in Path(tmpdir).rglob('*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg']])

        if len(image_files) != len(all_frames_data):
            print("Warning: The number of images does not match the number of frames in the JSON file.")
            print(f"Images found: {len(image_files)}, JSON frames: {len(all_frames_data)}")

        # --- 3. Visualization Loop ---
        current_frame_index = 0
        while True:
            # Boundary check
            if not (0 <= current_frame_index < len(image_files)):
                current_frame_index = max(0, min(current_frame_index, len(image_files) - 1))

            frame_image_path = image_files[current_frame_index]
            frame_data = all_frames_data[current_frame_index]
            
            image = cv2.imread(str(frame_image_path))
            if image is None:
                print(f"Warning: Could not read image {frame_image_path}")
                continue

            # Draw all detected skeletons for the current frame
            for i, person_data in enumerate(frame_data):
                color = SKELETON_COLORS[i % len(SKELETON_COLORS)]
                draw_skeleton(image, person_data, color)

            # Display info on the image
            info_text = f"Frame: {current_frame_index + 1}/{len(image_files)} | Persons: {len(frame_data)}"
            cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Skeleton Viewer", image)

            # --- 4. User Navigation ---
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'): # 'q' to quit
                break
            elif key == ord('d') or key == 83: # 'd' or right arrow
                current_frame_index += 1
            elif key == ord('a') or key == 81: # 'a' or left arrow
                current_frame_index -= 1

    cv2.destroyAllWindows()
    print("Viewer closed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize skeletons from JSON files.")
    parser.add_argument("json_file", type=str, help="Path to the JSON file to visualize.")
    args = parser.parse_args()
    
    visualize_skeletons(Path(args.json_file))

