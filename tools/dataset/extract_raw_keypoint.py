import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import cv2
import zipfile
import tempfile
import shutil

# Add the project root to the Python path to allow importing from yolo
try:
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(project_root))
    from yolo.yolo_skeleton import YOLOSkeleton
except ImportError as e:
    print(f"Error: {e}. Please ensure 'yolo/yolo_skeleton.py' exists.")
    sys.exit(1)

def process_activity_folder(input_dir: Path, 
                            output_render_dir: Path, 
                            output_json_dir: Path,
                            model_path: Path, 
                            activity_name: str,
                            conf_threshold: float):
    """
    Processes a folder of images for a single activity.
    It performs skeleton tracking, saves rendered images, and extracts keypoints to JSON files for each person.
    """
    
    # --- 1. Find and sort images ---
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = sorted([p for p in input_dir.glob('**/*') if p.suffix.lower() in image_extensions], key=lambda p: p.name)

    if not image_files:
        print(f"Error: No images found in '{input_dir}'")
        return

    print(f"Found {len(image_files)} images in '{input_dir}'.")

    # --- 2. Initialize Model ---
    try:
        yolo_model = YOLOSkeleton(model_path)
    except Exception as e:
        print(f"Error initializing YOLO skeleton detector: {e}")
        return

    # --- 3. Perform Tracking ---
    print("Performing skeleton tracking...")
    results_sequence = []
    for img_path in tqdm(image_files, desc="Tracking frames"):
        # Call track on a single image; persist=True maintains state
        result = yolo_model.model.track(source=str(img_path), persist=True, verbose=False, conf=conf_threshold)
        results_sequence.append(result[0]) # result is a list with one item

    # --- 4. Render and Save Images ---
    print(f"Saving rendered images to '{output_render_dir}'...")
    output_render_dir.mkdir(parents=True, exist_ok=True)
    for i, (img_path, frame_results) in enumerate(zip(image_files, results_sequence)):
        if frame_results:
            annotated_image = frame_results.plot()
            output_filename = f"{img_path.stem}_rendered{img_path.suffix}"
            output_path = output_render_dir / output_filename
            cv2.imwrite(str(output_path), annotated_image)

    # --- 5. Extract Keypoints ---
    print("Extracting keypoints to JSON...")
    tracked_skeletons = {}
    num_frames = len(image_files)

    for frame_idx, frame_results in enumerate(results_sequence):
        if frame_results.boxes.id is None:
            continue

        track_ids = frame_results.boxes.id.int().cpu().tolist()
        keypoints_xy = frame_results.keypoints.xy.cpu().numpy()
        scores = frame_results.keypoints.conf.cpu().numpy()

        for i, track_id in enumerate(track_ids):
            if track_id not in tracked_skeletons:
                tracked_skeletons[track_id] = [None] * num_frames
            
            person_data = {
                "keypoints": keypoints_xy[i].tolist(),
                "scores": scores[i].tolist()
            }
            tracked_skeletons[track_id][frame_idx] = person_data

    # --- 6. Save Keypoints to JSON ---
    if not tracked_skeletons:
        print("Warning: No persons were tracked. No JSON files will be created.")
        return

    activity_output_dir = output_json_dir / activity_name
    activity_output_dir.mkdir(parents=True, exist_ok=True)
    
    num_saved = 0
    folder_name_stem = input_dir.name
    for track_id, person_frames in tracked_skeletons.items():
        output_filename = f"{folder_name_stem}_person_{track_id}.json"
        json_output_path = activity_output_dir / output_filename
        
        with open(json_output_path, 'w') as f:
            json.dump(person_frames, f, indent=2)
        num_saved += 1
        
    print(f"Success: Saved {num_saved} JSON file(s) to '{activity_output_dir}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Process a folder of activity images (or a zip file of images): render skeletons and extract keypoints to JSON.",
        epilog="Example: python tools/dataset/extract_raw_keypoint.py my_activity_images.zip --output_render_dir rendered --output_json_dir keypoints"
    )
    parser.add_argument('input_path', type=str, help='Path to the directory of images or a zip file for one activity.')
    parser.add_argument('--output_render_dir', type=str, required=True, help='Directory to save the rendered images.')
    parser.add_argument('--output_json_dir', type=str, required=True, help='Root directory to save the output JSON files.')
    parser.add_argument('--activity_name', type=str, default=None, help='Name of the activity. If not provided, it is inferred from the input directory or zip file name.')
    parser.add_argument('--model-path', type=str, default='yolo/yolo26n-pose.pt', help='Path to the YOLO model directory.')
    parser.add_argument('--conf', type=float, default=0.1, help='Confidence threshold for detection.')
    
    args = parser.parse_args()

    input_path = Path(args.input_path)
    render_path = Path(args.output_render_dir)
    json_path = Path(args.output_json_dir)
    model_path = Path(args.model_path)
    
    temp_dir = None
    if input_path.is_file() and input_path.suffix.lower() == '.zip':
        try:
            temp_dir = tempfile.mkdtemp(prefix="har_zip_")
            print(f"Extracting '{input_path}' to temporary directory '{temp_dir}'...")
            with zipfile.ZipFile(input_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # The input for processing is now the temporary directory
            process_input_path = Path(temp_dir)
            activity_name = args.activity_name if args.activity_name else input_path.stem
        except (zipfile.BadZipFile, FileNotFoundError) as e:
            print(f"Error: {e}")
            if temp_dir:
                shutil.rmtree(temp_dir)
            return
    elif input_path.is_dir():
        process_input_path = input_path
        activity_name = args.activity_name if args.activity_name else input_path.name
    else:
        print(f"Error: Input path '{input_path}' is not a valid directory or .zip file.")
        return

    try:
        print("-" * 50)
        print(f"Processing Activity: {activity_name}")
        print(f"  Input: {input_path}")
        print(f"  Render output: {render_path}")
        print(f"  JSON output: {json_path}")
        print(f"  Confidence Threshold: {args.conf}")
        print("-" * 50)

        process_activity_folder(process_input_path, render_path, json_path, model_path, activity_name, args.conf)
    finally:
        if temp_dir:
            print(f"Cleaning up temporary directory '{temp_dir}'...")
            shutil.rmtree(temp_dir)

    print("="*50)
    print("Processing complete.")
    print("="*50)

if __name__ == '__main__':
    main()
