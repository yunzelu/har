import os
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_INPUT_PATH = r"E:\HAR_UP_Dataset"
BASE_OUTPUT_PATH = r"D:\lu\project\har\data\up_dataset\temp"
EXTRACT_SCRIPT = r".\tools\dataset\extract_raw_keypoint.py"
VIDEO_SCRIPT = r".\tools\frames_to_video.py"
MAX_WORKERS = 6

def process_zip(subject, activity, trial_num, camera_num):
    """Handles extraction and rendering for one specific zip file."""
    trial_name = f"Trial{trial_num}"
    camera_name = f"Camera{camera_num}"
    folder_name = f"{subject}{activity}{trial_name}{camera_name}"
    zip_file = f"{folder_name}.zip"
    
    zip_path = os.path.join(BASE_INPUT_PATH, subject, activity, trial_name, zip_file)
    output_dir = os.path.join(BASE_OUTPUT_PATH, folder_name)
    video_path = os.path.join(output_dir, f"{folder_name}.mp4")

    if not os.path.exists(zip_path):
        return f"Skipped (Not Found): {folder_name}"

    os.makedirs(output_dir, exist_ok=True)

    try:
        # Step 1: Extract Keypoints
        # capture_output=True keeps the terminal clean so the progress bar stays at the bottom
        subprocess.run([
            "python", EXTRACT_SCRIPT,
            "--output_render_dir", output_dir,
            "--output_json_dir", output_dir,
            zip_path
        ], check=True, capture_output=True)

        # Step 2: Render Video
        subprocess.run([
            "python", VIDEO_SCRIPT,
            "--frames-dir", output_dir,
            "--output-video-path", video_path
        ], check=True, capture_output=True)
        
        return f"Finished: {folder_name}"

    except subprocess.CalledProcessError:
        return f"Error: {folder_name}"

def main():
    parser = argparse.ArgumentParser(description="Batch process HAR dataset with progress bars.")
    parser.add_argument("--subject", required=True, help="e.g., Subject1")
    parser.add_argument("--activity", required=True, help="e.g., Activity8")
    args = parser.parse_args()

    # Generate the 6 specific tasks (3 Trials x 2 Cameras)
    tasks = []
    for trial in range(1, 4):
        for camera in range(1, 3):
            tasks.append((args.subject, args.activity, trial, camera))

    print(f"Processing {args.subject} - {args.activity} ({len(tasks)} tasks)...")

    # Use a context manager for the pool and tqdm for the visual bar
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = {executor.submit(process_zip, *task): task for task in tasks}
        
        # Progress bar setup
        with tqdm(total=len(tasks), desc="Overall Progress", unit="file") as pbar:
            for future in as_completed(futures):
                result = future.result()
                # Print status above the progress bar
                tqdm.write(f"[{result}]")
                pbar.update(1)

if __name__ == "__main__":
    main()