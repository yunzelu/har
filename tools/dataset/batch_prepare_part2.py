import os
import subprocess
import argparse
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_PATH = r"D:\lu\project\har\data\up_dataset"
COMBINE_SCRIPT = r".\tools\dataset\combine_json_lists.py"

def run_combine(subject, activity, label):
    # Ensure the target label directory exists (e.g., .../up_dataset/Walking)
    label_dir = os.path.join(BASE_PATH, label)
    os.makedirs(label_dir, exist_ok=True)

    # Prepare the 6 tasks (3 Trials x 2 Cameras)
    folders = []
    for trial in range(1, 4):
        for camera in range(1, 3):
            folder_name = f"{subject}{activity}Trial{trial}Camera{camera}"
            folders.append(folder_name)

    print(f"Combining JSONs for Subject: {subject}, Activity: {activity} -> Label: {label}")

    for folder_name in tqdm(folders, desc="Processing JSONs", unit="folder"):
        # Based on your example: D:\...\FolderName\FolderName
        input_dir = os.path.join(BASE_PATH, "temp", folder_name, folder_name)
        
        # Based on your example: D:\...\Label\FolderName.json
        output_json = os.path.join(label_dir, f"{folder_name}.json")

        if not os.path.exists(input_dir):
            tqdm.write(f"[SKIP] Directory not found: {input_dir}")
            continue

        try:
            # Running the combine script
            subprocess.run([
                "python", COMBINE_SCRIPT,
                input_dir,
                output_json
            ], check=True, capture_output=True, text=True)
            
            tqdm.write(f"[SUCCESS] Created: {label}\\{folder_name}.json")
            
        except subprocess.CalledProcessError as e:
            tqdm.write(f"[ERROR] Failed {folder_name}: {e.stderr.strip()}")

def main():
    parser = argparse.ArgumentParser(description="Batch combine HAR JSON lists.")
    parser.add_argument("--subject", required=True, help="e.g., Subject1")
    parser.add_argument("--activity", required=True, help="e.g., Activity6")
    parser.add_argument("--label", required=True, help="e.g., Walking")
    
    args = parser.parse_args()
    run_combine(args.subject, args.activity, args.label)

if __name__ == "__main__":
    main()