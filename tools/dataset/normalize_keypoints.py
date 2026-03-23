import json
import os
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==========================================
# CONFIGURATION
# ==========================================
# Point this to the ROOT folder containing all the activity subfolders
MAIN_FOLDER = r"D:\lu\project\har\data\up_dataset"

# The new root folder where the mirrored structure and normalized files will go
OUTPUT_FOLDER = r"D:\lu\project\har\data\normalized"

# Manually set your video dimensions here
FRAME_WIDTH = 640   
FRAME_HEIGHT = 480  

# COCO 17 Format Hips
LEFT_HIP_INDEX = 11
RIGHT_HIP_INDEX = 12

# Execution
MAX_WORKERS = 5
# ==========================================

def centralize_and_normalize(keypoints, frame_width, frame_height):
    """Centers the skeleton and normalizes by resolution."""
    normalized_kpts = []
    
    # Calculate Mid-Hip (Root) using the smoothed raw coordinates
    l_hip = keypoints[LEFT_HIP_INDEX]
    r_hip = keypoints[RIGHT_HIP_INDEX]
    root_x = (l_hip[0] + r_hip[0]) / 2.0
    root_y = (l_hip[1] + r_hip[1]) / 2.0

    for i in range(len(keypoints)):
        # 1. Centralize and Normalize
        centered_x = keypoints[i][0] - root_x
        centered_y = keypoints[i][1] - root_y
        
        final_x = centered_x / frame_width
        final_y = centered_y / frame_height
        
        normalized_kpts.append([final_x, final_y])
            
    return normalized_kpts

def process_single_json(input_path, output_path):
    """Processes a single JSON, handles nulls, and applies normalization."""
    
    with open(input_path, 'r') as f:
        data = json.load(f)
        
    # Track the last valid state for temporal smoothing
    last_valid_state = {
        "keypoints": [[0.0, 0.0]] * 17,
        "scores": [0.0] * 17
    }

    def process_frame(frame_data):
        # 1. Handle completely null frames
        if frame_data is None or frame_data.get("keypoints") is None:
            return {
                "keypoints": centralize_and_normalize(
                    last_valid_state["keypoints"], 
                    FRAME_WIDTH, FRAME_HEIGHT
                ),
                "scores": copy.deepcopy(last_valid_state["scores"])
            }

        current_kpts = frame_data.get("keypoints", [])
        current_scores = frame_data.get("scores", [])
        
        cleaned_kpts = []
        cleaned_scores = []
        
        # 2. Joint-by-joint temporal smoothing
        for i in range(17):
            try:
                kp = current_kpts[i]
                score = current_scores[i]
                
                if kp is None or kp[0] is None or kp[1] is None or score is None:
                    raise ValueError("Null coordinate")
                    
                cleaned_kpts.append(kp)
                cleaned_scores.append(score)
                last_valid_state["keypoints"][i] = kp
                last_valid_state["scores"][i] = score
                
            except (IndexError, ValueError):
                cleaned_kpts.append(last_valid_state["keypoints"][i])
                cleaned_scores.append(last_valid_state["scores"][i])

        # 3. Apply normalization
        normalized_kpts = centralize_and_normalize(
            cleaned_kpts, 
            FRAME_WIDTH, FRAME_HEIGHT
        )
        
        return {
            "keypoints": normalized_kpts,
            "scores": cleaned_scores
        }

    # Build a brand new list to avoid 'NoneType' assignment errors
    normalized_data = []
    
    for entry in data:
        if isinstance(entry, list):
            new_sub_list = []
            for sub_entry in entry:
                processed = process_frame(sub_entry)
                if sub_entry is None:
                    new_sub_list.append(processed)
                else:
                    sub_entry["keypoints"] = processed["keypoints"]
                    sub_entry["scores"] = processed["scores"]
                    new_sub_list.append(sub_entry)
            normalized_data.append(new_sub_list)
        else:
            processed = process_frame(entry)
            if entry is None:
                normalized_data.append(processed)
            else:
                entry["keypoints"] = processed["keypoints"]
                entry["scores"] = processed["scores"]
                normalized_data.append(entry)
            
    with open(output_path, 'w') as f:
        json.dump(normalized_data, f, indent=2)
        
    return f"Processed: {os.path.basename(input_path)} -> {os.path.basename(output_path)}"

def process_all_json_files(main_folder, output_folder, max_workers):
    """Finds target JSONs, replicates folder structure, and processes them in parallel."""
    target_files = []
    
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if file.endswith('.json') and not file.endswith('_normalized.json'):
                input_path = os.path.join(root, file)
                
                # Calculate the relative path (e.g., "Walking\Subject1Activity1.json")
                relative_path = os.path.relpath(input_path, main_folder)
                
                # Build the new output path
                output_path = os.path.join(output_folder, relative_path)
                
                # Append the _normalized suffix
                output_path = output_path.replace('.json', '_normalized.json')
                
                # Ensure the target subfolder exists (creates \Walking\ if it doesn't exist)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                target_files.append((input_path, output_path))
                
    print(f"Found {len(target_files)} files to normalize.")

    # Spin up parallel workers
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Pass both input and output paths to the worker
        futures = {executor.submit(process_single_json, in_path, out_path): in_path for in_path, out_path in target_files}
        
        for future in as_completed(futures):
            try:
                result_msg = future.result()
                print(result_msg)
            except Exception as e:
                json_path = futures[future]
                print(f"Error processing {json_path}: {str(e)}")

if __name__ == "__main__":
    print(f"Starting batch normalization. Target resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    process_all_json_files(MAIN_FOLDER, OUTPUT_FOLDER, MAX_WORKERS)
    print("Normalization complete for all files.")