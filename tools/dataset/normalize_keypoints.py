import json
import os

# ==========================================
# CONFIGURATION
# ==========================================
MAIN_FOLDER = "keypoints"

# Manually set your video dimensions here
FRAME_WIDTH = 1920   
FRAME_HEIGHT = 1080  

# COCO 17 Format Hips
LEFT_HIP_INDEX = 11
RIGHT_HIP_INDEX = 12
# ==========================================

def centralize_and_normalize(keypoints, frame_width, frame_height):
    """Centers the skeleton on the mid-hip, then normalizes by resolution."""
    normalized = []
    
    # Safely get hips to calculate the center (root)
    try:
        l_hip = keypoints[LEFT_HIP_INDEX]
        r_hip = keypoints[RIGHT_HIP_INDEX]
        
        # Calculate Mid-Hip
        root_x = (l_hip[0] + r_hip[0]) / 2.0
        root_y = (l_hip[1] + r_hip[1]) / 2.0
    except (IndexError, TypeError):
        # Fallback if the keypoints list is somehow malformed
        root_x, root_y = 0.0, 0.0

    for kp in keypoints:
        # 1. Centralize: subtract the root coordinate
        centered_x = kp[0] - root_x
        centered_y = kp[1] - root_y
        
        # 2. Normalize: divide by frame dimensions
        final_x = centered_x / frame_width
        final_y = centered_y / frame_height
        
        normalized.append([final_x, final_y])
        
    return normalized

def process_json(input_path, output_path, frame_width, frame_height):
    """Processes a single JSON, handles nulls, and applies normalization."""
    with open(input_path, 'r') as f:
        data = json.load(f)
        
    # Dictionary to store the last known valid position of each joint index
    last_valid_joints = {}

    def process_keypoints_list(keypoints):
        # 1. Clean the keypoints (Handle Nulls)
        cleaned_keypoints = []
        for i, kp in enumerate(keypoints):
            # If the keypoint or its coordinates are null
            if kp is None or kp[0] is None or kp[1] is None:
                # Use the last valid coordinate for this joint (or 0.0 if it's the first frame)
                fallback_kp = last_valid_joints.get(i, [0.0, 0.0])
                cleaned_keypoints.append(fallback_kp)
            else:
                last_valid_joints[i] = kp  # Update the last valid location
                cleaned_keypoints.append(kp)
        
        # 2. Centralize and Normalize the cleaned keypoints
        return centralize_and_normalize(cleaned_keypoints, frame_width, frame_height)

    # 3. Apply exactly matching the repo's input/output looping structure
    for entry in data:
        if isinstance(entry, list):
            for sub_entry in entry:
                sub_entry["keypoints"] = process_keypoints_list(sub_entry["keypoints"])
        else:
            entry["keypoints"] = process_keypoints_list(entry["keypoints"])
            
    # Save the normalized data to a new JSON file
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

def process_all_json_files(main_folder):
    """Iterates through the directory and processes all JSON files."""
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            # Process only JSON files, ignore ones that are already normalized
            if file.endswith('.json') and not file.endswith('_normalized.json'):
                json_path = os.path.join(root, file)
                
                try:
                    # Create the output path
                    output_filename = file.replace('.json', '_normalized.json')
                    output_path = os.path.join(root, output_filename)
                    
                    # Process using the manual frame width and height
                    process_json(json_path, output_path, FRAME_WIDTH, FRAME_HEIGHT)
                    print(f"Processed: {json_path} -> {output_path}")
                    
                except Exception as e:
                    print(f"Error processing {json_path}: {str(e)}")

if __name__ == "__main__":
    print(f"Starting normalization. Target resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    process_all_json_files(MAIN_FOLDER)
    print("Normalization complete for all files.")