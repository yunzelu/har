import json
import argparse
import copy

# --- CONFIGURATION ---
IMAGE_WIDTH = 640

# Standard 17-keypoint format swap pairs (Left vs Right)
SWAP_PAIRS = [
    (1, 2),   # Eyes
    (3, 4),   # Ears
    (5, 6),   # Shoulders
    (7, 8),   # Elbows
    (9, 10),  # Wrists
    (11, 12), # Hips
    (13, 14), # Knees
    (15, 16)  # Ankles
]

# --- FUNCTIONS ---
def parse_arguments():
    """Parses command line arguments for input and output paths."""
    parser = argparse.ArgumentParser(description="Mirror HAR keypoints horizontally.")
    parser.add_argument("--input", type=str, required=True, help="Path to the original JSON file")
    parser.add_argument("--output", type=str, required=True, help="Path to save the mirrored JSON file")
    return parser.parse_args()

def mirror_frame_data(keypoints, scores, image_width, swap_pairs):
    """
    Safely flips x-coordinates and swaps left/right indices for BOTH 
    keypoints and scores at the same time, keeping nulls intact.
    """
    if keypoints is None or scores is None:
        return keypoints, scores

    mirrored_kpts = [None] * len(keypoints)
    mirrored_scores = [None] * len(scores)

    # 1. Flip X coordinates and duplicate scores directly
    for i in range(len(keypoints)):
        kpt = keypoints[i]
        
        if kpt is not None:
            x, y = kpt[0], kpt[1]
            new_x = image_width - x
            mirrored_kpts[i] = [new_x, y]
        else:
            mirrored_kpts[i] = None
            
        if scores[i] is not None:
            mirrored_scores[i] = scores[i]
        else:
            mirrored_scores[i] = None

    # 2. Swap the Left and Right indices simultaneously
    for left_idx, right_idx in swap_pairs:
        temp_kpt = mirrored_kpts[left_idx]
        mirrored_kpts[left_idx] = mirrored_kpts[right_idx]
        mirrored_kpts[right_idx] = temp_kpt
        
        temp_score = mirrored_scores[left_idx]
        mirrored_scores[left_idx] = mirrored_scores[right_idx]
        mirrored_scores[right_idx] = temp_score

    return mirrored_kpts, mirrored_scores

def process_file(input_path, output_path, image_width, swap_pairs):
    """Loads, processes, and saves the mirrored dataset."""
    with open(input_path, 'r') as f:
        data = json.load(f)
        
    mirrored_data = []
    for frame in data:
        # NEW: Check if the entire frame object is null
        if frame is None:
            mirrored_data.append(None)
            continue
            
        new_frame = copy.deepcopy(frame)
        
        new_kpts, new_scores = mirror_frame_data(
            frame.get("keypoints"), 
            frame.get("scores"), 
            image_width, 
            swap_pairs
        )
        
        new_frame["keypoints"] = new_kpts
        new_frame["scores"] = new_scores
        
        mirrored_data.append(new_frame)
        
    with open(output_path, 'w') as f:
        json.dump(mirrored_data, f, indent=2)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    args = parse_arguments()
    process_file(args.input, args.output, IMAGE_WIDTH, SWAP_PAIRS)