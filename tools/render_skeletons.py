"""
This script renders skeleton frames from a CSV file containing keypoint data.
"""

import pandas as pd
import numpy as np
import cv2
import os
import argparse

# The 17 keypoints from COCO dataset
KEYPOINT_DICT = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle"
}

# Connections between keypoints to form a skeleton
SKELETON_CONNECTIONS = [
    # Head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Torso
    (5, 6), (5, 11), (6, 12), (11, 12),
    # Arms
    (5, 7), (7, 9), (6, 8), (8, 10),
    # Legs
    (11, 13), (13, 15), (12, 14), (14, 16)
]

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Render skeleton frames from a CSV file.")
    parser.add_argument('--csv-path', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the rendered frames.')
    parser.add_argument('--person-id', type=int, default=1, help='ID of the person to render.')
    parser.add_argument('--video-width', type=int, default=1920, help='Width of the video frame for the canvas.')
    parser.add_argument('--video-height', type=int, default=1080, help='Height of the video frame for the canvas.')
    parser.add_argument('--sample-step', type=int, default=1, help='Render every N-th frame.')
    return parser.parse_args()

def draw_skeleton(frame, keypoints, connections):
    """Draws a skeleton on a frame."""
    for i, (x, y) in enumerate(keypoints):
        if x > 0 and y > 0:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    for (start_idx, end_idx) in connections:
        start_point = keypoints[start_idx]
        end_point = keypoints[end_idx]
        if start_point[0] > 0 and start_point[1] > 0 and end_point[0] > 0 and end_point[1] > 0:
            cv2.line(frame, (int(start_point[0]), int(start_point[1])), (int(end_point[0]), int(end_point[1])), (255, 0, 0), 2)
    return frame

def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    try:
        df = pd.read_csv(args.csv_path)
    except FileNotFoundError:
        print(f"Error: The file {args.csv_path} was not found.")
        return

    # Filter by person ID
    person_df = df[df['ID'] == args.person_id]
    if person_df.empty:
        print(f"No data found for person ID {args.person_id}.")
        return
        
    # Get keypoint column names
    keypoint_cols = [f'KP{i}_{axis}' for i in range(len(KEYPOINT_DICT)) for axis in ['X', 'Y']]
    
    # Check if all keypoint columns exist
    if not all(col in person_df.columns for col in keypoint_cols):
        print("Error: CSV file is missing some keypoint columns.")
        return

    # Process frames
    for i in range(0, len(person_df), args.sample_step):
        row = person_df.iloc[i]
        
        # Create a blank black canvas
        frame = np.zeros((args.video_height, args.video_width, 3), dtype=np.uint8)
        
        # Extract keypoints
        keypoints = []
        for kp_idx in range(len(KEYPOINT_DICT)):
            x = row.get(f'KP{kp_idx}_X', 0)
            y = row.get(f'KP{kp_idx}_Y', 0)
            keypoints.append((x, y))
        
        keypoints = np.array(keypoints)

        # Draw the skeleton on the frame
        frame_with_skeleton = draw_skeleton(frame, keypoints, SKELETON_CONNECTIONS)
        
        # Save the frame
        frame_number = row.name  # Use the original index from the dataframe
        output_path = os.path.join(args.output_dir, f"frame_{frame_number:06d}.png")
        cv2.imwrite(output_path, frame_with_skeleton)

    print(f"Rendered {len(person_df) // args.sample_step} frames to {args.output_dir}")

if __name__ == '__main__':
    main()
