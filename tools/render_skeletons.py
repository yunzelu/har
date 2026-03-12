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
    parser.add_argument('--person-id', type=int, default=None, help='ID of the person to render. Renders all persons if not specified.')
    parser.add_argument('--video-width', type=int, default=1920, help='Width of the video frame for the canvas.')
    parser.add_argument('--video-height', type=int, default=1080, help='Height of the video frame for the canvas.')
    parser.add_argument('--sample-step', type=int, default=1, help='Render every N-th frame.')
    parser.add_argument('--show-timestamp', action='store_true', help='Show timestamp on the frames.')
    parser.add_argument('--conf-threshold', type=float, default=0.6, help='Confidence threshold for rendering keypoints (0.0 to 1.0).')
    return parser.parse_args()

def draw_skeleton(frame, keypoints, connections, color, threshold):
    """Draws a skeleton on a frame, filtering by confidence."""
    # Draw keypoints
    for x, y, c in keypoints:
        if c > threshold and x > 0 and y > 0:
            cv2.circle(frame, (int(x), int(y)), 5, color, -1)

    # Draw connections
    for (start_idx, end_idx) in connections:
        start_x, start_y, start_c = keypoints[start_idx]
        end_x, end_y, end_c = keypoints[end_idx]

        if start_c > threshold and end_c > threshold and start_x > 0 and start_y > 0 and end_x > 0 and end_y > 0:
            cv2.line(frame, (int(start_x), int(start_y)), (int(end_x), int(end_y)), color, 2)
    return frame

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        df = pd.read_csv(args.csv_path)
    except FileNotFoundError:
        print(f"Error: The file {args.csv_path} was not found.")
        return

    if args.person_id is not None:
        df = df[df['PersonID'] == args.person_id]
        if df.empty:
            print(f"No data found for person ID {args.person_id}.")
            return

    keypoint_cols = [f'KP{i}_{axis}' for i in range(len(KEYPOINT_DICT)) for axis in ['X', 'Y']]
    if not all(col in df.columns for col in keypoint_cols):
        print("Error: CSV file is missing some keypoint columns.")
        return

    # Define a list of colors for different skeletons
    colors = [
        (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), 
        (0, 255, 255), (128, 0, 128), (128, 128, 0), (0, 128, 128)
    ]

    # Group by a frame identifier, assuming 'UnixTime' can identify a frame 
    unique_unixtimes = df['UnixTime'].unique()

    for i in range(0, len(unique_unixtimes), args.sample_step):
        uts = unique_unixtimes[i]
        frame_df = df[df['UnixTime'] == uts]
        
        if frame_df.empty:
            continue

        frame = np.zeros((args.video_height, args.video_width, 3), dtype=np.uint8)
        
        if args.show_timestamp:
            timestamp = frame_df.iloc[0]['Timestamp']
            unixtime = frame_df.iloc[0]['UnixTime']
            cv2.putText(frame, f"Timestamp: {timestamp}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"UnixTime: {unixtime}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        for person_index, row in frame_df.iterrows():
            keypoints = []
            for kp_idx in range(len(KEYPOINT_DICT)):
                x = row.get(f'KP{kp_idx}_X', 0)
                y = row.get(f'KP{kp_idx}_Y', 0)
                c = row.get(f'KP{kp_idx}_C', 0)
                keypoints.append((x, y, c))
            
            person_id = int(row.get('PersonID', 0))
            color = colors[person_id % len(colors)] # Cycle through colors based on PersonID
            
            draw_skeleton(frame, keypoints, SKELETON_CONNECTIONS, color, args.conf_threshold)

        # The frame_number should be based on the unixtime order to have a consistent sequence
        frame_number = np.where(unique_unixtimes == uts)[0][0]
        output_path = os.path.join(args.output_dir, f"frame_{frame_number:06d}.png")
        cv2.imwrite(output_path, frame)

    rendered_frames_count = len(range(0, len(unique_unixtimes), args.sample_step))
    print(f"Rendered {rendered_frames_count} frames to {args.output_dir}")


if __name__ == '__main__':
    main()
