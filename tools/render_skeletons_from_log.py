"""
This script renders skeleton frames from a YOLO pose CSV file, based on timestamps from a radar log CSV.
"""

import pandas as pd
import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm

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
    parser = argparse.ArgumentParser(description="Render skeleton frames from YOLO pose and radar log CSV files.")
    parser.add_argument('--yolo-csv-path', type=str, required=True, help='Path to the input YOLO pose CSV file.')
    parser.add_argument('--radar-csv-path', type=str, required=True, help='Path to the input radar log CSV file.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the rendered frames.')
    parser.add_argument('--person-id', type=int, default=None, help='ID of the person to render. Renders all persons if not specified.')
    parser.add_argument('--video-width', type=int, default=1920, help='Width of the video frame for the canvas.')
    parser.add_argument('--video-height', type=int, default=1080, help='Height of the video frame for the canvas.')
    parser.add_argument('--show-info', action='store_true', help='Show Timestamp, UnixTime, and Activity on the frames.')
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
        yolo_df = pd.read_csv(args.yolo_csv_path)
        radar_df = pd.read_csv(args.radar_csv_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Rename radar timestamp column to match yolo's
    radar_df = radar_df.rename(columns={'timestamp': 'UnixTime'})

    # Ensure timestamp columns are numeric and sorted for merge_asof
    yolo_df['UnixTime'] = pd.to_numeric(yolo_df['UnixTime'], errors='coerce')
    radar_df['UnixTime'] = pd.to_numeric(radar_df['UnixTime'], errors='coerce')
    yolo_df.dropna(subset=['UnixTime'], inplace=True)
    radar_df.dropna(subset=['UnixTime'], inplace=True)
    
    yolo_df = yolo_df.sort_values('UnixTime')
    radar_df = radar_df.sort_values('UnixTime')

    # Use merge_asof to find the nearest yolo frame for each radar timestamp
    # A tolerance of 50ms (0.05s) is set. This assumes a frame rate of at least 20fps.
    # direction='nearest' finds the closest match in either direction.
    merged_df = pd.merge_asof(
        radar_df,
        yolo_df,
        on='UnixTime',
        direction='nearest',
        tolerance=0.05
    )

    # Drop radar events that didn't find a matching yolo frame
    merged_df.dropna(subset=['Timestamp'], inplace=True)

    if merged_df.empty:
        print("Error: No matching frames found within the tolerance. Please check timestamps or increase tolerance.")
        return

    print(f"Found {len(merged_df)} matching frames to render.")

    if args.person_id is not None:
        merged_df = merged_df[merged_df['PersonID'] == args.person_id]
        if merged_df.empty:
            print(f"No data found for person ID {args.person_id} after merging.")
            return

    keypoint_cols = [f'KP{i}_{axis}' for i in range(len(KEYPOINT_DICT)) for axis in ['X', 'Y']]
    if not all(col in merged_df.columns for col in keypoint_cols):
        print(f"Error: Merged dataframe is missing some keypoint columns.")
        return

    # Define a list of colors for different skeletons
    colors = [
        (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 128), (128, 128, 0), (0, 128, 128)
    ]

    # Group by the matched timestamp to draw all people in the same frame
    unique_timestamps = merged_df['UnixTime'].unique()
    for (uts, frame_group), frame_idx in tqdm(zip(merged_df.groupby('UnixTime'), range(len(unique_timestamps))), total=len(unique_timestamps), desc="Rendering frames"):
        frame = np.zeros((args.video_height, args.video_width, 3), dtype=np.uint8)

        # All rows in this group share the same activity from the radar data
        activity = frame_group.iloc[0]['activity']

        if args.show_info:
            timestamp = frame_group.iloc[0]['Timestamp']
            unixtime_val = frame_group.iloc[0]['UnixTime']
            cv2.putText(frame, f"Timestamp: {timestamp}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"UnixTime: {unixtime_val}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Activity: {activity}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        for _, person_row in frame_group.iterrows():
            keypoints = []
            for kp_idx in range(len(KEYPOINT_DICT)):
                x = person_row.get(f'KP{kp_idx}_X', 0)
                y = person_row.get(f'KP{kp_idx}_Y', 0)
                c = person_row.get(f'KP{kp_idx}_C', 0)
                keypoints.append((x, y, c))

            person_id = int(person_row.get('PersonID', 0))
            color = colors[person_id % len(colors)]

            draw_skeleton(frame, keypoints, SKELETON_CONNECTIONS, color, args.conf_threshold)

        output_path = os.path.join(args.output_dir, f"frame_{frame_idx:06d}.png")
        cv2.imwrite(output_path, frame)

    print(f"Rendered {len(merged_df['UnixTime'].unique())} frames to {args.output_dir}")


if __name__ == '__main__':
    main()
