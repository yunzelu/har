import cv2
import json
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

# The 17 keypoints from COCO dataset
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Connections between keypoints
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

# Colors for the keypoints and connections
KEYPOINT_COLOR = (0, 255, 0)  # Green
CONNECTION_COLOR = (255, 0, 0) # Blue

def render_video_from_json(
    json_path: Path, 
    output_path: Path, 
    width: int, 
    height: int, 
    fps: int, 
    score_threshold: float
):
    """
    Renders a video from a JSON file containing keypoint data.

    Args:
        json_path: Path to the input JSON file.
        output_path: Path to save the output MP4 video.
        width: Width of the output video.
        height: Height of the output video.
        fps: Frames per second of the output video.
        score_threshold: Keypoints with a score below this threshold will not be rendered.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not video_writer.isOpened():
        raise IOError(f"Could not open video writer for path: {output_path}")

    for frame_data in tqdm(data, desc=f"Rendering to {output_path}"):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        if frame_data:
            keypoints = np.array(frame_data['keypoints'])
            scores = np.array(frame_data['scores'])

            # Draw keypoints
            for i, (point, score) in enumerate(zip(keypoints, scores)):
                if score >= score_threshold:
                    x, y = int(point[0]), int(point[1])
                    if 0 <= x < width and 0 <= y < height:
                        cv2.circle(frame, (x, y), 5, KEYPOINT_COLOR, -1)

            # Draw connections
            for p1_idx, p2_idx in SKELETON_CONNECTIONS:
                if scores[p1_idx] >= score_threshold and scores[p2_idx] >= score_threshold:
                    p1 = (int(keypoints[p1_idx][0]), int(keypoints[p1_idx][1]))
                    p2 = (int(keypoints[p2_idx][0]), int(keypoints[p2_idx][1]))
                    
                    if (0 <= p1[0] < width and 0 <= p1[1] < height and
                        0 <= p2[0] < width and 0 <= p2[1] < height):
                        cv2.line(frame, p1, p2, CONNECTION_COLOR, 2)
        
        video_writer.write(frame)

    video_writer.release()
    print(f"Video rendering complete. Saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Render a video from a JSON file containing keypoints.",
        epilog="Example: python tools/dataset/render_json_to_video.py combined.json output.mp4"
    )
    parser.add_argument('json_path', type=str, help='Path to the input JSON file.')
    parser.add_argument('output_path', type=str, help='Path to the output MP4 video file.')
    parser.add_argument('--width', type=int, default=640, help='Width of the output video.')
    parser.add_argument('--height', type=int, default=480, help='Height of the output video.')
    parser.add_argument('--fps', type=int, default=20, help='Frames per second of the output video.')
    parser.add_argument('--threshold', type=float, default=0.25, help='Score threshold to show a keypoint.')
    
    args = parser.parse_args()

    json_path = Path(args.json_path)
    output_path = Path(args.output_path)

    if not json_path.is_file():
        print(f"Error: Input JSON file not found at '{json_path}'")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        render_video_from_json(
            json_path, 
            output_path, 
            args.width, 
            args.height, 
            args.fps, 
            args.threshold
        )
    except Exception as e:
        print(f"An error occurred during rendering: {e}")

if __name__ == '__main__':
    main()
