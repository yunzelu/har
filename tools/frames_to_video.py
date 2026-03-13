"""
This script combines a sequence of image frames into a video file.
"""

import cv2
import os
import argparse
from tqdm import tqdm

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Create a video from a directory of frames.")
    parser.add_argument('--frames-dir', type=str, required=True, help='Directory containing the image frames.')
    parser.add_argument('--output-video-path', type=str, required=True, help='Path to save the output video file (e.g., output.mp4).')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for the output video.')
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.isdir(args.frames_dir):
        print(f"Error: The directory {args.frames_dir} does not exist.")
        return

    # Get all image files from the directory
    frame_files = [f for f in os.listdir(args.frames_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not frame_files:
        print(f"No image frames found in {args.frames_dir}.")
        return

    # Sort the files numerically based on the frame number in the filename
    try:
        frame_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    except ValueError:
        print("Warning: Could not sort frames numerically. Using alphanumeric sort.")
        frame_files.sort()

    # Read the first frame to get the video dimensions
    first_frame_path = os.path.join(args.frames_dir, frame_files[0])
    frame = cv2.imread(first_frame_path)
    if frame is None:
        print(f"Error: Could not read the first frame: {first_frame_path}")
        return
    height, width, layers = frame.shape

    # Initialize the VideoWriter
    # Using 'mp4v' codec for .mp4 files. For .avi, you might use 'XVID'.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(args.output_video_path, fourcc, args.fps, (width, height))

    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for path {args.output_video_path}")
        return

    print(f"Creating video '{args.output_video_path}' from {len(frame_files)} frames...")

    # Write each frame to the video
    for filename in tqdm(frame_files, desc=f"Creating video '{args.output_video_path}'"):
        frame_path = os.path.join(args.frames_dir, filename)
        img = cv2.imread(frame_path)
        if img is not None:
            video_writer.write(img)

    # Release the video writer
    video_writer.release()
    print("Video creation complete.")

if __name__ == '__main__':
    main()
