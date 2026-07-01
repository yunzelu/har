import argparse
from pathlib import Path
from datetime import datetime


def render_skeletons_from_csv(
    csv_path,
    image_path,
    video_out,
    view_size=(640, 360),
    fps=30,
    source_size=None,
    max_frames=1000,
):
    import cv2
    import torch
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from ultralytics.utils.plotting import Annotator, colors

    # 1. Read the CSV data
    df = pd.read_csv(csv_path)
    
    # Check if the new bounding box columns exist in this CSV
    bbox_cols = ['BBox_X1', 'BBox_Y1', 'BBox_X2', 'BBox_Y2']
    has_bbox_cols = all(col in df.columns for col in bbox_cols)
    
    # Sort by UnixTime to ensure chronological frame rendering
    if 'UnixTime' in df.columns:
        df = df.sort_values(by='UnixTime')
    
    # Configuration
    w, h = view_size
    padding = 0  # Internal padding for the bounding box

    bg_original = cv2.imread(str(image_path))
    if bg_original is None:
        raise FileNotFoundError(f"Could not load background image at {image_path}")

    if source_size is None:
        source_w, source_h = bg_original.shape[1], bg_original.shape[0]
    else:
        source_w, source_h = source_size

    if source_w <= 0 or source_h <= 0:
        raise ValueError("Source width and height must be greater than zero.")

    scale_x = w / source_w
    scale_y = h / source_h

    # Group by timestamps to act as our frame sequences.
    if 'UnixTime' in df.columns:
        frame_key = 'UnixTime'
    elif 'Timestamp' in df.columns:
        frame_key = 'Timestamp'
    else:
        frame_key = '_FrameIndex'
        df = df.copy()
        df[frame_key] = np.arange(len(df))

    frame_groups = list(df.groupby(frame_key, sort=False))
    if max_frames and max_frames > 0:
        frame_groups = frame_groups[:max_frames]
    
    # Resize image to match the video dimensions
    bg_resized = cv2.resize(bg_original, (w, h))
    
    # 2. Setup VideoWriter
    video_out = Path(video_out)
    video_out.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if not out.isOpened():
        raise RuntimeError(f"Could not open video writer for {video_out}")
    
    print(f"Baking native YOLO-style render on black background to: {video_out}")
    print(f"Configured Canvas Size: {w}x{h}")
    print(f"Coordinate Source Size: {source_w}x{source_h}")
    if max_frames and max_frames > 0:
        print(f"Frame Cap: {max_frames}")

    # 3. Iterate through frames with a progress bar
    for _, people_in_frame in tqdm(frame_groups, total=len(frame_groups), desc="Rendering frames"):
        # Safely grab the dual timestamps
        first_row = people_in_frame.iloc[0]
        # timestamp = datetime.fromtimestamp(first_row.get('Timestamp', ''))
        timestamp = first_row.get('Timestamp', '')
        unix_time = first_row.get('UnixTime', 0.0)
        
        # Create a blank black frame
        frame = bg_resized.copy()
        
        # Initialize YOLO annotator
        annotator = Annotator(frame, line_width=2, example=str("person"))
        
        for _, person in people_in_frame.iterrows():
            pid = int(person.get('ID', person.get('PersonID', 0)))
            
            # Extract keypoints into shape (17, 3) -> [x, y, conf]
            kpts_data = []
            for i in range(17):
                kpts_data.append([
                    person[f'KP{i}_X'],
                    person[f'KP{i}_Y'],
                    person[f'KP{i}_C']
                ])
                
            kpts_array = np.array(kpts_data, dtype=np.float32)
            kpts_array[:, 0] *= scale_x
            kpts_array[:, 1] *= scale_y
            
            # 4. Compute or Read Bounding Box
            box = None
            
            if has_bbox_cols and not pd.isna(person['BBox_X1']):
                x1 = int(person['BBox_X1'] * scale_x)
                y1 = int(person['BBox_Y1'] * scale_y)
                x2 = int(person['BBox_X2'] * scale_x)
                y2 = int(person['BBox_Y2'] * scale_y)
                box = [
                    max(0, min(w, x1)),
                    max(0, min(h, y1)),
                    max(0, min(w, x2)),
                    max(0, min(h, y2)),
                ]
            else:
                valid_kpts = kpts_array[kpts_array[:, 2] > 0.2]
                
                if len(valid_kpts) > 0:
                    min_x = np.min(valid_kpts[:, 0])
                    min_y = np.min(valid_kpts[:, 1])
                    max_x = np.max(valid_kpts[:, 0])
                    max_y = np.max(valid_kpts[:, 1])
                    
                    box = [
                        max(0, int(min_x - padding)),
                        max(0, int(min_y - padding)),
                        min(w, int(max_x + padding)),
                        min(h, int(max_y + padding))
                    ]
            
            if box is not None and pid != -1:
                annotator.box_label(box, f"ID: {pid}", color=colors(pid, True))
            
            # 5. Draw the native YOLO skeleton
            kpts_tensor = torch.tensor(kpts_array)
            annotator.kpts(kpts_tensor, shape=(h, w), kpt_line=True)
            
        # Get the annotated frame
        final_frame = annotator.result()
        
        # 6. Add Timestamp and UnixTime to the bottom-left corner
        text_str = f"Time: {timestamp} | Unix: {unix_time}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5   # Scaled down from 1.0
        thickness = 1      # Scaled down from 2

        # Set a 15-pixel margin from the bottom and left edges
        margin = 15
        text_x = margin
        text_y = h - margin
        
        cv2.putText(final_frame, text_str, (text_x, text_y), font, 
                    font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
        
        out.write(final_frame)
        
    out.release()
    print("Video saved successfully.")


def parse_args():
    parser = argparse.ArgumentParser(description="Render YOLO Skeletons without Inference")
    csv_input = parser.add_mutually_exclusive_group(required=True)
    csv_input.add_argument('--csv-path', type=str, help="Path to a single input CSV file")
    csv_input.add_argument('--csv-dir', type=str, help="Path to a folder containing input CSV files")
    parser.add_argument('--image-path', type=str, required=True, help="Path to the background layout image")
    parser.add_argument('--video-out', type=str, help="Path to save the output MP4 video when using --csv-path")
    parser.add_argument('--output-dir', type=str, help="Folder to save output MP4 videos when using --csv-dir")
    parser.add_argument('--width', type=int, default=640, help="Canvas width (default: 640)")
    parser.add_argument('--height', type=int, default=360, help="Canvas height (default: 360)")
    parser.add_argument('--source-width', type=int, help="Original coordinate width for CSV keypoints/boxes. Defaults to background image width.")
    parser.add_argument('--source-height', type=int, help="Original coordinate height for CSV keypoints/boxes. Defaults to background image height.")
    parser.add_argument('--max-frames', type=int, default=1000, help="Maximum frames to render per CSV. Use 0 to render all frames. (default: 1000)")
    parser.add_argument('--fps', type=int, default=30, help="Video frames per second (default: 30)")
    args = parser.parse_args()

    if (args.source_width is None) != (args.source_height is None):
        parser.error("--source-width and --source-height must be provided together.")
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be greater than zero.")
    if args.fps <= 0:
        parser.error("--fps must be greater than zero.")
    if args.csv_dir and args.video_out:
        parser.error("--video-out can only be used with --csv-path. Use --output-dir with --csv-dir.")
    if args.max_frames < 0:
        parser.error("--max-frames must be 0 or a positive integer.")
    if args.source_width is not None and (args.source_width <= 0 or args.source_height <= 0):
        parser.error("--source-width and --source-height must be greater than zero.")

    return args


def build_render_jobs(args):
    if args.csv_path:
        csv_path = Path(args.csv_path)
        video_out = Path(args.video_out) if args.video_out else csv_path.with_name(f"{csv_path.stem}_skeleton.mp4")
        return [(csv_path, video_out)]

    csv_dir = Path(args.csv_dir)
    csv_paths = sorted(csv_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else csv_dir / "rendered_videos"
    return [(csv_path, output_dir / f"{csv_path.stem}.mp4") for csv_path in csv_paths]

if __name__ == "__main__":
    args = parse_args()
    source_size = None
    if args.source_width is not None:
        source_size = (args.source_width, args.source_height)

    for csv_path, video_out in build_render_jobs(args):
        print(f"\nProcessing CSV: {csv_path}")
        render_skeletons_from_csv(
            csv_path=csv_path,
            image_path=args.image_path,
            video_out=video_out,
            view_size=(args.width, args.height),
            fps=args.fps,
            source_size=source_size,
            max_frames=args.max_frames,
        )
