import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics.utils.plotting import Annotator, colors

def render_skeletons_from_csv(csv_path, video_out, view_size=(1920, 1080), fps=30):
    # 1. Read the CSV data
    df = pd.read_csv(csv_path)
    
    # Sort by UnixTime to ensure chronological frame rendering
    df = df.sort_values(by='UnixTime')
    
    # Group by timestamps to act as our frame sequences
    unique_timestamps = df[['Timestamp', 'UnixTime']].drop_duplicates()
    
    # Configuration
    w, h = view_size
    padding = 25  # Internal padding for the bounding box
    
    # 2. Setup VideoWriter
    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    print(f"Baking native YOLO-style render on black background to: {video_out}")
    print(f"Configured Canvas Size: {w}x{h}")

    # 3. Iterate through frames with a progress bar
    for _, row in tqdm(unique_timestamps.iterrows(), total=len(unique_timestamps), desc="Rendering frames"):
        timestamp = row['Timestamp']
        unix_time = row['UnixTime']
        
        # Create a blank black frame
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Initialize YOLO annotator
        annotator = Annotator(frame, line_width=2, example=str("person"))
        
        # Get all people present in this specific frame/timestamp
        people_in_frame = df[df['UnixTime'] == unix_time]
        
        for _, person in people_in_frame.iterrows():
            pid = int(person['PersonID'])
            
            # Extract keypoints into shape (17, 3) -> [x, y, conf]
            kpts_data = []
            for i in range(17):
                kpts_data.append([
                    person[f'KP{i}_X'],
                    person[f'KP{i}_Y'],
                    person[f'KP{i}_C']
                ])
                
            kpts_array = np.array(kpts_data, dtype=np.float32)
            
            # 4. Compute Bounding Box from valid keypoints
            # Filter out keypoints with very low confidence (e.g., < 0.2) to prevent box skewing
            valid_kpts = kpts_array[kpts_array[:, 2] > 0.2]
            
            if len(valid_kpts) > 0:
                min_x = np.min(valid_kpts[:, 0])
                min_y = np.min(valid_kpts[:, 1])
                max_x = np.max(valid_kpts[:, 0])
                max_y = np.max(valid_kpts[:, 1])
                
                # Apply padding and clamp to the configured canvas boundaries
                box = [
                    max(0, int(min_x - padding)),
                    max(0, int(min_y - padding)),
                    min(w, int(max_x + padding)),
                    min(h, int(max_y + padding))
                ]
                
                # Draw the generated Bounding Box and ID
                annotator.box_label(box, f"ID: {pid}", color=colors(pid, True))
            
            # 5. Draw the native YOLO skeleton
            # Convert keypoints to tensor for the annotator
            kpts_tensor = torch.tensor(kpts_array)
            annotator.kpts(kpts_tensor, shape=(h, w), kpt_line=True)
            
        # Get the annotated frame
        final_frame = annotator.result()
        
        # 6. Add Timestamp and UnixTime to the top-left corner
        text_str = f"Time: {timestamp} | Unix: {unix_time}"
        cv2.putText(final_frame, text_str, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.0, (255, 255, 255), 2, cv2.LINE_AA)
        
        out.write(final_frame)
        
    out.release()
    print("Video saved successfully.")

if __name__ == "__main__":
    render_skeletons_from_csv(
        csv_path="data/Elaine/12-12.csv",
        video_out="data/Elaine/12-12.mp4",
        view_size=(1920, 1080),
        fps=30
    )