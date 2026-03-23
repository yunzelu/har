import cv2
import csv
import os

class RenderConfig:
    def __init__(self):
        self.input_video_path = "test_video.mp4"
        self.input_csv_path = "frame_predictions.csv"
        self.output_video_path = "test_video_rendered.mp4"
        
        # Rendering settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1.0
        self.thickness = 2
        
        # Colors (B, G, R)
        self.color_chunk = (0, 255, 0)   # Green for the smoothed chunk prediction
        self.color_frame = (0, 255, 255) # Yellow for the exact frame prediction
        self.color_bg = (0, 0, 0)        # Black background for text

def load_predictions(csv_path):
    """Loads the CSV into a dictionary mapped by frame index."""
    predictions = {}
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return predictions
        
    with open(csv_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            frame_idx = int(row['Frame_Index'])
            predictions[frame_idx] = {
                'chunk_action': row['Chunk_Smoothed_Action'],
                'frame_action': row['Exact_Frame_Action']
            }
    return predictions

def render_video(config):
    print(f"Loading predictions from {config.input_csv_path}...")
    predictions = load_predictions(config.input_csv_path)
    
    print(f"Opening video {config.input_video_path}...")
    cap = cv2.VideoCapture(config.input_video_path)
    
    if not cap.isOpened():
        print("Error: Could not open input video.")
        return

    # Get video properties for the writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(config.output_video_path, fourcc, fps, (width, height))
    
    print(f"Rendering {total_frames} frames to {config.output_video_path}...")
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Get predictions for this frame (default to "Unknown" if not in CSV)
        preds = predictions.get(frame_idx, {'chunk_action': 'Unknown', 'frame_action': 'Unknown'})
        
        text_chunk = f"Smoothed Action: {preds['chunk_action']}"
        text_frame = f"Exact Frame Action: {preds['frame_action']}"
        
        # Draw background rectangle for readability
        cv2.rectangle(frame, (10, 10), (550, 100), config.color_bg, -1)
        
        # Draw text
        cv2.putText(frame, text_chunk, (20, 45), config.font, config.font_scale, config.color_chunk, config.thickness)
        cv2.putText(frame, text_frame, (20, 85), config.font, config.font_scale, config.color_frame, config.thickness)
        
        out.write(frame)
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} / {total_frames} frames...")

    cap.release()
    out.release()
    print("Rendering complete!")

if __name__ == "__main__":
    config = RenderConfig()
    render_video(config)