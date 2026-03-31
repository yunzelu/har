import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
import csv
import time

from my_training import CNNTransformerModel, Config 

# Ensure YOLO can be imported
try:
    project_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(project_root))
    from yolo.yolo_skeleton import YOLOSkeleton
except ImportError as e:
    print(f"Error: {e}. Please ensure 'yolo/yolo_skeleton.py' exists.")
    sys.exit(1)

# Centralized configuration for Inference
class InferenceConfig(Config):
    def __init__(self):
        super().__init__() 
        
        # I/O Paths
        self.video_path = "WIN_20260326_15_06_23_Pro.mp4"  
        self.yolo_model_path = "yolo/yolo26n-pose.pt"
        self.output_csv_path = "frame_predictions.csv" 
        self.output_video_path = "WIN_20260326_15_06_23_Pro_RR.mp4" # NEW: Video output
        
        # YOLO Settings
        self.yolo_conf_threshold = 0.5
        
        # Video Dimensions
        self.frame_width = 1920
        self.frame_height = 1080

        self.training_output_dir = "2026-03-23_10-09-07"

def extract_keypoints_from_video(config):
    print(f"Loading YOLO model from {config.yolo_model_path}...")
    try:
        yolo = YOLOSkeleton(config.yolo_model_path)
    except Exception as e:
        print(f"Error initializing YOLO: {e}")
        return

    print(f"Opening video feed: {config.video_path}...")
    cap = cv2.VideoCapture(config.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {config.video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # We yield the original width/height so the VideoWriter knows exactly what to expect
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video loaded: {total_frames} frames @ {fps:.2f} FPS ({orig_width}x{orig_height})")

    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # START YOLO TIMER
        start_yolo = time.perf_counter()
        
        results = yolo.model.track(source=frame, persist=True, verbose=False, conf=config.yolo_conf_threshold)
        frame_results = results[0]
        
        # NEW: Render the YOLO skeleton directly onto the frame!
        annotated_frame = frame_results.plot()
        
        yolo_time = time.perf_counter() - start_yolo
        # END YOLO TIMER
        
        raw_keypoints = None
        raw_scores = None
        target_id = None
        
        if frame_results.boxes.id is not None:
            track_ids = frame_results.boxes.id.int().cpu().tolist()
            keypoints_xy = frame_results.keypoints.xy.cpu().numpy()
            scores = frame_results.keypoints.conf.cpu().numpy()
            
            target_index = 0 
            target_id = track_ids[target_index]
            raw_keypoints = keypoints_xy[target_index]
            raw_scores = scores[target_index] 
            
        yield {
            'frame_idx': frame_idx,
            'frame_image': annotated_frame, # We now pass the pre-drawn YOLO image
            'track_id': target_id,
            'keypoints': raw_keypoints, 
            'scores': raw_scores,
            'yolo_time': yolo_time,
            'orig_width': orig_width,
            'orig_height': orig_height,
            'fps': fps
        }
        frame_idx += 1

    cap.release()
    print("Video processing complete.")

class PosePreprocessor:
    # ... [Keep your exact same PosePreprocessor class here] ...
    def __init__(self, config):
        self.config = config
        self.frame_width = config.frame_width
        self.frame_height = config.frame_height
        
        self.left_hip_idx = 11
        self.right_hip_idx = 12
        
        self.last_valid_raw_kpts = np.zeros((17, 2))
        self.last_valid_raw_scores = np.zeros(17)
        self.prev_normalized_kpts = None
        self.prev_cleaned_scores = None

    def process_frame(self, raw_keypoints, raw_scores):
        if raw_keypoints is None or raw_scores is None:
            raw_keypoints = self.last_valid_raw_kpts.copy()
            raw_scores = self.last_valid_raw_scores.copy()
            
        cleaned_kpts = np.zeros((17, 2))
        cleaned_scores = np.zeros(17)
        
        for i in range(17):
            if raw_scores[i] < self.config.score_threshold:
                cleaned_kpts[i] = self.last_valid_raw_kpts[i]
                cleaned_scores[i] = self.last_valid_raw_scores[i]
            else:
                cleaned_kpts[i] = raw_keypoints[i]
                cleaned_scores[i] = raw_scores[i]
                self.last_valid_raw_kpts[i] = raw_keypoints[i]
                self.last_valid_raw_scores[i] = raw_scores[i]

        l_hip = cleaned_kpts[self.left_hip_idx]
        r_hip = cleaned_kpts[self.right_hip_idx]
        root_x = (l_hip[0] + r_hip[0]) / 2.0
        root_y = (l_hip[1] + r_hip[1]) / 2.0
        
        normalized_kpts = np.zeros((17, 2))
        for i in range(17):
            centered_x = cleaned_kpts[i][0] - root_x
            centered_y = cleaned_kpts[i][1] - root_y
            normalized_kpts[i][0] = centered_x / self.frame_width
            normalized_kpts[i][1] = centered_y / self.frame_height

        velocities = np.zeros((17, 2))
        if self.prev_normalized_kpts is not None and self.prev_cleaned_scores is not None:
            velocities = normalized_kpts - self.prev_normalized_kpts
            for i in range(17):
                if cleaned_scores[i] < self.config.score_threshold or self.prev_cleaned_scores[i] < self.config.score_threshold:
                    velocities[i] = [0.0, 0.0]
                    
        if not np.isfinite(velocities).all():
            velocities = np.zeros((17, 2))

        self.prev_normalized_kpts = normalized_kpts.copy()
        self.prev_cleaned_scores = cleaned_scores.copy()

        final_vector = np.concatenate([
            normalized_kpts.flatten(), 
            velocities.flatten(), 
            cleaned_scores
        ])
        return final_vector
    
class SlidingWindowBuffer:
    # ... [Keep your exact same SlidingWindowBuffer class here] ...
    def __init__(self, config):
        self.chunk_size = config.chunk_size 
        self.overlap = config.overlap 
        self.step_size = self.chunk_size - self.overlap
        
        self.buffer = []
        self.frame_indices = [] 

    def add_frame(self, feature_vector, frame_idx):
        self.buffer.append(feature_vector)
        self.frame_indices.append(frame_idx)
        
        if len(self.buffer) == self.chunk_size:
            chunk_to_process = np.array(self.buffer, dtype=np.float32)
            chunk_frames = list(self.frame_indices)
            
            self.buffer = self.buffer[self.step_size:]
            self.frame_indices = self.frame_indices[self.step_size:]
            
            return chunk_to_process, chunk_frames
            
        return None, None
    
def run_live_inference():
    config = InferenceConfig()
    
    saved_model_path = f"cnn_transformer/training_outputs/{config.training_output_dir}/cnn_transformer_model.pth"
    label_encoder_path = f"cnn_transformer/training_outputs/{config.training_output_dir}/label_encoder_classes.npy"
    
    class_names = np.load(label_encoder_path)
    config.num_classes = len(class_names)
    
    print("Loading CNN-Transformer Model...")
    model = CNNTransformerModel(config).to(config.device)
    model.load_state_dict(torch.load(saved_model_path, map_location=config.device))
    model.eval()
    
    preprocessor = PosePreprocessor(config=config)
    buffer = SlidingWindowBuffer(config)
    
    # Trackers for drawing and timing
    all_frame_predictions = {}
    frame_image_queue = [] # Small queue to hold images until inference is done
    video_writer = None
    
    # Timing statistics
    total_yolo_time = 0.0
    total_inf_time = 0.0
    frames_processed = 0
    
    print("Starting Video Stream & Inference...")
    
    with torch.no_grad():
        for frame_data in extract_keypoints_from_video(config):
            # 1. Initialize VideoWriter on the first frame
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(
                    config.output_video_path, fourcc, frame_data['fps'], 
                    (frame_data['orig_width'], frame_data['orig_height'])
                )
            
            current_frame_idx = frame_data['frame_idx']
            raw_kpts = frame_data['keypoints']
            raw_scores = frame_data['scores']
            img_to_draw = frame_data['frame_image']
            
            # Accumulate YOLO time
            total_yolo_time += frame_data['yolo_time']
            
            # Add image to our memory-safe queue
            frame_image_queue.append((current_frame_idx, img_to_draw))
            
            # START PREP+INFERENCE TIMER
            start_inf = time.perf_counter()
            
            feature_vector = preprocessor.process_frame(raw_kpts, raw_scores)
            chunk_tensor, chunk_frames = buffer.add_frame(feature_vector, current_frame_idx)
            
            if chunk_tensor is not None:
                inputs = torch.FloatTensor(chunk_tensor).unsqueeze(0).to(config.device)
                seq_outputs, frame_outputs = model(inputs)
                
                # --- NEW: GET OVERALL CHUNK PREDICTION ---
                seq_probs = F.softmax(seq_outputs, dim=1).squeeze()
                predicted_class_idx = torch.argmax(seq_probs).item()
                chunk_action = class_names[predicted_class_idx]
                chunk_confidence = seq_probs[predicted_class_idx].item()
                
                # --- PER-FRAME PREDICTIONS ---
                frame_probs = F.softmax(frame_outputs, dim=2).squeeze(1) 
                frame_class_indices = torch.argmax(frame_probs, dim=1).cpu().numpy()
                frame_confidences = torch.max(frame_probs, dim=1)[0].cpu().numpy()
                
                for i, f_idx in enumerate(chunk_frames):
                    # Save both chunk and frame data into the dictionary
                    all_frame_predictions[f_idx] = {
                        'chunk_action': chunk_action,
                        'chunk_conf': chunk_confidence,
                        'frame_action': class_names[frame_class_indices[i]],
                        'frame_conf': frame_confidences[i]
                    }
                    
                # END PREP+INFERENCE TIMER
                inf_time = time.perf_counter() - start_inf
                total_inf_time += inf_time
                
                # --- RENDER AND WRITE OUT FRAMES ---
                frames_to_write = buffer.step_size 
                
                for _ in range(frames_to_write):
                    if not frame_image_queue:
                        break
                        
                    q_idx, q_img = frame_image_queue.pop(0)
                    
                    # Look up the prediction (Default to Unknown)
                    pred = all_frame_predictions.get(q_idx, {
                        'chunk_action': 'Analyzing...', 'chunk_conf': 0.0,
                        'frame_action': 'Analyzing...', 'frame_conf': 0.0
                    })
                    
                    # --- NEW: DRAW BOTH LABELS ---
                    text_chunk = f"Chunk Action: {pred['chunk_action']} ({pred['chunk_conf']*100:.1f}%)"
                    text_frame = f"Frame Action: {pred['frame_action']} ({pred['frame_conf']*100:.1f}%)"
                    
                    # Taller black background box to fit two lines
                    cv2.rectangle(q_img, (10, 10), (550, 100), (0, 0, 0), -1)
                    
                    # Draw Chunk (Green) and Frame (Yellow) text
                    cv2.putText(q_img, text_chunk, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.putText(q_img, text_frame, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    
                    video_writer.write(q_img)
            else:
                inf_time = time.perf_counter() - start_inf
                total_inf_time += inf_time
                
            frames_processed += 1
            if frames_processed % 100 == 0:
                print(f"Processed {frames_processed} frames...")

    # --- END OF VIDEO CLEANUP ---
    print("Flushing remaining frames to video...")
    while frame_image_queue:
        q_idx, q_img = frame_image_queue.pop(0)
        pred = all_frame_predictions.get(q_idx, {
            'chunk_action': 'Finished', 'chunk_conf': 0.0,
            'frame_action': 'Finished', 'frame_conf': 0.0
        })
        
        text_chunk = f"Chunk Action: {pred['chunk_action']} ({pred['chunk_conf']*100:.1f}%)"
        text_frame = f"Frame Action: {pred['frame_action']} ({pred['frame_conf']*100:.1f}%)"
        
        cv2.rectangle(q_img, (10, 10), (550, 100), (0, 0, 0), -1)
        cv2.putText(q_img, text_chunk, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(q_img, text_frame, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        video_writer.write(q_img)

    if video_writer:
        video_writer.release()

    # --- SAVE CSV (Updated to include Chunk data) ---
    print(f"\nWriting predictions to {config.output_csv_path}...")
    with open(config.output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame_Index', 'Chunk_Action', 'Chunk_Confidence', 'Frame_Action', 'Frame_Confidence'])
        for f_idx in sorted(all_frame_predictions.keys()):
            writer.writerow([
                f_idx, 
                all_frame_predictions[f_idx]['chunk_action'],
                f"{all_frame_predictions[f_idx]['chunk_conf']:.4f}",
                all_frame_predictions[f_idx]['frame_action'],
                f"{all_frame_predictions[f_idx]['frame_conf']:.4f}"
            ])
            
    # --- PROFILING REPORT ---
    avg_yolo = (total_yolo_time / frames_processed) * 1000 # convert to ms
    avg_inf = (total_inf_time / frames_processed) * 1000
    total_avg = avg_yolo + avg_inf
    
    print("\n" + "="*40)
    print("⏱️ INFERENCE PROFILING REPORT")
    print("="*40)
    print(f"Total Frames: {frames_processed}")
    print(f"1. YOLO Tracking:      {avg_yolo:.2f} ms/frame")
    print(f"2. Norm + Inference:   {avg_inf:.2f} ms/frame")
    print(f"----------------------------------------")
    print(f"Total Pipeline Cost:   {total_avg:.2f} ms/frame")
    print(f"Estimated Max FPS:     {1000/total_avg:.1f} FPS")
    print("="*40)
    print(f"Video saved to: {config.output_video_path}")

if __name__ == "__main__":
    run_live_inference()