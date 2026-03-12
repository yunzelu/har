"""
This script is for activity recognition using a CNN-Transformer model with keypoint data from a CSV file.
"""

import json
import numpy as np
import pandas as pd
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import argparse
import os

# Import model architecture from the training script
from cnn_transformer_train import Config, SpatialAttention, PositionalEncoding, CNNTransformerModel

def normalize_keypoints(keypoints, width, height):
    """
    Normalize keypoints to a [0, 1] range based on video dimensions.
    """
    keypoints_copy = keypoints.copy()
    # X coordinates (even indices)
    keypoints_copy[0::2] = keypoints_copy[0::2] / width
    # Y coordinates (odd indices)
    keypoints_copy[1::2] = keypoints_copy[1::2] / height
    return keypoints_copy

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Inference with CNN-Transformer model using CSV input.")
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model weights (.pth file)')
    parser.add_argument('--label-encoder', type=str, required=True, help='Path to the label encoder classes (.npy file)')
    parser.add_argument('--csv-path', type=str, required=True, help='Path to the input CSV file with keypoints')
    parser.add_argument('--person-id', type=int, default=1, help='ID of the person to perform inference on')
    parser.add_argument('--video-width', type=int, required=True, help='Width of the original video in pixels')
    parser.add_argument('--video-height', type=int, required=True, help='Height of the original video in pixels')
    parser.add_argument('--smooth-window', type=int, default=5, help='Window size for temporal smoothing of predictions')
    parser.add_argument('--output-path', type=str, default='cnn_transformer/activity_results_csv.json', help='Path to save the JSON output file')
    return parser.parse_args()

def main():
    args = parse_args()

    # Verify that required files exist
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    if not os.path.exists(args.label_encoder):
        raise FileNotFoundError(f"Label encoder file not found: {args.label_encoder}")
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")

    # Load label encoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(args.label_encoder, allow_pickle=True)

    # Initialize model configuration
    config = Config()
    config.num_classes = len(label_encoder.classes_)
    device = config.device

    # Load the pretrained model
    model = CNNTransformerModel(config).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    print("Model loaded successfully!")
    print(f"Activity classes: {list(label_encoder.classes_)}")

    # Load and process CSV data
    df = pd.read_csv(args.csv_path)
    person_df = df[df['ID'] == args.person_id]

    if person_df.empty:
        print(f"No data found for person ID: {args.person_id}")
        return

    keypoint_columns = [f'KP{i}_{axis}' for i in range(17) for axis in ['X', 'Y']]
    person_keypoints = person_df[keypoint_columns].values

    # Normalize keypoints and prepare sequences
    sequence_buffer = []
    prev_kps_flat = None
    
    for i in range(len(person_keypoints)):
        current_kps_flat = person_keypoints[i]

        # Normalize keypoints
        current_kps_flat = normalize_keypoints(current_kps_flat, args.video_width, args.video_height)

        # Calculate velocity
        if prev_kps_flat is None:
            velocity = np.zeros_like(current_kps_flat)
        else:
            velocity = current_kps_flat - prev_kps_flat
        
        if not np.isfinite(velocity).all():
            velocity = np.zeros_like(current_kps_flat)

        kp_with_velocity = np.concatenate([current_kps_flat, velocity])
        sequence_buffer.append(kp_with_velocity)
        prev_kps_flat = current_kps_flat.copy()

    # Inference using a sliding window
    sequence_length = config.chunk_size
    overlap = config.overlap
    results = []
    recent_predictions = []
    
    print(f"Processing data with chunk size {sequence_length} and overlap {overlap}")

    for i in range(0, len(sequence_buffer) - sequence_length + 1, sequence_length - overlap):
        seq = np.array(sequence_buffer[i:i + sequence_length])

        if np.isfinite(seq).all():
            with torch.no_grad():
                inputs = torch.FloatTensor(seq).unsqueeze(0).to(device)
                outputs = model(inputs)
                _, pred = torch.max(outputs[0], 1)
                pred_class = pred.item()

                recent_predictions.append(pred_class)
                if len(recent_predictions) > args.smooth_window:
                    recent_predictions.pop(0)

                if recent_predictions:
                    smoothed_pred = Counter(recent_predictions).most_common(1)[0][0]
                    activity = label_encoder.inverse_transform([smoothed_pred])[0]
                    confidence = torch.softmax(outputs[0], dim=1)[0][pred].item()

                    start_frame = person_df.index[i]
                    end_frame = person_df.index[i + sequence_length - 1]

                    results.append({
                        'frames': f"{start_frame}-{end_frame}",
                        'activity': activity,
                        'confidence': float(confidence),
                        'raw_prediction': int(pred_class),
                        'smoothed_prediction': int(smoothed_pred)
                    })
                    print(f"Frames {start_frame}-{end_frame}: {activity} (conf: {confidence:.2f})")
        else:
            print(f"Skipping prediction for frames {person_df.index[i]}-{person_df.index[i + sequence_length -1]} due to invalid values.")

    # Save results
    if results:
        print(f"Saving {len(results)} predictions to {args.output_path}")
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, 'w') as f:
            json.dump({
                'model_path': args.model_path,
                'csv_path': args.csv_path,
                'person_id': args.person_id,
                'sequence_length': sequence_length,
                'overlap': overlap,
                'classes': list(label_encoder.classes_),
                'results': results
            }, f, indent=2)
        print(f"Results saved to {args.output_path}")
    else:
        print("No activity predictions were generated.")

if __name__ == '__main__':
    main()