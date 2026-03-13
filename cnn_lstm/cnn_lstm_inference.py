'''
Activity Recognition Inference using CNN-LSTM Model

This script performs inference on video data to recognize human activities
using a pre-trained CNN-LSTM model. It extracts pose keypoints using MMPose
and feeds them to the trained model for activity classification.
'''

# Import necessary libraries
from argparse import ArgumentParser
from typing import Dict
from mmpose.apis.inferencers import MMPoseInferencer
import json
import numpy as np
from tqdm import tqdm  # For progress bar
import os
import cv2
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from tensorflow.keras.mixed_precision import Policy  # Import Policy class
from tensorflow.keras.layers import InputLayer

# Default filtering arguments for pose estimation
filter_args = dict(bbox_thr=0.3, nms_thr=0.3, pose_based_nms=False)

# Model-specific filtering arguments to optimize detection
POSE2D_SPECIFIC_ARGS = dict(
    yoloxpose=dict(bbox_thr=0.01, nms_thr=0.65, pose_based_nms=True),
    rtmo=dict(bbox_thr=0.1, nms_thr=0.65, pose_based_nms=True),
)

# Function to normalize keypoints based on image dimensions
def normalize_keypoints(keypoints, width, height):
    keypoints_copy = keypoints.copy()
    # X coordinates (even indices)
    keypoints_copy[0::2] = keypoints_copy[0::2] / width
    # Y coordinates (odd indices)
    keypoints_copy[1::2] = keypoints_copy[1::2] / height
    return keypoints_copy

# Function to parse command-line arguments
def parse_args():
    parser = ArgumentParser()

    # Activity recognition specific parameters
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.3,
        help='Threshold for keypoint confidence')
        
    parser.add_argument(
        '--smooth-window',
        type=int,
        default=5,
        help='Window size for temporal smoothing of predictions')

    parser.add_argument(
        '--model-path', 
        type=str, 
        required=True,
        help='Path to trained model weights')
    
    parser.add_argument(
        '--label-encoder', 
        type=str, 
        required=True,
        help='Path to label encoder classes')
    
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=30,
        help='Sequence length for activity recognition (must match training)')

    # Input path for image/video or folder
    parser.add_argument(
        'inputs',
        type=str,
        nargs='?',
        help='Input image/video path or folder path.')

    # init args for MMPose
    parser.add_argument(
        '--pose2d',
        type=str,
        default=None,
        help='Pretrained 2D pose estimation algorithm. It\'s the path to the '
        'config file or the model name defined in metafile.')
    parser.add_argument(
        '--pose2d-weights',
        type=str,
        default=None,
        help='Path to the custom checkpoint file of the selected pose model.')

    parser.add_argument(
        '--pose3d',
        type=str,
        default=None,
        help='Pretrained 3D pose estimation algorithm.')
    parser.add_argument(
        '--pose3d-weights',
        type=str,
        default=None,
        help='Path to the custom checkpoint file of the selected pose model.')

    # Arguments for person detection model
    parser.add_argument(
        '--det-model',
        type=str,
        default=None,
        help='Config path or alias of detection model.')
    parser.add_argument(
        '--det-weights',
        type=str,
        default=None,
        help='Path to the checkpoints of detection model.')
    parser.add_argument(
        '--det-cat-ids',
        type=int,
        nargs='+',
        default=0,
        help='Category id for detection model.')

    # Other general arguments for MMPose
    parser.add_argument(
        '--scope',
        type=str,
        default='mmpose',
        help='Scope where modules are defined.')
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device used for inference.')
    parser.add_argument(
        '--show-progress',
        action='store_true',
        help='Display the progress bar during inference.')

    # The default arguments for prediction filtering
    args, _ = parser.parse_known_args()
    for model in POSE2D_SPECIFIC_ARGS:
        if args.pose2d is not None and model in args.pose2d:
            filter_args.update(POSE2D_SPECIFIC_ARGS[model])
            break

    # Visualization and output control arguments
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the image/video in a popup window.')
    parser.add_argument(
        '--draw-bbox',
        action='store_true',
        help='Whether to draw the bounding boxes.')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Whether to draw the predicted heatmaps.')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=filter_args['bbox_thr'],
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=filter_args['nms_thr'],
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--pose-based-nms',
        type=lambda arg: arg.lower() in ('true', 'yes', 't', 'y', '1'),
        default=filter_args['pose_based_nms'],
        help='Whether to use pose-based NMS')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--use-oks-tracking',
        action='store_true',
        help='Whether to use OKS as similarity in tracking')
    parser.add_argument(
        '--vis-out-dir',
        type=str,
        default='',
        help='Directory for saving visualized results.')
    parser.add_argument(
        '--pred-out-dir',
        type=str,
        default='',
        help='Directory for saving inference results.')

    # Parse arguments and separate initialization and call arguments
    call_args = vars(parser.parse_args())

    # Extract custom arguments for activity recognition
    custom_args = {
        'confidence_threshold': call_args.pop('confidence_threshold'),
        'smooth_window': call_args.pop('smooth_window'),
        'model_path': call_args.pop('model_path'),
        'label_encoder': call_args.pop('label_encoder'),
        'sequence_length': call_args.pop('sequence_length')
    }

    # Separate initialization arguments for MMPose
    init_kws = [
        'pose2d', 'pose2d_weights', 'scope', 'device', 'det_model',
        'det_weights', 'det_cat_ids', 'pose3d', 'pose3d_weights',
        'show_progress'
    ]
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    # Combine custom args with MMPose init args
    init_args.update(custom_args)

    return init_args, call_args

# 1. Custom InputLayer to fix 'batch_shape' error
class CustomInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        super().__init__(*args, **kwargs)

# 2. Alias DTypePolicy to the current Policy class
DTypePolicy = Policy  # Map saved 'DTypePolicy' to current Policy

def main():
    # Parse all arguments for pose estimation and activity recognition
    init_args, call_args = parse_args()
    
    # Extract activity recognition specific parameters
    confidence_threshold = init_args.pop('confidence_threshold')
    smooth_window = init_args.pop('smooth_window')
    sequence_length = init_args.pop('sequence_length')
    
    # Verify that required files exist
    if not os.path.exists(init_args['model_path']):
        raise FileNotFoundError(f"Model file not found: {init_args['model_path']}")
    if not os.path.exists(init_args['label_encoder']):
        raise FileNotFoundError(f"Label encoder file not found: {init_args['label_encoder']}")
    
    # Load label encoder with class names for activity recognition
    # Instead of loading a saved numpy file, let's handle activities directly
    # This avoids numpy version compatibility issues
    try:
        # First try to load the file if it's a simple text file with activity names
        with open(init_args['label_encoder'], 'r') as f:
            activities = [line.strip() for line in f.readlines()]
            label_encoder = LabelEncoder()
            label_encoder.fit(activities)
    except:
        # Fallback to hardcoded activities from the training script
        print("Warning: Could not load label encoder file, using default activities")
        activities = ['Clapping', 'Sitting', 'Standing Still', 
                      'Walking While Reading Book', 'Walking While Using Phone', 'Walking']
        label_encoder = LabelEncoder()
        label_encoder.fit(activities)
    
    # Load the pretrained CNN-LSTM model with custom objects
    model = tf.keras.models.load_model(
        init_args['model_path'],
        custom_objects={
            'InputLayer': CustomInputLayer,  # Fixes InputLayer error
            'DTypePolicy': DTypePolicy       # Fixes DTypePolicy error
        },
        compile=False  # Skip loading optimizer state
    )
    print("Model loaded successfully!")
    print(f"Activity classes: {list(label_encoder.classes_)}")
    
    # Get input video/image path
    video_path = call_args.get('inputs', '')
    
    # Get video or image metadata (dimensions and total frames)
    width, height, total_frames = 0, 0, 0
    if video_path.endswith(('.mp4', '.avi')):
        # Video case - get dimensions and frame count
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    elif os.path.isfile(video_path):  # Single image case
        img = cv2.imread(video_path)
        if img is not None:
            height, width, _ = img.shape
            total_frames = 1
    else:
        raise ValueError("Invalid input path")
    
    # Create a copy of init_args without activity recognition arguments
    # to pass only pose estimation related parameters to MMPose
    pose_init_args = init_args.copy()
    pose_init_args.pop('model_path', None)
    pose_init_args.pop('label_encoder', None)
    
    # Initialize MMPose inferencer with only pose estimation arguments
    inferencer = MMPoseInferencer(**pose_init_args)
    
    # Buffer to store keypoint sequences for activity recognition
    sequence_buffer = []
    overlap = sequence_length // 4  # Use 25% overlap between sequences
    prev_kps_flat = None  # Track previous frame's keypoints
    results = []  # Store all activity recognition results
    recent_predictions = []  # Buffer for temporal smoothing of predictions
    prediction_history_size = smooth_window  # Size of window for majority voting
    
    # Set up output directory and file for saving prediction results
    if call_args.get('pred_out_dir', ''):
        os.makedirs(call_args.get('pred_out_dir', ''), exist_ok=True)
        output_path = os.path.join(call_args.get('pred_out_dir', ''), 'activity_results.json')
    else:
        output_path = 'cnn_lstm/activity_results.json'  # Default output path
    
    # Process frames with a progress bar
    print(f"Processing video with sequence length {sequence_length} and overlap {overlap}")
    with tqdm(total=total_frames, desc="Processing") as pbar:
        # Iterate through frames processed by the pose estimator
        for result in inferencer(**call_args):
            # Check if pose predictions are available for current frame
            if 'predictions' in result and result['predictions']:
                frame_keypoints = []  # Store keypoints for current frame
                
                # Extract and process keypoints (focus on first person only)
                if result['predictions'][0]:  # Check if any detections in this frame
                    # Get keypoint confidences
                    keypoint_confidences = np.array(result['predictions'][0][0]['keypoint_scores'])
                    
                    # Only process keypoints if average confidence is above threshold
                    if np.mean(keypoint_confidences) >= confidence_threshold:
                        # Extract keypoint coordinates (17 joints, x and y)
                        current_kps = np.array(result['predictions'][0][0]['keypoints'])[:,:2]  # (17, 2)
                        current_kps_flat = current_kps.flatten()  # Flatten to 1D array (34,)
                        
                        # Normalize keypoints to [0,1] range
                        current_kps_flat = normalize_keypoints(current_kps_flat, width, height)
                        frame_keypoints.append(current_kps_flat)
                        
                        # Store current keypoints for next frame
                        prev_kps_flat = current_kps_flat.copy()
                
                # Decide what to add to the sequence buffer
                if frame_keypoints:
                    # If keypoints detected, add them
                    sequence_buffer.append(frame_keypoints[0])
                else:
                    # If no detection in current frame, use last known keypoints or zeros
                    if prev_kps_flat is not None:
                        sequence_buffer.append(prev_kps_flat)
                    else:
                        # If no previous keypoints either, add zeros (empty frame)
                        sequence_buffer.append(np.zeros(34))  # 17 keypoints × 2 coordinates
                
                # Process a complete sequence when buffer reaches required length
                if len(sequence_buffer) >= sequence_length:
                    # Extract exactly sequence_length frames for processing
                    seq = np.array(sequence_buffer[:sequence_length])
                    
                    # Sanity check - ensure no NaN or inf values
                    if np.isfinite(seq).all():
                        # Prepare input tensor - reshape according to CNN-LSTM model
                        # The model expects a flattened input as in the training script
                        # Based on your training code, input shape is (batch_size, sequence_length * feature_dim)
                        # where feature_dim is 34 (17 keypoints * 2 coordinates)
                        seq_reshaped = seq.reshape(1, len(seq) * len(seq[0]))  # Flatten for CNN input
                        
                        # Run activity recognition model inference
                        predictions = model.predict(seq_reshaped, verbose=0)
                        pred_class = np.argmax(predictions[0])
                            
                        # Apply temporal smoothing to stabilize predictions
                        recent_predictions.append(pred_class)
                        if len(recent_predictions) > prediction_history_size:
                            recent_predictions.pop(0)  # Remove oldest prediction
                        
                        # Get most common prediction in the temporal window (majority voting)
                        if recent_predictions:
                            smoothed_pred = Counter(recent_predictions).most_common(1)[0][0]
                            # Convert numerical class to human-readable label
                            activity = label_encoder.inverse_transform([smoothed_pred])[0]
                            # Calculate prediction confidence
                            confidence = float(predictions[0][pred_class])
                            
                            # Store results with frame range information
                            start_frame = max(0, pbar.n - sequence_length + 1)
                            results.append({
                                'frames': f"{start_frame}-{pbar.n}",
                                'activity': activity,
                                'confidence': float(confidence),
                                'raw_prediction': int(pred_class),
                                'smoothed_prediction': int(smoothed_pred)
                            })
                            
                            # Print current prediction
                            print(f"\nFrames {start_frame}-{pbar.n}: {activity} (conf: {confidence:.2f})")
                    else:
                        print("\nSkipping prediction due to invalid values in sequence")
                    
                    # Reset buffer with overlap for sliding window approach
                    sequence_buffer = sequence_buffer[sequence_length - overlap:]
                
                # Update progress bar
                pbar.update(1)
    
    # Save final results to JSON file
    if results:
        print(f"\nSaving {len(results)} predictions to {output_path}")
        with open(output_path, 'w') as f:
            json.dump({
                'model_path': init_args['model_path'],
                'video_path': video_path,
                'sequence_length': sequence_length,
                'overlap': overlap,
                'classes': list(label_encoder.classes_),
                'results': results
            }, f, indent=2)
        print(f"Results saved to {output_path}")
    else:
        print("No activity predictions were generated!")

# Entry point of the script
if __name__ == '__main__':
    main()