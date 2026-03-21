import cv2
import os

def get_video_metadata(video_path):
    """
    Extracts the FPS and total number of frames from a video.
    """
    # Check if the file actually exists before trying to open it
    if not os.path.exists(video_path):
        print(f"Error: The file '{video_path}' does not exist.")
        return None, None

    # Initialize the video capture object
    cap = cv2.VideoCapture(video_path)

    # Verify that OpenCV could successfully open the video
    if not cap.isOpened():
        print(f"Error: Could not open the video file '{video_path}'. It might be corrupted or an unsupported format.")
        return None, None

    # Extract FPS and frame count using OpenCV properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Always release the capture object to free up resources
    cap.release()

    return fps, total_frames

if __name__ == "__main__":
    # Replace with the path to your actual video file
    sample_video_path = "D:/lu/project/har/Human_Activity_Recognition_Video_Dataset/human-activity-recognition-video-dataset/versions/1/Human Activity Recognition - Video Dataset/Sitting/Sitting (1).mp4" 
    
    fps, frames = get_video_metadata(sample_video_path)

    if fps is not None and frames is not None:
        print("-" * 30)
        print(f"Video File:   {sample_video_path}")
        print(f"FPS:          {fps:.2f}")
        print(f"Total Frames: {frames}")
        print("-" * 30)