## Create the environment
conda env create -f environment.yml

## Commands
### Yolo Pose Estimation from Video
python .\yolo\yolo_video.py --input {D:\lu\project\auto-labeler\data\raw_videos\sit-15.mp4}
### Render Skeleton
python .\tools\render_skeletons.py --csv-path {"d:\lu\project\har\Pose_Data\sit-15_pose_data.csv"} --output-dir "rendered_frames" --video-width 1440 --video-height 1080
### Convert Frames to Video
python .\tools\frames_to_video.py --frames-dir "rendered_frames" --output-video-path "skeleton_video.mp4" --fps 30
### Yolo Pose Estimation via Webcam
python .\yolo\yolo_skeleton.py
### CNN-Transformer HAR Inference
python cnn_transformer/cnn_transformer_inference_csv.py --model-path cnn_transformer/cnn_transformer_model.pth --label-encoder cnn_transformer/cnn_transformer_label_encoder_classes.npy --csv-path {"d:\lu\project\har\Pose_Data\sit-15_pose_data.csv"} --video-width 1440 --video-height 1080