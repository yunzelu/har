import os
import shutil
from ultralytics import YOLO

# 1. Load your base PyTorch model
# (Change this to your specific .pt file if it's custom)
base_model = YOLO(r"yolo/yolo26n-pose.pt") 

# 2. Define the ablation sizes (Height, Width). 
# YOLO requires dimensions to be multiples of 32.
ablation_sizes = [
    (256, 448),
    (352, 640),  # Your current baseline
    (480, 832),
    (736, 1280)
]

for h, w in ablation_sizes:
    print(f"\n[INFO] Exporting model for size: {h}x{w}...")
    
    # Export the model
    export_dir = base_model.export(format="openvino", imgsz=(h, w))
    
    # Create a unique folder name for this resolution
    unique_folder_name = f"yolo_openvino_{h}x{w}_openvino_model"
    
    # If a folder with this name already exists from a previous run, delete it
    if os.path.exists(unique_folder_name):
        shutil.rmtree(unique_folder_name)
        
    # Rename the exported folder to our unique name
    os.rename(export_dir, unique_folder_name)
    
    print(f"[SUCCESS] Saved OpenVINO model to: {unique_folder_name}")