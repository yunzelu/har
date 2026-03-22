import cv2
import numpy as np
from pathlib import Path
import argparse
import sys

# Add the project root to the Python path to allow for package imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Now we can import from the yolo package
from yolo.yolo_skeleton import YOLOSkeleton

def process_images(input_dir: Path, output_dir: Path, model_path: Path, use_tracking: bool = False):
    """
    Processes all images in a directory, detects skeletons using YOLOSkeleton,
    and saves the annotated images to an output directory.

    Args:
        input_dir (Path): The directory containing the source images.
        output_dir (Path): The directory where annotated images will be saved.
        model_path (Path): The path to the YOLO model directory.
        use_tracking (bool): If True, use tracking to maintain person IDs across images.
    """
    if not input_dir.is_dir():
        print(f"Error: Input directory not found at {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Annotated images will be saved to: {output_dir}")

    try:
        skeleton_detector = YOLOSkeleton(model_path)
    except Exception as e:
        print(f"Error initializing YOLO skeleton detector: {e}")
        print("Please ensure the model path is correct and ultralytics is installed.")
        return

    # Common image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    # Recursively find all image files
    image_files = [p for p in input_dir.glob('**/*') if p.suffix.lower() in image_extensions]

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    if use_tracking:
        print("Processing with skeleton tracking enabled...")
        # Sort files by name to process them in a sequential order, crucial for tracking
        image_files = sorted(image_files, key=lambda p: p.name)
        
        print(f"Found {len(image_files)} images to process.")

        for image_path in image_files:
            print(f"Processing {image_path.name}...")
            try:
                # Call track on each image individually.
                # 'persist=True' is crucial for maintaining track IDs across calls.
                results = skeleton_detector.model.track(str(image_path), persist=True, verbose=False)
                
                # The result is a list containing one Results object.
                if results and results[0].boxes.id is not None:
                    annotated_image = results[0].plot()

                    output_filename = f"{image_path.stem}_rendered{image_path.suffix}"
                    output_path = output_dir / output_filename
                    cv2.imwrite(str(output_path), annotated_image)
                    print(f"  -> Saved annotated image to {output_path}")
                else:
                    print(f"  -> No skeletons detected in {image_path.name}")

            except Exception as e:
                print(f"An error occurred while processing {image_path.name}: {e}")
    else:
        print("Processing without skeleton tracking...")
        print(f"Found {len(image_files)} images to process.")
        for image_path in image_files:
            print(f"Processing {image_path.name}...")
            try:
                # Read the image
                img = cv2.imread(str(image_path))
                if img is None:
                    print(f"Warning: Could not read image {image_path}, skipping.")
                    continue

                # Detect skeletons for a single image
                results = skeleton_detector.get_keypoints(img)

                # Draw the skeletons on the original image
                annotated_image = skeleton_detector.draw_keypoints(img, results)

                # Define the output path and save the image
                output_filename = f"{image_path.stem}_rendered{image_path.suffix}"
                output_path = output_dir / output_filename
                cv2.imwrite(str(output_path), annotated_image)
                print(f"  -> Saved annotated image to {output_path}")

            except Exception as e:
                print(f"An error occurred while processing {image_path.name}: {e}")

def main():
    """
    Main function to parse command-line arguments and run the image processing.
    """
    parser = argparse.ArgumentParser(
        description="Batch process a folder of images to detect and render skeletons using YOLOv8-pose.",
        epilog="Example: python tools/batch_render_skeletons.py path/to/your/images path/to/output --track"
    )
    
    parser.add_argument(
        "input_dir", 
        type=str, 
        help="Path to the directory containing input images."
    )
    
    parser.add_argument(
        "output_dir", 
        type=str, 
        help="Path to the directory where annotated images will be saved."
    )
    
    # Allow user to specify a different model if needed
    default_model_path = project_root / 'yolo' / 'yolov8n-pose_openvino_model'
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=str(default_model_path),
        help=f"Path to the YOLO model directory. Defaults to: {default_model_path}"
    )

    parser.add_argument(
        "--track",
        action="store_true",
        help="Enable skeleton tracking to assign and render person IDs across images."
    )

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    model_path = Path(args.model_path)

    process_images(input_path, output_path, model_path, use_tracking=args.track)

if __name__ == '__main__':
    main()
