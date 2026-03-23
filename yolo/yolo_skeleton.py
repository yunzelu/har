
import cv2
import numpy as np
from pathlib import Path

# Attempt to import YOLO from ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("The 'ultralytics' package is not installed. Please install it with 'pip install ultralytics'.")

class YOLOSkeleton:
    """
    A wrapper class for the YOLOv8 pose estimation model, specifically for use
    with OpenVINO exported models.
    """
    def __init__(self, model_path_str: str):
        """
        Initializes the YOLOSkeleton model from a .pt file or an OpenVINO model directory.

        Args:
            model_path_str (str): The path to the YOLOv8 pose model. 
                                  Can be a .pt file or a directory for an OpenVINO model.
        """
        model_path = Path(model_path_str)

        if not model_path.exists():
            raise FileNotFoundError(f"The model path was not found at: {model_path}")

        if model_path.is_dir():
            print(f"Loading OpenVINO YOLOv8-pose model from directory: {model_path}")
        elif model_path.is_file() and model_path.suffix == '.pt':
            print(f"Loading YOLO-pose model from .pt file: {model_path}")
        else:
            raise ValueError(f"Unsupported model path: {model_path}. Please provide a .pt file or an OpenVINO model directory.")

        # The YOLO class can handle both a .pt file and a directory for OpenVINO.
        self.model = YOLO(model_path)
        print("Model loaded successfully.")

    def get_keypoints(self, image: np.ndarray) -> list:
        """
        Performs pose estimation on a single image.

        Args:
            image (np.ndarray): The input image in BGR format (as read by OpenCV).

        Returns:
            list: A list of ultralytics.engine.results.Results objects.
                  Each object in the list contains the detected keypoints,
                  bounding boxes, and scores for one person.
        """
        # Perform inference on the image.
        # The model expects RGB images, but the YOLO class handles the conversion.
        # 'verbose=False' prevents extensive logging for each prediction.
        results = self.model.predict(image, verbose=False)
        return results

    def track_skeletons(self, image_sequence: list) -> iter:
        """
        Performs pose tracking across a sequence of images using a generator.

        Args:
            image_sequence (list): A list of images (as np.ndarray) or image paths.

        Returns:
            generator: A generator of ultralytics.engine.results.Results objects,
                       one for each image in the sequence, with tracker IDs assigned.
        """
        # stream=True is crucial for long videos or large image sequences to
        # prevent memory errors and to process frames one by one.
        results_generator = self.model.track(source=image_sequence, persist=True, verbose=False, stream=True)
        return results_generator

    def draw_keypoints(self, image: np.ndarray, results: list) -> np.ndarray:
        """
        Draws the detected keypoints and bounding boxes on an image.

        Args:
            image (np.ndarray): The original image to draw on.
            results (list): The list of Results objects from the get_keypoints method.

        Returns:
            np.ndarray: The image with annotations drawn on it.
        """
        if not results:
            return image

        # The plot() method from the Results object handles all the drawing
        annotated_frame = results[0].plot()
        return annotated_frame
