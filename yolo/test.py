import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def open_camera(camera_id: int, width: int, height: int, fps: int):
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)  # CAP_DSHOW is useful on Windows

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera ID {camera_id}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"[INFO] Requested camera: {width}x{height} @ {fps} FPS")
    print(f"[INFO] Actual camera:    {actual_width}x{actual_height} @ {actual_fps:.2f} FPS")

    return cap, actual_width, actual_height


def main():
    parser = argparse.ArgumentParser(
        description="Run OpenVINO YOLO pose on webcam, show skeleton preview, save only raw video."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to OpenVINO YOLO model folder, e.g. yolo26n-pose_openvino_model"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam ID. Usually 0 for laptop webcam."
    )
    parser.add_argument(
        "--out",
        type=str,
        default="raw_recording.mp4",
        help="Output raw video path. The saved video will NOT include skeleton rendering."
    )
    parser.add_argument(
        "--cam-width",
        type=int,
        default=640,
        help="Requested webcam capture width."
    )
    parser.add_argument(
        "--cam-height",
        type=int,
        default=352,
        help="Requested webcam capture height."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Requested recording FPS."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0,
        help="Recording duration in seconds. Use 0 to record until pressing q."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device. For OpenVINO model, usually use cpu."
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(str(model_path))

    cap, actual_width, actual_height = open_camera(
        camera_id=args.camera,
        width=args.cam_width,
        height=args.cam_height,
        fps=args.fps,
    )

    # mp4v is usually safer for .mp4 on Windows/macOS/Linux.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(out_path),
        fourcc,
        args.fps,
        (actual_width, actual_height),
        True
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video writer for: {out_path}")

    print("[INFO] Recording started.")
    print("[INFO] Press 'q' to stop.")
    if args.duration > 0:
        print(f"[INFO] Auto-stop after {args.duration:.1f} seconds.")

    start_time = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame from camera.")
                break

            # Save ONLY the raw, unannotated frame.
            writer.write(frame)

            # Run YOLO pose inference for preview only.
            # imgsz=(352, 640) means model inference size height x width.
            results = model.predict(
                source=frame,
                imgsz=(352, 640),
                conf=args.conf,
                verbose=False,
                device=args.device
            )

            # Render skeleton on preview frame only.
            annotated = results[0].plot()

            elapsed = time.time() - start_time
            frame_count += 1

            cv2.putText(
                annotated,
                f"REC raw only | {elapsed:.1f}s | frames: {frame_count}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

            cv2.imshow("HPE Preview - saved video is raw only", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if args.duration > 0 and elapsed >= args.duration:
                break

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    total_time = time.time() - start_time
    real_fps = frame_count / total_time if total_time > 0 else 0

    print("[INFO] Recording finished.")
    print(f"[INFO] Saved raw video to: {out_path}")
    print(f"[INFO] Frames saved: {frame_count}")
    print(f"[INFO] Actual average FPS: {real_fps:.2f}")


if __name__ == "__main__":
    main()