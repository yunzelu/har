#!/usr/bin/env python3
"""
Quick HPE reliability evaluation for YOLO-Pose OpenVINO models.

Metrics:
1) Lost detection rate, assuming every video frame contains exactly one subject.
2) Multiple-detection rate, because detecting more than one subject is wrong for this clip.
3) Exact-one-person detection rate.
4) Mean keypoint confidence.
5) Mean running FPS.

Convenience:
- You can pass either full OpenVINO model folders OR short keywords.
- Example model folder:
      yolo/yolo26n-pose_736x1280_openvino_model
- You can run:
      python hpe_quick_eval.py --video clip.mp4 --models yolo26n-pose --model-root yolo
- The script will recursively search under --model-root and infer imgsz from names like 736x1280.

Python compatibility:
- Works with Python 3.8+.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


ImgSize = Union[int, List[int]]


def parse_imgsz_value(value: str) -> ImgSize:
    """
    Parse image size.

    Supported:
      "640"       -> 640
      "384x640"   -> [384, 640], meaning height x width
      "384,640"   -> [384, 640], meaning height, width
    """
    value = value.strip().lower().replace(",", "x")

    if "x" not in value:
        return int(value)

    parts = [p for p in value.split("x") if p]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Invalid image size '{value}'. Use 640 or heightxwidth, e.g., 384x640."
        )

    h, w = int(parts[0]), int(parts[1])
    return [h, w]


def format_imgsz(imgsz: Optional[ImgSize]) -> str:
    if imgsz is None:
        return "auto"
    if isinstance(imgsz, list):
        return "x".join(str(v) for v in imgsz)
    return str(imgsz)


def infer_imgsz_from_name(path: Path) -> Optional[ImgSize]:
    """
    Infer image size from a model folder name.

    Examples:
      yolo26n-pose_736x1280_openvino_model -> [736, 1280]
      yolov8n-pose_640_openvino_model      -> 640
    """
    name = path.name.lower()

    # Rectangular size, e.g., 736x1280
    m = re.search(r"(?<!\d)(\d{2,5})\s*[xX]\s*(\d{2,5})(?!\d)", name)
    if m:
        return [int(m.group(1)), int(m.group(2))]

    # Square size, e.g., _640_openvino_model
    m = re.search(r"_(\d{2,5})_openvino_model$", name)
    if m:
        return int(m.group(1))

    return None


def looks_like_openvino_model_dir(path: Path) -> bool:
    """
    Check whether a directory is probably an Ultralytics OpenVINO export folder.
    """
    if not path.is_dir():
        return False

    if "openvino_model" not in path.name.lower():
        return False

    has_xml = any(path.glob("*.xml"))
    has_metadata = (path / "metadata.yaml").exists()
    has_model_xml = (path / "model.xml").exists()

    return has_xml or has_metadata or has_model_xml


def find_model_dirs_for_query(query: str, model_root: Path) -> List[Path]:
    """
    Find OpenVINO model folders under model_root matching a short query.

    Example:
      query='yolo26n-pose'
      match='yolo/yolo26n-pose_736x1280_openvino_model'
    """
    q = query.lower().strip()
    root = model_root.expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"model root does not exist: {root}")

    matches = []
    for p in root.rglob("*openvino_model*"):
        if looks_like_openvino_model_dir(p) and q in p.name.lower():
            matches.append(p)

    matches = sorted(set(matches), key=lambda x: (x.name.lower(), str(x).lower()))
    return matches


def resolve_model_inputs(model_inputs: List[str], model_root: Path) -> List[Path]:
    """
    Resolve user model inputs into concrete model folders.

    Each input may be:
      1) an exact folder path, or
      2) a short keyword to search under --model-root.
    """
    resolved = []

    for item in model_inputs:
        raw = Path(item).expanduser()

        if raw.exists():
            if not raw.is_dir():
                raise ValueError(f"Model path exists but is not a directory: {raw}")
            resolved.append(raw.resolve())
            continue

        matches = find_model_dirs_for_query(item, model_root=model_root)

        if not matches:
            raise FileNotFoundError(
                f"No OpenVINO model folder found for keyword '{item}' under '{model_root}'.\n"
                "Tip: run from the project parent folder, or set --model-root yolo."
            )

        print(f"\nKeyword '{item}' matched {len(matches)} model folder(s):", flush=True)
        for m in matches:
            inferred = infer_imgsz_from_name(m)
            inferred_text = format_imgsz(inferred) if inferred is not None else "not found"
            print(f"  - {m}  | inferred imgsz: {inferred_text}", flush=True)

        resolved.extend(matches)

    # Remove duplicates while preserving order.
    unique = []
    seen = set()

    for p in resolved:
        key = str(p.resolve())
        if key not in seen:
            unique.append(p)
            seen.add(key)

    return unique


def build_imgsz_per_model(
    models: List[Path],
    imgsz_args: Optional[List[ImgSize]],
) -> List[ImgSize]:
    """
    Build one image size per model.

    Priority:
      1) If --imgsz is given: use it for all models, or one per model.
      2) If --imgsz is not given: infer from each folder name.
    """
    if imgsz_args is not None and len(imgsz_args) > 0:
        if len(imgsz_args) == 1:
            return imgsz_args * len(models)

        if len(imgsz_args) == len(models):
            return imgsz_args

        raise ValueError(
            "--imgsz must contain either one value for all resolved models, "
            "or the same number of values as the resolved model folders.\n"
            f"Got {len(imgsz_args)} image sizes for {len(models)} resolved models."
        )

    inferred = []
    missing = []

    for m in models:
        size = infer_imgsz_from_name(m)
        if size is None:
            missing.append(str(m))
        else:
            inferred.append(size)

    if missing:
        raise ValueError(
            "Could not infer image size from these model folder names:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\nPlease pass --imgsz 640 or --imgsz 736x1280."
        )

    return inferred


def safe_name(text: str) -> str:
    name = Path(text).name or str(text)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

def parse_frame_ranges(range_args: Optional[List[str]]) -> List[tuple]:
    """
    Parse frame ranges from command line.

    Example:
      --lying-ranges 250-936 2301-2610 2767-3263

    Returns:
      [(250, 936), (2301, 2610), (2767, 3263)]

    The ranges are inclusive:
      250-936 means frame_index >= 250 and frame_index <= 936.
    """
    if not range_args:
        return []

    ranges = []

    for item in range_args:
        item = item.strip()

        if "-" not in item:
            raise ValueError(
                f"Invalid frame range '{item}'. Use format start-end, e.g., 250-936."
            )

        start_text, end_text = item.split("-", 1)
        start = int(start_text)
        end = int(end_text)

        if start < 0 or end < 0:
            raise ValueError(f"Frame range cannot be negative: {item}")

        if end < start:
            raise ValueError(f"Frame range end must be >= start: {item}")

        ranges.append((start, end))

    return ranges


def frame_in_any_range(frame_index: int, ranges: List[tuple]) -> bool:
    """
    Return True if frame_index is inside any inclusive frame range.
    """
    for start, end in ranges:
        if start <= frame_index <= end:
            return True
    return False

def to_numpy(x: Any) -> Optional[np.ndarray]:
    """
    Convert torch/ultralytics tensors or arrays to numpy.
    """
    if x is None:
        return None

    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()

    if hasattr(x, "cpu"):
        return x.cpu().numpy()

    return np.asarray(x)


def extract_person_metrics(
    result: Any,
    task: str,
    person_class_id: int,
    kp_conf_thres: float,
) -> dict:
    """
    Extract person-detection reliability metrics from YOLO result.

    Supports:
      - pose model: uses boxes + keypoints
      - detection model: uses boxes only
      - segmentation model: uses boxes + masks if available

    Assumption for this experiment:
      every frame should contain exactly one subject.

    Therefore:
      num_persons == 0 -> lost detection
      num_persons == 1 -> correct person-count detection
      num_persons > 1  -> multiple-detection error

    For lying-pose detection metric:
      num_persons > 0 is counted as detected.
      Multiple detections are still counted as detected for lying-specific detection.
    """
    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    masks = getattr(result, "masks", None)

    bbox_conf = math.nan
    mean_kpt_conf = math.nan
    min_kpt_conf = math.nan
    num_valid_keypoints = 0
    mask_area_ratio = math.nan

    if boxes is None:
        num_persons = 0
        return {
            "person_detected": False,
            "exactly_one_person": False,
            "multiple_persons_detected": False,
            "person_count_error": True,
            "num_persons": 0,
            "bbox_conf": bbox_conf,
            "mean_keypoint_conf": mean_kpt_conf,
            "min_keypoint_conf": min_kpt_conf,
            "num_valid_keypoints": num_valid_keypoints,
            "mask_area_ratio": mask_area_ratio,
        }

    try:
        total_boxes = len(boxes)
    except TypeError:
        total_boxes = 0

    if total_boxes <= 0:
        num_persons = 0
        return {
            "person_detected": False,
            "exactly_one_person": False,
            "multiple_persons_detected": False,
            "person_count_error": True,
            "num_persons": 0,
            "bbox_conf": bbox_conf,
            "mean_keypoint_conf": mean_kpt_conf,
            "min_keypoint_conf": min_kpt_conf,
            "num_valid_keypoints": num_valid_keypoints,
            "mask_area_ratio": mask_area_ratio,
        }

    box_conf_arr = to_numpy(getattr(boxes, "conf", None))
    box_cls_arr = to_numpy(getattr(boxes, "cls", None))

    if box_conf_arr is None:
        box_conf_arr = np.ones((total_boxes,), dtype=float)
    else:
        box_conf_arr = np.asarray(box_conf_arr).reshape(-1).astype(float)

    if box_cls_arr is None:
        # If class is missing, assume all boxes are person boxes.
        # This is often okay for pose models exported only for human pose.
        person_indices = np.arange(total_boxes)
    else:
        box_cls_arr = np.asarray(box_cls_arr).reshape(-1).astype(int)
        person_indices = np.where(box_cls_arr == person_class_id)[0]

    num_persons = int(len(person_indices))

    if num_persons <= 0:
        return {
            "person_detected": False,
            "exactly_one_person": False,
            "multiple_persons_detected": False,
            "person_count_error": True,
            "num_persons": 0,
            "bbox_conf": bbox_conf,
            "mean_keypoint_conf": mean_kpt_conf,
            "min_keypoint_conf": min_kpt_conf,
            "num_valid_keypoints": num_valid_keypoints,
            "mask_area_ratio": mask_area_ratio,
        }

    # Choose the highest-confidence person.
    person_confs = box_conf_arr[person_indices]
    best_local_idx = int(np.nanargmax(person_confs))
    best_idx = int(person_indices[best_local_idx])
    bbox_conf = float(box_conf_arr[best_idx])

    # Pose-specific keypoint confidence.
    # Only available for pose models.
    if keypoints is not None:
        kpt_conf_arr = to_numpy(getattr(keypoints, "conf", None))

        if kpt_conf_arr is None:
            kpt_data = to_numpy(getattr(keypoints, "data", None))
            if kpt_data is not None and kpt_data.ndim >= 3 and kpt_data.shape[-1] >= 3:
                kpt_conf_arr = kpt_data[..., 2]

        if kpt_conf_arr is not None and np.asarray(kpt_conf_arr).size > 0:
            kpt_conf_arr = np.asarray(kpt_conf_arr)

            if kpt_conf_arr.ndim == 1:
                kpt_conf_arr = kpt_conf_arr[None, :]

            if best_idx >= kpt_conf_arr.shape[0]:
                best_idx_for_kpt = 0
            else:
                best_idx_for_kpt = best_idx

            person_kpt_conf = kpt_conf_arr[best_idx_for_kpt].astype(float)

            if person_kpt_conf.size > 0:
                mean_kpt_conf = float(np.nanmean(person_kpt_conf))
                min_kpt_conf = float(np.nanmin(person_kpt_conf))
                num_valid_keypoints = int(np.sum(person_kpt_conf >= kp_conf_thres))

    # Segmentation-specific mask area.
    # Optional diagnostic metric: how much image area the selected person's mask occupies.
    if masks is not None:
        mask_data = to_numpy(getattr(masks, "data", None))

        if mask_data is not None and np.asarray(mask_data).size > 0:
            mask_data = np.asarray(mask_data)

            if mask_data.ndim == 2:
                mask_data = mask_data[None, :, :]

            if best_idx < mask_data.shape[0]:
                selected_mask = mask_data[best_idx]
                mask_area = float(np.sum(selected_mask > 0.5))
                image_area = float(selected_mask.shape[0] * selected_mask.shape[1])

                if image_area > 0:
                    mask_area_ratio = mask_area / image_area

    exactly_one_person = num_persons == 1
    multiple_persons_detected = num_persons > 1

    return {
        "person_detected": True,
        "exactly_one_person": bool(exactly_one_person),
        "multiple_persons_detected": bool(multiple_persons_detected),
        "person_count_error": bool(not exactly_one_person),
        "num_persons": int(num_persons),
        "bbox_conf": bbox_conf,
        "mean_keypoint_conf": mean_kpt_conf,
        "min_keypoint_conf": min_kpt_conf,
        "num_valid_keypoints": num_valid_keypoints,
        "mask_area_ratio": mask_area_ratio,
    }

def run_model_on_video(
    model_path: Path,
    video_path: Path,
    imgsz: ImgSize,
    out_dir: Path,
    conf: float,
    kp_conf_thres: float,
    device: Optional[str],
    max_frames: Optional[int],
    warmup: int,
    lying_ranges: List[tuple],
    task: str,
    person_class_id: int,
) -> dict:
    model_label = f"{safe_name(str(model_path))}_imgsz{format_imgsz(imgsz)}"
    model_out_dir = out_dir / model_label
    model_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Running model: {model_path} | imgsz={format_imgsz(imgsz)} ===", flush=True)

    if task == "auto":
        model = YOLO(str(model_path))
    else:
        model = YOLO(str(model_path), task=task)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if max_frames is not None:
        progress_total = min(max_frames, total_video_frames) if total_video_frames > 0 else max_frames
    else:
        progress_total = total_video_frames if total_video_frames > 0 else None

    predict_kwargs = {
        "imgsz": imgsz,
        "conf": conf,
        "verbose": False,
    }

    if device:
        predict_kwargs["device"] = device

    # Warmup: not counted in FPS.
    ret, first_frame = cap.read()

    if ret and warmup > 0:
        print(f"Warmup predictions: {warmup}", flush=True)
        for _ in range(warmup):
            _ = model.predict(first_frame, **predict_kwargs)[0]

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    rows = []
    frame_index = 0
    total_inference_s = 0.0
    lost_count = 0
    multi_count = 0

    if tqdm is not None:
        pbar = tqdm(
            total=progress_total,
            desc=model_label,
            unit="frame",
            dynamic_ncols=True,
        )
    else:
        pbar = None
        print("tqdm is not installed. Progress will be printed every 100 frames.", flush=True)

    while True:
        if max_frames is not None and frame_index >= max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        result = model.predict(frame, **predict_kwargs)[0]
        inference_s = time.perf_counter() - t0
        total_inference_s += inference_s

        m = extract_person_metrics(
            result=result,
            task=task,
            person_class_id=person_class_id,
            kp_conf_thres=kp_conf_thres,
        )

        if not m["person_detected"]:
            lost_count += 1

        if m["multiple_persons_detected"]:
            multi_count += 1

        timestamp_s = frame_index / source_fps if source_fps > 0 else math.nan
        running_fps = 1.0 / inference_s if inference_s > 0 else math.nan
        effective_fps_so_far = (frame_index + 1) / total_inference_s if total_inference_s > 0 else math.nan
        is_lying_frame = frame_in_any_range(frame_index, lying_ranges)

        # For this lying-pose metric:
        # 0 persons = not detected
        # 1 person = detected
        # more than 1 persons = still detected
        lying_not_detected = bool(is_lying_frame and m["num_persons"] == 0)

        rows.append(
            {
                "model": str(model_path),
                "model_label": model_label,
                "imgsz": format_imgsz(imgsz),
                "frame_index": frame_index,
                "timestamp_s": timestamp_s,
                "person_detected": int(m["person_detected"]),
                "exactly_one_person": int(m["exactly_one_person"]),
                "multiple_persons_detected": int(m["multiple_persons_detected"]),
                "person_count_error": int(m["person_count_error"]),
                "num_persons": m["num_persons"],
                "bbox_conf": m["bbox_conf"],
                "mean_keypoint_conf": m["mean_keypoint_conf"],
                "min_keypoint_conf": m["min_keypoint_conf"],
                "num_valid_keypoints": m["num_valid_keypoints"],
                "inference_s": inference_s,
                "running_fps": running_fps,
                "lying_frame": int(is_lying_frame),
                "lying_detected": int(is_lying_frame and m["num_persons"] > 0),
                "lying_not_detected": int(lying_not_detected),
                "mask_area_ratio": m["mask_area_ratio"],
            }
        )

        frame_index += 1

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(
                fps=f"{effective_fps_so_far:.2f}",
                lost=lost_count,
                multi=multi_count,
            )
        else:
            if frame_index % 100 == 0:
                print(
                    f"  processed {frame_index} frames | "
                    f"fps={effective_fps_so_far:.2f} | "
                    f"lost={lost_count} | multi={multi_count}",
                    flush=True,
                )

    if pbar is not None:
        pbar.close()

    cap.release()

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError("No frames were processed. Check the video path or codec.")

    per_frame_csv = model_out_dir / "per_frame_metrics.csv"
    df.to_csv(per_frame_csv, index=False)

    processed_frames = len(df)

    detected_frames = int(df["person_detected"].sum())
    lost_frames = processed_frames - detected_frames
    lost_detection_rate = lost_frames / processed_frames

    exactly_one_person_frames = int(df["exactly_one_person"].sum())
    multiple_detection_frames = int(df["multiple_persons_detected"].sum())
    person_count_error_frames = int(df["person_count_error"].sum())

    exact_one_person_rate = exactly_one_person_frames / processed_frames
    multiple_detection_rate = multiple_detection_frames / processed_frames
    person_count_error_rate = person_count_error_frames / processed_frames

    # Mean keypoint confidence only for frames where a skeleton/person was detected.
    mean_kpt_conf_detected = float(
        df.loc[df["person_detected"] == 1, "mean_keypoint_conf"].mean()
    )

    # Stricter version: lost frames are counted as zero keypoint confidence.
    mean_kpt_conf_lost_as_zero = float(df["mean_keypoint_conf"].fillna(0.0).mean())

    mean_bbox_conf_detected = float(
        df.loc[df["person_detected"] == 1, "bbox_conf"].mean()
    )

    mean_mask_area_ratio_detected = float(
        df.loc[df["person_detected"] == 1, "mask_area_ratio"].mean()
    )

    effective_running_fps = (
        processed_frames / total_inference_s if total_inference_s > 0 else math.nan
    )

    mean_per_frame_fps = float(df["running_fps"].mean())

    lying_total_frames = int(df["lying_frame"].sum())

    if lying_total_frames > 0:
        lying_not_detected_frames = int(df["lying_not_detected"].sum())
        lying_detected_frames = int(df["lying_detected"].sum())
        lying_not_detected_rate = lying_not_detected_frames / lying_total_frames
        lying_detected_rate = lying_detected_frames / lying_total_frames
    else:
        lying_not_detected_frames = 0
        lying_detected_frames = 0
        lying_not_detected_rate = math.nan
        lying_detected_rate = math.nan

    summary = {
        "model": str(model_path),
        "model_label": model_label,
        "imgsz": format_imgsz(imgsz),
        "video": str(video_path),
        "source_fps": source_fps,
        "total_video_frames_reported_by_cv2": total_video_frames,
        "processed_frames": processed_frames,
        "detected_frames": detected_frames,
        "lost_frames": lost_frames,
        "lost_detection_rate": lost_detection_rate,
        "exactly_one_person_frames": exactly_one_person_frames,
        "multiple_detection_frames": multiple_detection_frames,
        "person_count_error_frames": person_count_error_frames,
        "exact_one_person_rate": exact_one_person_rate,
        "multiple_detection_rate": multiple_detection_rate,
        "person_count_error_rate": person_count_error_rate,
        "mean_keypoint_conf_detected_frames": mean_kpt_conf_detected,
        "mean_keypoint_conf_lost_as_zero": mean_kpt_conf_lost_as_zero,
        "mean_bbox_conf_detected_frames": mean_bbox_conf_detected,
        "total_inference_s": total_inference_s,
        "effective_running_fps": effective_running_fps,
        "mean_per_frame_fps": mean_per_frame_fps,
        "per_frame_csv": str(per_frame_csv),
        "lying_total_frames": lying_total_frames,
        "lying_detected_frames": lying_detected_frames,
        "lying_not_detected_frames": lying_not_detected_frames,
        "lying_detected_rate": lying_detected_rate,
        "lying_not_detected_rate": lying_not_detected_rate,
        "mean_mask_area_ratio_detected_frames": mean_mask_area_ratio_detected,
    }

    # Per-model plots.
    plot_x = df["timestamp_s"] if source_fps > 0 else df["frame_index"]
    x_label = "Time (s)" if source_fps > 0 else "Frame index"

    plt.figure(figsize=(10, 4))
    plt.plot(plot_x, df["mean_keypoint_conf"])
    plt.ylim(0, 1)
    plt.xlabel(x_label)
    plt.ylabel("Mean keypoint confidence")
    plt.title(f"Mean keypoint confidence over time\n{model_label}")
    plt.tight_layout()
    plt.savefig(model_out_dir / "mean_keypoint_conf_over_time.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(plot_x, df["person_detected"])
    plt.ylim(-0.05, 1.05)
    plt.xlabel(x_label)
    plt.ylabel("Person detected")
    plt.title(f"Detection availability over time\n{model_label}")
    plt.tight_layout()
    plt.savefig(model_out_dir / "detection_availability_over_time.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(plot_x, df["num_persons"])
    plt.xlabel(x_label)
    plt.ylabel("Number of detected persons")
    plt.title(f"Detected person count over time\n{model_label}")
    plt.tight_layout()
    plt.savefig(model_out_dir / "detected_person_count_over_time.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(plot_x, df["running_fps"])
    plt.xlabel(x_label)
    plt.ylabel("Running FPS")
    plt.title(f"Running FPS over time\n{model_label}")
    plt.tight_layout()
    plt.savefig(model_out_dir / "running_fps_over_time.png", dpi=200)
    plt.close()

    print(
        f"\nDone: {model_label}\n"
        f"  lost_detection_rate = {lost_detection_rate:.4f}\n"
        f"  multiple_detection_rate = {multiple_detection_rate:.4f}\n"
        f"  exact_one_person_rate = {exact_one_person_rate:.4f}\n"
        f"  person_count_error_rate = {person_count_error_rate:.4f}\n"
        f"  lying_total_frames = {lying_total_frames}\n"
        f"  lying_not_detected_frames = {lying_not_detected_frames}\n"
        f"  lying_not_detected_rate = {lying_not_detected_rate:.4f}\n"
        f"  mean_keypoint_conf_detected_frames = {mean_kpt_conf_detected:.4f}\n"
        f"  effective_running_fps = {effective_running_fps:.2f}\n"
        f"  saved: {per_frame_csv}",
        flush=True,
    )

    return summary


def save_summary_plots(summary_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Save simple comparison bar plots across models.
    """
    plot_specs = [
        ("lost_detection_rate", "Lost detection rate", "summary_lost_detection_rate.png"),
        ("multiple_detection_rate", "Multiple-detection rate", "summary_multiple_detection_rate.png"),
        ("person_count_error_rate", "Person-count error rate", "summary_person_count_error_rate.png"),
        ("exact_one_person_rate", "Exact-one-person rate", "summary_exact_one_person_rate.png"),
        (
            "mean_keypoint_conf_detected_frames",
            "Mean keypoint confidence",
            "summary_mean_keypoint_conf.png",
        ),
        ("effective_running_fps", "Effective running FPS", "summary_effective_running_fps.png"),
    ]

    labels = summary_df["model_label"].astype(str).tolist()

    for col, ylabel, filename in plot_specs:
        plt.figure(figsize=(max(8, 1.8 * len(labels)), 4.8))
        plt.bar(labels, summary_df[col])
        plt.ylabel(ylabel)
        plt.title(ylabel + " by model")
        plt.xticks(rotation=30, ha="right")

        if "rate" in col or "confidence" in col:
            plt.ylim(0, 1)

        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=200)
        plt.close()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Quickly evaluate YOLO-Pose OpenVINO HPE detection loss, "
            "multiple detection, keypoint confidence, and running FPS."
        )
    )

    parser.add_argument(
        "--video",
        required=True,
        type=Path,
        help="Path to input video clip.",
    )

    parser.add_argument(
        "--models",
        required=True,
        nargs="+",
        type=str,
        help=(
            "One or more exact OpenVINO model folders OR short search keywords. "
            "Example: --models yolo26n-pose. The script searches under --model-root."
        ),
    )

    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("."),
        help="Root folder used when --models contains keywords instead of exact paths. Default: current folder.",
    )

    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=parse_imgsz_value,
        default=None,
        help=(
            "Optional override. One size for all resolved models, or one per resolved model. "
            "Use 640 or heightxwidth, e.g., 736x1280. If omitted, inferred from folder names."
        ),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/hpe_quick_eval"),
        help="Output folder.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO person detection confidence threshold.",
    )

    parser.add_argument(
        "--kp-conf-thres",
        type=float,
        default=0.50,
        help=(
            "Keypoint confidence threshold used only to count valid keypoints. "
            "Mean confidence is not thresholded."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device passed to Ultralytics, e.g., cpu, CPU, gpu, GPU. Omit to use default.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process for a quick test.",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup predictions before timing. Default: 3.",
    )

    parser.add_argument(
        "--lying-ranges",
        nargs="+",
        default=["250-936", "2301-2610", "2767-3263"],
        help=(
            "Inclusive frame ranges for lying-pose periods. "
            "Example: --lying-ranges 250-936 2301-2610 2767-3263. "
            "For this metric, multiple detections still count as detected; only zero detections count as not detected."
        ),
    )

    parser.add_argument(
        "--task",
        type=str,
        default="auto",
        choices=["auto", "pose", "detect", "segment"],
        help=(
            "YOLO task type. Use pose for YOLO-Pose, detect for YOLO detection, "
            "segment for YOLO segmentation, or auto to let Ultralytics infer it."
        ),
    )

    parser.add_argument(
        "--person-class-id",
        type=int,
        default=0,
        help=(
            "Class ID for person. For COCO-trained YOLO models, person is usually class 0."
        ),
    )

    return parser


def main() -> int:
    args = build_argparser().parse_args()

    if not args.video.exists():
        print(f"ERROR: video does not exist: {args.video}", file=sys.stderr)
        return 2

    try:
        resolved_models = resolve_model_inputs(args.models, model_root=args.model_root)
        imgsz_per_model = build_imgsz_per_model(resolved_models, args.imgsz)
        lying_ranges = parse_frame_ranges(args.lying_ranges)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if not resolved_models:
        print("ERROR: no model folders resolved.", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)

    print("\nResolved model list:", flush=True)
    print(f"Lying-pose frame ranges: {lying_ranges}", flush=True)
    for m, s in zip(resolved_models, imgsz_per_model):
        print(f"  - {m} | imgsz={format_imgsz(s)}", flush=True)

    summaries = []

    for model_path, imgsz in zip(resolved_models, imgsz_per_model):
        summary = run_model_on_video(
            model_path=model_path,
            video_path=args.video,
            imgsz=imgsz,
            out_dir=args.out,
            conf=args.conf,
            kp_conf_thres=args.kp_conf_thres,
            device=args.device,
            max_frames=args.max_frames,
            warmup=args.warmup,
            lying_ranges=lying_ranges,
            task=args.task,
            person_class_id=args.person_class_id,
        )
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)

    summary_csv = args.out / "summary_metrics.csv"
    summary_df.to_csv(summary_csv, index=False)

    save_summary_plots(summary_df, args.out)

    print("\n=== Summary ===")

    cols = [
        "model_label",
        "processed_frames",
        "lost_detection_rate",
        "multiple_detection_rate",
        "person_count_error_rate",
        "exact_one_person_rate",
        "lying_total_frames",
        "lying_not_detected_frames",
        "lying_not_detected_rate",
        "mean_keypoint_conf_detected_frames",
        "mean_keypoint_conf_lost_as_zero",
        "mean_bbox_conf_detected_frames",
        "effective_running_fps",
        "mean_mask_area_ratio_detected_frames",
    ]

    print(summary_df[cols].to_string(index=False))

    print(f"\nSaved summary CSV: {summary_csv}")
    print(f"Saved plots in: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())