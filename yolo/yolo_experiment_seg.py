import cv2
import numpy as np
import time
import os
from ultralytics import YOLO

# --- CONFIGURATION ---
VIDEO_PATH = r"data\WIN_20260326_15_06_23_Pro_trim.mp4"  
IMGSZ = (352, 640)          
# UPDATE THIS: You must use a segmentation model (e.g., yolo11l-seg.pt)
MODEL_PATH = r"yolo\yolo26n-seg_352x640_openvino_model"

filename, ext = os.path.splitext(VIDEO_PATH)
OUTPUT_PATH = f"{filename}_yolo11l_seg.mp4"

def draw_transparent_silhouettes(display_frame, results, scale_x, scale_y):
    """
    Draws thin bounding boxes and solid transparent human silhouettes (masks).
    Draws fully opaque text (IDs) on top for readability.
    """
    # Check if there are detections and masks
    if results[0].boxes is None or results[0].boxes.id is None or results[0].masks is None:
        return display_frame 

    # Create an overlay for the transparent shapes
    overlay = display_frame.copy()
    
    # List to store text so we can draw it AFTER the transparency blend
    opaque_text_list = [] 
    
    ids = results[0].boxes.id.cpu().numpy().astype(int)
    bboxes = results[0].boxes.xyxy.cpu().numpy()
    
    # Get the polygon points for the segmentation masks
    masks = results[0].masks.xy 

    for i, p_id in enumerate(ids):
        # 1. Scale and Draw Bounding Box (on overlay)
        x1 = int(bboxes[i][0] * scale_x)
        y1 = int(bboxes[i][1] * scale_y)
        x2 = int(bboxes[i][2] * scale_x)
        y2 = int(bboxes[i][3] * scale_y)
        
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 255), 1)

        # Save Person ID to be drawn opaquely
        opaque_text_list.append((f"ID:{p_id}", (x1, max(15, y1 - 5)), 0.5, (255, 0, 255)))

        # 2. Scale and Draw the Silhouette Mask (on overlay)
        polygon = masks[i]
        if len(polygon) > 0:
            # Scale the points to match the display size
            scaled_polygon = np.zeros_like(polygon)
            scaled_polygon[:, 0] = polygon[:, 0] * scale_x
            scaled_polygon[:, 1] = polygon[:, 1] * scale_y
            scaled_polygon = scaled_polygon.astype(np.int32)
            
            # Fill the polygon with a color (Cyan)
            cv2.fillPoly(overlay, [scaled_polygon], (255, 255, 0))

    # 3. Blend the shapes with the main frame (50% transparency)
    cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)
    
    # 4. Draw all the text at 100% opacity on top of the blended frame
    for text, position, font_scale, color in opaque_text_list:
        cv2.putText(display_frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

    return display_frame

def draw_hud_overlay(display_frame, results, fps):
    """
    Draws the static, semi-transparent dashboard on the left side of the screen.
    Updated for segmentation (removed keypoints).
    """
    hud_width = 240
    hud_height = 30 
    
    ids = []
    if results[0].boxes is not None and results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        box_confs = results[0].boxes.conf.cpu().numpy()
        hud_height += len(ids) * 35 
    
    overlay = display_frame.copy()
    
    cv2.rectangle(overlay, (10, 10), (10 + hud_width, 10 + hud_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, display_frame, 0.6, 0, display_frame)
    
    cv2.putText(display_frame, f"FPS: {fps:.1f}", (15, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if len(ids) > 0:
        y_offset = 55
        for i, p_id in enumerate(ids):
            box_conf = box_confs[i]
            cv2.putText(display_frame, f"--- [ID: {p_id}] Box Conf: {box_conf:.2f} ---", 
                        (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            y_offset += 25
            
    return display_frame

def run_live_viewer():
    print(f"[INFO] Loading YOLO Engine from: {MODEL_PATH}...")
    try:
        # Changed task to 'segment'
        model = YOLO(MODEL_PATH, task='segment')
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"[ERROR] Video file not found at: {VIDEO_PATH}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        video_fps = 30 

    disp_w, disp_h = 1280, 720

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, video_fps, (disp_w, disp_h))

    print(f"[INFO] Video playback started. Saving to: {OUTPUT_PATH}")
    print("[INFO] Press 'q' to quit early.")

    scale_x = disp_w / IMGSZ[1]
    scale_y = disp_h / IMGSZ[0]

    prev_time = 0 

    while True:
        success, frame = cap.read()
        
        if not success:
            print("[INFO] End of video reached.")
            break

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if prev_time > 0 else 0.0
        prev_time = current_time

        small_frame = cv2.resize(frame, (IMGSZ[1], IMGSZ[0]))

        # Run tracking with segmentation
        results = model.track(
            small_frame,             
            persist=True, 
            verbose=False, 
            imgsz=IMGSZ,
            conf=0.1, 
            classes=[0],
            tracker=r"yolo/custom_botsort.yaml" 
        )

        display_frame = cv2.resize(small_frame, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)

        # Draw Silhouettes instead of Skeletons
        display_frame = draw_transparent_silhouettes(display_frame, results, scale_x, scale_y)

        # Draw HUD
        display_frame = draw_hud_overlay(display_frame, results, fps)

        out.write(display_frame)

        cv2.imshow("InvisiGuard Video View", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Video saved successfully!")

if __name__ == "__main__":
    run_live_viewer()