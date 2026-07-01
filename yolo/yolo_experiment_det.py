import cv2
import numpy as np
import time
import os
from ultralytics import YOLO

# --- CONFIGURATION ---
VIDEO_PATH = r"data\WIN_20260326_15_06_23_Pro_trim.mp4"  
IMGSZ = (352, 640)          
# UPDATE THIS: You must use a standard detection model (e.g., yolo11l.pt)
MODEL_PATH = r"yolo\yolo26n-det_352x640_openvino_model"

filename, ext = os.path.splitext(VIDEO_PATH)
OUTPUT_PATH = f"{filename}_yolo26l_det_640.mp4"

def draw_transparent_boxes(display_frame, results, scale_x, scale_y):
    """
    Draws semi-transparent bounding boxes around humans.
    Draws fully opaque text (IDs) on top for readability.
    """
    # Check if there are detections and IDs
    if results[0].boxes is None or results[0].boxes.id is None:
        return display_frame 

    # Create an overlay for the transparent boxes
    overlay = display_frame.copy()
    
    # List to store text so we can draw it AFTER the transparency blend
    opaque_text_list = [] 
    
    ids = results[0].boxes.id.cpu().numpy().astype(int)
    bboxes = results[0].boxes.xyxy.cpu().numpy()

    for i, p_id in enumerate(ids):
        # 1. Scale coordinates to fit the display frame
        x1 = int(bboxes[i][0] * scale_x)
        y1 = int(bboxes[i][1] * scale_y)
        x2 = int(bboxes[i][2] * scale_x)
        y2 = int(bboxes[i][3] * scale_y)
        
        # Draw a solid rectangle on the overlay (we will make it transparent later)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 0), cv2.FILLED)
        
        # Draw a sharp border around the box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Save Person ID to be drawn clearly
        opaque_text_list.append((f"ID:{p_id}", (x1, max(15, y1 - 5)), 0.6, (255, 255, 255)))

    # 2. Blend the colored boxes with the main frame (40% transparency)
    cv2.addWeighted(overlay, 0.4, display_frame, 0.6, 0, display_frame)
    
    # 3. Draw all the text at 100% opacity on top of the blended frame
    for text, position, font_scale, color in opaque_text_list:
        # Draw a black shadow first so the white text is easy to read
        cv2.putText(display_frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3)
        cv2.putText(display_frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

    return display_frame

def draw_hud_overlay(display_frame, results, fps):
    """
    Draws the static, semi-transparent dashboard on the left side of the screen.
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
    print(f"[INFO] Loading YOLO Detection Engine from: {MODEL_PATH}...")
    try:
        # Task changed to 'detect'
        model = YOLO(MODEL_PATH, task='detect')
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

        # Run tracking with DETECTION, filtered for humans (classes=[0])
        results = model.track(
            small_frame,             
            persist=True, 
            verbose=False, 
            imgsz=IMGSZ,
            conf=0.1, 
            classes=[0], # 0 is the ID for 'person'
            tracker=r"yolo/custom_botsort.yaml" 
        )

        display_frame = cv2.resize(small_frame, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)

        # Draw bounding boxes
        display_frame = draw_transparent_boxes(display_frame, results, scale_x, scale_y)

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