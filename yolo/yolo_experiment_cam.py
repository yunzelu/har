import cv2
import numpy as np
import time
from ultralytics import YOLO

# --- CONFIGURATION ---
WEBCAM_INDEX = 0      
IMGSZ = (352, 640)          
MODEL_PATH = r"yolo/yolo11s_openvino_352x640_openvino_model"

# --- RENDERING CONFIG ---
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16) # Legs
]

KPT_VISIBILITY_THRESHOLD = 0.5 

def draw_transparent_skeletons(display_frame, results, scale_x, scale_y):
    """
    Draws thin, semi-transparent bounding boxes and skeletons.
    Draws fully opaque text (IDs and Keypoint Indices) on top for readability.
    """
    if results[0].boxes.id is None:
        return display_frame 

    # Create an overlay for the transparent shapes
    overlay = display_frame.copy()
    
    # List to store text so we can draw it AFTER the transparency blend
    opaque_text_list = [] 
    
    ids = results[0].boxes.id.cpu().numpy().astype(int)
    bboxes = results[0].boxes.xyxy.cpu().numpy()
    kpts_data = results[0].keypoints.data.cpu().numpy()

    for i, p_id in enumerate(ids):
        # 1. Scale and Draw Bounding Box (on overlay)
        x1 = int(bboxes[i][0] * scale_x)
        y1 = int(bboxes[i][1] * scale_y)
        x2 = int(bboxes[i][2] * scale_x)
        y2 = int(bboxes[i][3] * scale_y)
        
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 255), 1)

        # Save Person ID to be drawn opaquely
        opaque_text_list.append((f"ID:{p_id}", (x1, max(15, y1 - 5)), 0.5, (255, 0, 255)))

        # 2. Scale and Draw Individual Keypoints (on overlay)
        person_kpts = kpts_data[i] 
        scaled_kpts = []

        for kpt_idx in range(17):
            kpt_x, kpt_y, kpt_conf = person_kpts[kpt_idx]
            
            if kpt_x == 0 and kpt_y == 0:
                scaled_kpts.append((0, 0, kpt_conf))
                continue

            scaled_x = int(kpt_x * scale_x)
            scaled_y = int(kpt_y * scale_y)
            scaled_kpts.append((scaled_x, scaled_y, kpt_conf))

            kpt_color = (0, 255, 0) # Green
            if kpt_conf < KPT_VISIBILITY_THRESHOLD:
                kpt_color = (0, 0, 255) # Red

            cv2.circle(overlay, (scaled_x, scaled_y), 2, kpt_color, cv2.FILLED)
            
            # Save the Keypoint Index (0-16) to be drawn opaquely next to the joint
            opaque_text_list.append((str(kpt_idx), (scaled_x + 3, scaled_y + 3), 0.35, (255, 255, 255)))

        # 3. Draw Skeleton Structure (on overlay)
        for start_node, end_node in SKELETON_EDGES:
            p_start = scaled_kpts[start_node]
            p_end = scaled_kpts[end_node]
            
            if (p_start[2] > KPT_VISIBILITY_THRESHOLD and p_end[2] > KPT_VISIBILITY_THRESHOLD):
                cv2.line(overlay, (p_start[0], p_start[1]), (p_end[0], p_end[1]), (0, 255, 255), 1) 

    # 4. Blend the shapes with the main frame (50% transparency)
    cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)
    
    # 5. Draw all the text at 100% opacity on top of the blended frame
    for text, position, font_scale, color in opaque_text_list:
        cv2.putText(display_frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

    return display_frame

def draw_hud_overlay(display_frame, results, fps):
    """
    Draws the static, semi-transparent dashboard on the left side of the screen.
    """
    hud_width = 240
    hud_height = 30 
    
    ids = []
    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        box_confs = results[0].boxes.conf.cpu().numpy()
        kpts_data = results[0].keypoints.data.cpu().numpy()
        hud_height += len(ids) * 105 
    
    overlay = display_frame.copy()
    
    cv2.rectangle(overlay, (10, 10), (10 + hud_width, 10 + hud_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, display_frame, 0.6, 0, display_frame)
    
    cv2.putText(display_frame, f"FPS: {fps:.1f}", (15, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if len(ids) > 0:
        y_offset = 55
        for i, p_id in enumerate(ids):
            box_conf = box_confs[i]
            person_kpts = kpts_data[i]
            
            cv2.putText(display_frame, f"--- [ID: {p_id}] Box Conf: {box_conf:.2f} ---", 
                        (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            y_offset += 18
            
            kpt_scores = [f"{idx}:.{int(person_kpts[idx][2] * 100):02d}" for idx in range(17)]
            for j in range(0, 17, 4):
                chunk = kpt_scores[j:j+4]
                line_text = "   ".join(chunk)
                cv2.putText(display_frame, line_text, (15, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 15
                
            y_offset += 10 
            
    return display_frame

def run_live_viewer():
    print(f"[INFO] Loading YOLO Engine from: {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH, task='pose')
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FPS, 35)
    
    disp_w, disp_h = 1280, 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, disp_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, disp_h)

    if not cap.isOpened():
        print("[ERROR] Camera not found.")
        return

    print("[INFO] Live view started. Press 'q' to quit.")

    scale_x = disp_w / IMGSZ[1]
    scale_y = disp_h / IMGSZ[0]

    prev_time = 0 

    while True:
        success, frame = cap.read()
        if not success:
            print("[WARNING] Failed to grab frame.")
            break

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if prev_time > 0 else 0.0
        prev_time = current_time

        # 1. Down-sample
        small_frame = cv2.resize(frame, (IMGSZ[1], IMGSZ[0]))

        results = model.track(
            small_frame,             
            persist=True, 
            verbose=False, 
            imgsz=IMGSZ,
            conf=0.1, 
            tracker=r"yolo/custom_botsort.yaml" 
        )

        # 2. Upscale
        display_frame = cv2.resize(small_frame, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)

        # 3. Draw Skeletons (now with opaque indices)
        display_frame = draw_transparent_skeletons(display_frame, results, scale_x, scale_y)

        # 4. Draw HUD
        display_frame = draw_hud_overlay(display_frame, results, fps)

        cv2.imshow("InvisiGuard Down-Sampled View", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_viewer()