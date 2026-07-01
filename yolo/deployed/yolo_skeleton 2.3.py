#yolo26 headless daily file rotation auto reconnect unix

import cv2
import time
import sys
import os
import datetime
import csv
import traceback
from pathlib import Path

# --- CONFIGURATION ---
WEBCAM_INDEX = 0      

# 1. RESOLUTION: Matches the export log (352 height, 640 width)
IMGSZ = (352, 640)           

# 2. FOLDER NAME: Absolute path
MODEL_PATH = r"C:\Users\jeffr\Desktop\yolo26n-pose_openvino_model"

# 3. LOGGING DIRECTORY: Hardcoded Windows absolute path
LOG_DIR = Path(r"C:\Users\jeffr\Desktop\InvisiGuard\shared_data\logs")

# --- SETUP CHECKS ---
LOG_DIR.mkdir(parents=True, exist_ok=True)

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: Library missing. Run: pip install ultralytics")
    sys.exit(1)

if not os.path.exists(MODEL_PATH):
    print(f"\n[CRITICAL ERROR] Model folder not found: {MODEL_PATH}")
    sys.exit(1)


class HeadlessTracker:
    def __init__(self, model):
        self.model = model
        self.video_cap = None
        
        # Log file handlers
        self.log_date = None
        self.pose_file = None
        self.pose_writer = None
        self.people_file = None
        self.people_writer = None
        
        self.frame_count = 0
        self.last_print_time = time.time()

        self.init_logs()

    def init_logs(self):
        """Creates or opens today's CSV files and writes headers if they are new."""
        self.log_date = datetime.date.today()
        date_str = self.log_date.strftime("%Y-%m-%d")
        
        pose_path = LOG_DIR / f"pose_{date_str}.csv"
        people_path = LOG_DIR / f"num_people_{date_str}.csv"
        
        pose_needs_header = not pose_path.exists()
        people_needs_header = not people_path.exists()
        
        self.pose_file = open(pose_path, 'a', newline='', buffering=1)
        self.pose_writer = csv.writer(self.pose_file)
        
        self.people_file = open(people_path, 'a', newline='', buffering=1)
        self.people_writer = csv.writer(self.people_file)
        
        if pose_needs_header:
            header = ["Timestamp", "ID", "BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2", "Box_Conf"] + [f"KP{i}_{c}" for i in range(17) for c in ["X","Y","C"]]
            self.pose_writer.writerow(header)
            
        if people_needs_header:
            self.people_writer.writerow(["Timestamp", "NumPeople"])
            
        print(f"[REC] Auto-Logging initialized for {date_str}")

    def rotate_logs(self):
        """Closes yesterday's files and initializes today's files."""
        print("[REC] Midnight passed. Rotating logs...")
        self.close_logs()
        self.init_logs()

    def close_logs(self):
        """Safely closes open file handles."""
        if self.pose_file: self.pose_file.close()
        if self.people_file: self.people_file.close()

    def run(self):
        self.video_cap = cv2.VideoCapture(WEBCAM_INDEX)
        
        self.video_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.video_cap.set(cv2.CAP_PROP_FPS, 60)
        self.video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.video_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        
        if not self.video_cap.isOpened():
            raise RuntimeError("Camera not found or unavailable. Check connection.")

        print("[INFO] Camera initialized. Tracking started.")

        while True:
            # 1. CHECK DAILY ROTATION
            if datetime.date.today() != self.log_date:
                self.rotate_logs()

            # 2. READ FRAME
            success, frame = self.video_cap.read()
            if not success:
                # Breaking out of the inner loop will trigger the auto-restart wrapper
                print("\n[WARNING] Failed to grab frame. Camera disconnected?")
                break 

            # --- UPDATED: Unix Timestamp (Float) ---
            capture_ts = time.time()

            # 3. INFERENCE
            results = self.model.track(
                frame, 
                persist=True, 
                verbose=False, 
                imgsz=IMGSZ,
                conf=0.1,
                tracker=r"C:\Users\jeffr\Desktop\InvisiGuard\custom_botsort.yaml" 
            )
            
            # 4. SAVE DATA
            person_count = 0
            if results[0].boxes.id is not None:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                bboxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                kpts = results[0].keypoints.data.cpu().numpy()
                person_count = len(ids)

                for i, p_id in enumerate(ids):
                    row = [capture_ts, p_id]
                    row.extend(bboxes[i].flatten())
                    row.append(confs[i])
                    row.extend(kpts[i].flatten())
                    self.pose_writer.writerow(row)

            self.people_writer.writerow([capture_ts, person_count])

            # 5. CONSOLE HEARTBEAT (Prints stats every 10 seconds)
            self.frame_count += 1
            if time.time() - self.last_print_time >= 10.0:
                fps = self.frame_count / 10.0
                # Note: Formatting capture_ts to .3f just makes the console output cleaner, 
                # but the CSV still gets the full raw float.
                print(f"[{capture_ts:.3f}] Heartbeat | FPS: {fps:.1f} | People Visible: {person_count}")
                self.frame_count = 0
                self.last_print_time = time.time()


if __name__ == "__main__":
    # Load model ONCE so we don't waste time reloading it on camera disconnects
    print(f"[INFO] Loading YOLO Engine from: {MODEL_PATH}...")
    try:
        yolo_model = YOLO(MODEL_PATH, task='pose') 
        print("[INFO] Engine Ready.")
    except Exception as e:
        print(f"[CRITICAL] Failed to load model: {e}")
        sys.exit(1)

    # --- AUTO-RESTART WRAPPER ---
    while True:
        tracker = None
        try:
            tracker = HeadlessTracker(yolo_model)
            tracker.run()
        except KeyboardInterrupt:
            print("\n[INFO] Manual termination requested. Exiting...")
            if tracker:
                tracker.close_logs()
                if tracker.video_cap: tracker.video_cap.release()
            sys.exit(0)
        except Exception as e:
            print(f"\n[ERROR] Crash detected: {e}")
            traceback.print_exc()
        finally:
            print("[INFO] Cleaning up resources...")
            if tracker:
                tracker.close_logs()
                if tracker.video_cap: 
                    tracker.video_cap.release()
            
            print("[INFO] Auto-restarting in 5 seconds...\n")
            time.sleep(5)