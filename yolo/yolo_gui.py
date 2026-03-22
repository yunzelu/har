import tkinter as tk
from tkinter import ttk
import cv2
import time
import sys
import os
import datetime
import csv
from PIL import Image, ImageTk

# --- CONFIGURATION ---
WEBCAM_INDEX = 0      

# 1. RESOLUTION: Matches the export log (352 height, 640 width)
IMGSZ = (352, 640)           

# 2. FOLDER NAME: Matches what the log actually created
MODEL_PATH = 'yolo/yolov8n-pose_openvino_model/' 

OUTPUT_DIR = "Pose_Data"

# --- SETUP CHECKS ---
if not os.path.exists(OUTPUT_DIR): 
    os.makedirs(OUTPUT_DIR)

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: Library missing. Run: pip install ultralytics")
    sys.exit(1)

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"\n[CRITICAL ERROR] Model folder not found: {MODEL_PATH}")
    print("It seems the export didn't save where we expected.")
    print("Check your folder for 'yolov8n-pose_openvino_model'.\n")
    sys.exit(1)

class WideTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Wide-FOV Tracker (13ms)")
        
        self.running = False
        self.recording = False
        self.video_cap = None
        self.csv_file = None
        self.csv_writer = None

        print(f"[INFO] Loading Wide-FOV Engine from: {MODEL_PATH}...")
        self.model = YOLO(MODEL_PATH, task='pose') 
        print("[INFO] Engine Ready.")

        # --- GUI Layout ---
        # Canvas set to 640x360 (16:9 Aspect Ratio) to match Wide Input
        self.canvas = tk.Canvas(self.root, width=640, height=360, bg="#222")
        self.canvas.pack()

        stats = ttk.Frame(self.root)
        stats.pack(pady=5)
        self.fps_lbl = ttk.Label(stats, text="FPS: 0", font=("Consolas", 12, "bold"))
        self.fps_lbl.pack(side=tk.LEFT, padx=10)
        self.lat_lbl = ttk.Label(stats, text="Lat: 0ms", font=("Consolas", 12))
        self.lat_lbl.pack(side=tk.LEFT, padx=10)
        self.count_lbl = ttk.Label(stats, text="Ppl: 0", font=("Consolas", 12))
        self.count_lbl.pack(side=tk.LEFT, padx=10)

        btns = ttk.Frame(self.root)
        btns.pack(pady=5)
        self.btn_start = ttk.Button(btns, text="Start Camera", command=self.start)
        self.btn_start.grid(row=0, column=0, padx=5)
        self.btn_rec = ttk.Button(btns, text="Record Data", command=self.toggle_rec, state=tk.DISABLED)
        self.btn_rec.grid(row=0, column=1, padx=5)

        self.photo = None
        self.frame_count = 0
        self.last_time = time.time()

    def start(self):
        if not self.running:
            self.video_cap = cv2.VideoCapture(WEBCAM_INDEX)
            
            # --- PERFORMANCE & FOV SETTINGS ---
            # 1. Force MJPG (Solves USB Bottleneck)
            self.video_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
            self.video_cap.set(cv2.CAP_PROP_FPS, 60)
            
            # 2. Force HD Resolution (1280x720)
            # This forces the camera to use the full WIDESCREEN sensor (16:9).
            # The code will resize this to 640x352 automatically.
            self.video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # 3. Low Latency Buffer
            self.video_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
            
            if not self.video_cap.isOpened():
                print("Error: Camera not found.")
                return

            self.running = True
            self.btn_rec.config(state=tk.NORMAL)
            self.btn_start.config(state=tk.DISABLED)
            self.update()

    def toggle_rec(self):
        if not self.recording:
            # Start Recording
            self.recording = True
            self.btn_rec.config(text="Stop Recording")
            
            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            fname = os.path.join(OUTPUT_DIR, f"pose_{ts}.csv")
            
            self.csv_file = open(fname, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            
            # Header: Timestamp, PersonID, Keypoints...
            header = ["Timestamp", "ID"] + [f"KP{i}_{c}" for i in range(17) for c in ["X","Y","C"]]
            self.csv_writer.writerow(header)
            print(f"[REC] Started: {fname}")
        else:
            # Stop Recording
            self.recording = False
            self.btn_rec.config(text="Record Data")
            if self.csv_file: self.csv_file.close()
            print("[REC] Saved.")

    def update(self):
        if not self.running: return

        # 1. GROUND TRUTH TIMESTAMP
        capture_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        success, frame = self.video_cap.read()
        if not success:
            self.root.after(1, self.update)
            return

        t0 = time.time()

        # 2. INFERENCE
        # We explicitly pass the wide shape tuple (352, 640)
        results = self.model.track(
            frame, 
            persist=True, 
            verbose=False, 
            imgsz=IMGSZ,
            conf=0.5
        )
        
        t1 = time.time()
        latency = (t1 - t0) * 1000

        # 3. VISUALIZATION
        annotated_frame = results[0].plot()
        
        # 4. SAVE DATA
        person_count = 0
        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            kpts = results[0].keypoints.data.cpu().numpy()
            person_count = len(ids)

            if self.recording and self.csv_writer:
                for i, p_id in enumerate(ids):
                    row = [capture_ts, p_id]
                    row.extend(kpts[i].flatten())
                    self.csv_writer.writerow(row)

        # 5. UI UPDATE
        self.count_lbl.config(text=f"Ppl: {person_count}")
        
        # Resize for GUI display (Matches canvas size)
        display_frame = cv2.resize(annotated_frame, (640, 360))
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.canvas.create_image(0, 0, image=img_tk, anchor=tk.NW)
        self.photo = img_tk

        self.frame_count += 1
        if time.time() - self.last_time >= 1.0:
            fps = self.frame_count
            self.fps_lbl.config(text=f"FPS: {fps}")
            self.lat_lbl.config(text=f"Lat: {latency:.1f}ms")
            self.frame_count = 0
            self.last_time = time.time()

        self.root.after(1, self.update)

if __name__ == "__main__":
    root = tk.Tk()
    app = WideTrackerApp(root)
    root.mainloop()