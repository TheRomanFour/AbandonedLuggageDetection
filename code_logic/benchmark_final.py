import cv2
import numpy as np
from ultralytics import YOLO
import os
import json
import pandas as pd
import torch
from collections import deque
import radius_logic as rl

# --- CONFIGURATION ---
BASE_DIR = "/mnt/SamsungSSD/IvanVrsalovic/Prtljaga/Datasets/Uniri dataset/Dataset/"
ANNOTATION_DIR = os.path.join(BASE_DIR, "annotations")
MODEL_PATH = "/mnt/SamsungSSD/IvanVrsalovic/Prtljaga/Datasets/Uniri dataset/yolov11/yolo11s/weights/best.pt"
MASTER_CSV = "master_results_dynamic_fixed.csv"

# Global Parameters
T_OWN = 2.0  # Seconds a person must stay near a bag to "own" it
T_ABANDON = 5.0  # Seconds a bag must be alone to trigger alarm
GRACE_PERIOD_FRAMES = 10  # Frames to "remember" an ID if it flickers

METHODS = {
    "FIXED_100": rl.radius_strictly_fixed_100,
    "FIXED_200": rl.radius_strictly_fixed_200,
    "BASE_DYNAMIC": rl.radius_fixed_equivalent,
    "HEIGHT_ONLY": rl.radius_height_adaptive,
    "VELOCITY_ONLY": rl.radius_velocity_dynamic,
    "ULTIMATE_V1.3": rl.radius_ultimate_dynamic_v1_3,
    "ULTIMATE_V1.4": rl.radius_ultimate_dynamic_v1_4
}

def run_benchmark():
    video_files = sorted([f for f in os.listdir(BASE_DIR) if f.endswith(".mp4")])
    results = []
    
    # Load model once outside the loops
    model = YOLO(MODEL_PATH)

    for m_name, m_func in METHODS.items():
        print(f"\n>>> TESTING METHOD: {m_name}")
        
        for v_name in video_files:
            v_path = os.path.join(BASE_DIR, v_name)
            j_path = os.path.join(ANNOTATION_DIR, v_name.replace(".mp4", ".json"))
            if not os.path.exists(j_path): continue
            
            cap = cv2.VideoCapture(v_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            
            p_history, s_history, p_info = {}, {}, {}
            proximity_timers, abandon_timers = {}, {}
            detected_alarms = []
            frame_idx = 0

            # Persistent detections for stride handling
            last_curr_p, last_curr_s = {}, {}

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                t_curr = frame_idx / fps

                # 1. TRACKING (Every 2nd frame for speed)
                if frame_idx % 2 == 0:
                    res = model.track(frame, persist=True, classes=[0, 1], verbose=False)
                    curr_p, curr_s = {}, {}
                    if res and res[0].boxes.id is not None:
                        for box in res[0].boxes:
                            obj_id, cls = int(box.id[0]), int(box.cls[0])
                            coords = box.xyxy[0].cpu().numpy()
                            # Use BOTTOM-CENTER for person to get "feet" position
                            cx = int((coords[0] + coords[2]) / 2)
                            cy = int(coords[3]) if cls == 1 else int((coords[1] + coords[3]) / 2)
                            h_px = abs(coords[3] - coords[1])
                            
                            if cls == 1: # Person
                                curr_p[obj_id] = (cx, cy)
                                p_info.setdefault(obj_id, {"max_h": 0})
                                p_info[obj_id]["max_h"] = max(p_info[obj_id]["max_h"], h_px)
                                p_history.setdefault(obj_id, deque(maxlen=30)).append((cx, cy))
                            else: # Luggage
                                curr_s[obj_id] = (cx, cy)
                                s_history.setdefault(obj_id, deque(maxlen=30)).append((cx, cy))
                    last_curr_p, last_curr_s = curr_p, curr_s

                # 2. PROXIMITY & OWNERSHIP LOGIC
                for sid, s_xy in last_curr_s.items():
                    guards = 0
                    for pid, p_xy in last_curr_p.items():
                        # Radius Calculation
                        neighbors = sum(1 for opid, opxy in last_curr_p.items() if opid != pid and np.linalg.norm(np.array(p_xy)-np.array(opxy)) < 150)
                        
                        # Use a fallback stats dict for v1.4 if no samples yet
                        R = m_func(p_info=p_info, pid=pid, p_history=p_history, s_history=s_history, sid=sid, neighbors=neighbors, stats={'n':0})
                        dist = np.linalg.norm(np.array(s_xy) - np.array(p_xy))
                        
                        if dist < R:
                            proximity_timers.setdefault((sid, pid), t_curr)
                            if (t_curr - proximity_timers[(sid, pid)]) >= T_OWN:
                                guards += 1
                        else:
                            # Instead of immediate pop, you could add a grace period here
                            proximity_timers.pop((sid, pid), None)

                    # 3. ALARM LOGIC
                    if guards == 0:
                        abandon_timers.setdefault(sid, t_curr)
                        if (t_curr - abandon_timers[sid]) > T_ABANDON:
                            # Avoid spamming multiple alarms for the same bag
                            if not any(abs(t_curr - a) < 10.0 for a in detected_alarms):
                                detected_alarms.append(t_curr)
                    else:
                        abandon_timers.pop(sid, None)

                frame_idx += 1
            cap.release()

            # --- SCIENTIFIC EVALUATION ---
            with open(j_path, "r") as f: gt_data = json.load(f)
            gt_events = [e for e in gt_data.get("events", []) if e.get("type") == "abandonment"]
            
            # Logic: If GT says abandon, did we find an alarm in that window?
            if not gt_events:
                status = "TN" if not detected_alarms else "FP"
            else:
                is_caught = False
                for event in gt_events:
                    e_start = event.get("start_time")
                    # Success window: alarm must happen after abandonment + buffer
                    if any(e_start <= a <= (e_start + T_ABANDON + 10.0) for a in detected_alarms):
                        is_caught = True
                status = "TP" if is_caught else "FN"

            results.append({
                "Method": m_name, "Video": v_name, "Status": status,
                "Alarms_Found": len(detected_alarms)
            })
            print(f"  Result: {v_name} -> {status}")

    pd.DataFrame(results).to_csv(MASTER_CSV, index=False)
    print(f"\nBenchmark Complete! Saved to {MASTER_CSV}")

if __name__ == "__main__":
    run_benchmark()