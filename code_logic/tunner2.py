import cv2
import numpy as np
from ultralytics import YOLO
import os
import json
import pandas as pd
from datetime import datetime
from itertools import product

# --- CONFIGURATION ---
VIDEO_DIR = "/mnt/SamsungSSD/IvanVrsalovic/Prtljaga/Datasets/Uniri dataset/Dataset/"
ANNOTATION_DIR = "/mnt/SamsungSSD/IvanVrsalovic/Prtljaga/Datasets/Uniri dataset/Dataset/annotations_update/"
MODEL_PATH = "/mnt/SamsungSSD/IvanVrsalovic/Prtljaga/Datasets/Uniri dataset/yolov11/yolo11s/weights/best.pt"

# --- REFINED PARAMETER GRID (Finding the "Best of the Best") ---
# We are expanding the ranges based on the previous winners (S:1.1, D:0.3)
param_grid = {
    "v_mul": [0.4, 0.5, 0.6, 0.7, 0.8],      # Testing more steps in velocity
    "sticky": [0.9, 1.0, 1.1, 1.2],         # Testing lower values (previous winner was 1.1)
    "dens": [0.3, 0.4, 0.5]                 # Testing higher values (previous winner was 0.3)
}

# Generate METHODS list dynamically
METHODS = []
for v, s, d in product(param_grid["v_mul"], param_grid["sticky"], param_grid["dens"]):
    METHODS.append({
        "name": f"V{v}_S{s}_D{d}",
        "v_mul": v, "sticky": s, "dens": d
    })

def run_fine_tuned_benchmarking():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"Best_of_Best_Search_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO(MODEL_PATH)
    video_files = sorted([f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")])
    all_results = []

    for v_name in video_files:
        j_path = os.path.join(ANNOTATION_DIR, v_name.replace(".mp4", ".json"))
        if not os.path.exists(j_path): continue
        
        with open(j_path, "r") as f: gt_data = json.load(f)
        gt_events = [e for e in gt_data.get("events", []) if e.get("type") == "abandonment" and e.get("start_time") is not None]
        
        print(f"🔍 Testing {len(METHODS)} configs on: {v_name}")
        cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, v_name))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        method_timers = {m['name']: {} for m in METHODS}
        method_alarms = {m['name']: [] for m in METHODS}

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            results = model.track(frame, persist=True, verbose=False)
            curr_time = frame_idx / fps
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes
                # Filter indices for Persons and Bags once per frame to save speed
                persons = [b for b in boxes if int(b.cls[0]) == 0]
                bags = [b for b in boxes if int(b.cls[0]) == 1]

                for m in METHODS:
                    m_name = m['name']
                    for b_box in bags:
                        bid = int(b_box.id[0])
                        is_guarded = False
                        
                        for p_box in persons:
                            h = abs(p_box.xyxy[0][3] - p_box.xyxy[0][1])
                            # Logic formula
                            radius = (h * 0.75) * m['sticky'] * (1 - m['dens'])
                            dist = np.linalg.norm(b_box.xyxy[0][:2] - p_box.xyxy[0][:2])
                            if dist < radius:
                                is_guarded = True; break
                        
                        if not is_guarded:
                            method_timers[m_name][bid] = method_timers[m_name].get(bid, 0) + (1/fps)
                            if method_timers[m_name][bid] >= 10.0:
                                method_alarms[m_name].append(curr_time)
                        else:
                            method_timers[m_name][bid] = 0
            frame_idx += 1
        cap.release()

        # Evaluate current video against all methods
        for m in METHODS:
            alarms = method_alarms[m['name']]
            if not gt_events:
                status = "FP" if alarms else "TN"
            else:
                hit = any(any(e["start_time"] <= a <= (e["start_time"] + 30) for a in alarms) for e in gt_events)
                status = "TP" if hit else "FN"

            all_results.append({
                "Method": m['name'], "Video": v_name, "Status": status,
                "V_Mul": m['v_mul'], "Sticky": m['sticky'], "Density": m['dens']
            })

    # --- FINAL AGGREGATION ---
    df_raw = pd.DataFrame(all_results)
    summary_stats = []
    for m in METHODS:
        m_df = df_raw[df_raw['Method'] == m['name']]
        tp, fp, fn = (m_df['Status'] == "TP").sum(), (m_df['Status'] == "FP").sum(), (m_df['Status'] == "FN").sum()
        
        prec = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
        rec = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        
        summary_stats.append({
            "Method": m['name'], "F1": round(f1, 2), "Recall": round(rec, 2),
            "Precision": round(prec, 2), "TP": tp, "FP": fp, "FN": fn,
            "V_Mul": m['v_mul'], "Sticky": m['sticky'], "Dens": m['dens']
        })

    summary_df = pd.DataFrame(summary_stats).sort_values(by="F1", ascending=False)
    summary_df.to_csv(os.path.join(output_dir, "best_of_the_best_leaderboard.csv"), index=False)
    
    print(f"\nOptimization Complete. Best Config Found:")
    print(summary_df.head(1))

if __name__ == "__main__":
    run_fine_tuned_benchmarking()