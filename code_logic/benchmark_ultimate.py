import cv2
import numpy as np
from ultralytics import YOLO
import os
import json
import pandas as pd
import torch
from collections import deque
import radius_logic as rl
from datetime import datetime

# --- KONFIGURACIJA PUTEVA ---
BASE_DIR = "/mnt/SamsungSSD/IvanVrsalovic/Prtljaga/Datasets/Uniri dataset/Dataset/"
ANNOTATION_DIR = os.path.join(BASE_DIR, "annotations_update")
MODEL_PATH = "/mnt/SamsungSSD/IvanVrsalovic/Prtljaga/Datasets/Uniri dataset/yolov11/yolo11s/weights/best.pt"

# Generiranje imena datoteke s vremenskim žigom
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
MASTER_CSV = f"scientific_master_results_{timestamp}.csv"

# Metode iz tvog radius_logic.py
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
    
    # Inicijalizacija YOLO modela (jednom, radi performansi)
    model = YOLO(MODEL_PATH)

    for m_name, m_func in METHODS.items():
        print(f"\n>>> TESTIRANJE METODE: {m_name}")
        
        for v_name in video_files:
            v_path = os.path.join(BASE_DIR, v_name)
            j_path = os.path.join(ANNOTATION_DIR, v_name.replace(".mp4", ".json"))
            
            if not os.path.exists(j_path):
                continue
            
            # --- 1. UČITAVANJE KONFIGURACIJE IZ JSON-A ---
            with open(j_path, "r") as f:
                gt_data = json.load(f)
            
            timers = gt_data.get("timers", {})
            # Dinamički pragovi iz JSON-a (uz default vrijednosti)
            t_abandon_thresh = timers.get("time_to_abandoned", 5.0)
            t_own_thresh = timers.get("time_to_own", 2.0)
            fps_json = gt_data.get("fps", 30.0)

            print(f"  Obrada: {v_name} (Thresh: {t_abandon_thresh}s)...", end="\r")
            
            cap = cv2.VideoCapture(v_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or fps_json
            
            p_history, s_history, p_info = {}, {}, {}
            proximity_timers, abandon_timers = {}, {}
            alarm_time = None
            frame_idx = 0

            # Liste za statistiku po videu
            video_radii = []
            video_distances = []
            video_guards_count = []

            # --- 2. PETLJA OBRADE OKVIRA ---
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # Standardizacija rezolucije za konzistentnost radijusa
                frame = cv2.resize(frame, (800, 600))
                t_curr = frame_idx / fps

                # Tracking (vid_stride=2 preskače svaki drugi okvir radi brzine)
                res = model.track(frame, persist=True, classes=[0, 1], verbose=False, vid_stride=2)
                
                curr_p, curr_s = {}, {}
                if res and res[0].boxes.id is not None:
                    for box in res[0].boxes:
                        obj_id, cls = int(box.id[0]), int(box.cls[0])
                        coords = box.xyxy[0].cpu().numpy()
                        
                        # Pozicioniranje: Noge za osobu, Centar za prtljagu
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

                # Logika radijusa i vlasništva
                for sid, s_xy in curr_s.items():
                    guards = 0
                    for pid, p_xy in curr_p.items():
                        # Brojanje susjeda za naprednu logiku (ULTIMATE)
                        neighbors = sum(1 for opid, opxy in curr_p.items() if opid != pid and np.linalg.norm(np.array(p_xy)-np.array(opxy)) < 150)
                        
                        # Izračun radijusa putem tvojih funkcija
                        R = m_func(p_info=p_info, pid=pid, p_history=p_history, s_history=s_history, sid=sid, neighbors=neighbors)
                        dist = np.linalg.norm(np.array(s_xy) - np.array(p_xy))
                        
                        video_radii.append(R)
                        video_distances.append(dist)

                        if dist < R:
                            proximity_timers.setdefault((sid, pid), t_curr)
                            # Provjera je li osoba provela dovoljno vremena uz torbu da postane "čuvar"
                            if (t_curr - proximity_timers[(sid, pid)]) >= t_own_thresh:
                                guards += 1
                        else:
                            proximity_timers.pop((sid, pid), None)
                    
                    video_guards_count.append(guards)

                    # --- ALARM LOGIKA ---
                    if guards == 0:
                        abandon_timers.setdefault(sid, t_curr)
                        if (t_curr - abandon_timers[sid]) > t_abandon_thresh:
                            alarm_time = t_curr
                            break
                    else:
                        abandon_timers.pop(sid, None)
                
                if alarm_time: break
                frame_idx += 2 # Mora pratiti vid_stride
            
            cap.release()

            # --- 3. ZNANSTVENA EVALUACIJA (TP, FP, TN, FN) ---
            gt_events = [e for e in gt_data.get("events", []) if e.get("type") == "abandonment"]
            is_gt_abandon = len(gt_events) > 0
            
            status = ""
            if is_gt_abandon:
                # Provjera je li alarm pao unutar fer prozora (start_time + tolerance)
                is_correct_alarm = False
                for event in gt_events:
                    e_start = event.get("start_time")
                    e_tol = event.get("tolerance", 15.0)
                    if alarm_time and (e_start <= alarm_time <= (e_start + t_abandon_thresh + e_tol)):
                        is_correct_alarm = True
                
                status = "TP (Success)" if is_correct_alarm else "FN (Miss)"
            else:
                status = "TN (Correct Safe)" if not alarm_time else "FP (False Alarm)"

            # Prikupljanje svih podataka za ovaj video
            results.append({
                "Method": m_name,
                "Video": v_name,
                "Status": status,
                "GT_Label": "ABANDON" if is_gt_abandon else "SAFE",
                "System_Label": "ALARM" if alarm_time else "SAFE",
                "Alarm_Time": round(alarm_time, 2) if alarm_time else "N/A",
                "Avg_Radius": round(np.mean(video_radii), 1) if video_radii else 0,
                "Avg_Guards": round(np.mean(video_guards_count), 2) if video_guards_count else 0,
                "Max_Dist_PX": round(max(video_distances), 1) if video_distances else 0,
                "Max_H_Observed": round(max([p["max_h"] for p in p_info.values()]), 1) if p_info else 0
            })

    # --- 4. SPREMANJE REZULTATA ---
    df = pd.DataFrame(results)
    df.to_csv(MASTER_CSV, index=False)
    
    print(f"\n\nBenchmark završen!")
    print(f"Glavni rezultati: {MASTER_CSV}")

if __name__ == "__main__":
    run_benchmark()