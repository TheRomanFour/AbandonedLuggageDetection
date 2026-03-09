import cv2
import numpy as np
from ultralytics import YOLO
import torch
import json
import os
from collections import deque
import radius_logic as rl  # Assumes your provided script is named radius_logic.py

# --- CONFIGURATION ---
VIDEO_ID = "0025"  # Change to the desired video ID
BASE_PATH = "/mnt/SamsungSSD/IvanVrsalovic/Prtljaga/Datasets/Uniri dataset/Dataset"
VIDEO_PATH = f"{BASE_PATH}/{VIDEO_ID}.mp4"
JSON_PATH = f"{BASE_PATH}/annotations_update/{VIDEO_ID}.json"
MODEL_PATH = "/mnt/SamsungSSD/IvanVrsalovic/Prtljaga/Datasets/Uniri dataset/yolov11/yolo11s/weights/best.pt"

SELECTED_METHOD = rl.radius_ultimate_dynamic_v1_4
METHOD_NAME = "ULTIMATE_DYN_V1.4"
TTA_THRESHOLD = 5.0 

def load_ground_truth(path):
    """Parses nested JSON events lisqt for abandonment timestamps."""
    if not os.path.exists(path): return None
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            events = data.get("events", [])
            if events:
                main_event = events[0]
                etype = "ABANDON" if main_event.get("type") == "abandonment" else "SAFE"
                stime = main_event.get("start_time", 9999)
                tolerance = main_event.get("tolerance", 10)
                return {"type": etype, "start_time": stime, "tolerance": tolerance}
            return {"type": "SAFE", "start_time": 9999, "tolerance": 0}
    except Exception: return None

def run_visual_test():
    gt = load_ground_truth(JSON_PATH)
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    # State tracking
    p_history, s_history, p_info = {}, {}, {}
    proximity_timers, abandon_timers = {}, {}
    dist_stats_history = {} # For V1.4 Statistical Layer: {(sid, pid): deque of distances}
    last_res_p, last_res_s = {}, {}
    first_alarm_time = None

    print(f"--- STARTING VISUALIZATION: {VIDEO_ID} ---")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_vis = cv2.resize(frame, (1280, 720)) # Larger for better visibility
        t_curr = frame_idx / fps

        # 1. YOLO TRACKING
        if frame_idx % 2 == 0:
            res = model.track(frame_vis, persist=True, classes=[0, 1], verbose=False)
            curr_p, curr_s = {}, {}
            if res and res[0].boxes.id is not None:
                for box in res[0].boxes:
                    obj_id = int(box.id[0])
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    cx, cy = (coords[0] + coords[2]) // 2, (coords[1] + coords[3]) // 2
                    
                    if int(box.cls[0]) == 1: # PERSON
                        curr_p[obj_id] = {'cnt': (cx, cy), 'box': coords}
                        h_px = abs(coords[3] - coords[1])
                        p_info.setdefault(obj_id, {"max_h": 0})
                        p_info[obj_id]["max_h"] = max(p_info[obj_id]["max_h"], h_px)
                        p_history.setdefault(obj_id, deque(maxlen=30)).append((cx, cy))
                    else: # LUGGAGE
                        curr_s[obj_id] = {'cnt': (cx, cy), 'box': coords}
                        s_history.setdefault(obj_id, deque(maxlen=30)).append((cx, cy))
            last_res_p, last_res_s = curr_p, curr_s

        # 2. RADIUS LOGIC & ENHANCED DRAWING
        for sid, s_data in last_res_s.items():
            scx, scy = s_data['cnt']
            guards_found = 0

            for pid, p_data in last_res_p.items():
                pcx, pcy = p_data['cnt']
                
                # Calculate neighbors for density factor
                others = sum(1 for opid, op in last_res_p.items() if opid != pid and np.linalg.norm(np.array(p_data['cnt'])-np.array(op['cnt'])) < 150)
                
                # Track Connection/Sticky State
                is_sticky = False
                if (sid, pid) in proximity_timers:
                    elapsed_prox = t_curr - proximity_timers[(sid, pid)]
                    if elapsed_prox >= 2.0: is_sticky = True

                # Calculate Distance Stats for V1.4
                stats_input = None
                pair_key = (sid, pid)
                dist = np.linalg.norm(np.array(s_data['cnt']) - np.array(p_data['cnt']))
                if is_sticky:
                    dist_stats_history.setdefault(pair_key, deque(maxlen=50)).append(dist)
                    if len(dist_stats_history[pair_key]) > 15:
                        stats_input = {
                            'mean': np.mean(dist_stats_history[pair_key]),
                            'std': np.std(dist_stats_history[pair_key]),
                            'n': len(dist_stats_history[pair_key])
                        }

                # CALL V1.4 LOGIC
                R = SELECTED_METHOD(p_info=p_info, pid=pid, p_history=p_history, 
                                    s_history=s_history, sid=sid, neighbors=others, 
                                    is_sticky=is_sticky, stats=stats_input)

                # Visual Feedback for Connection
                if dist < R:
                    proximity_timers.setdefault((sid, pid), t_curr)
                    if is_sticky:
                        guards_found += 1
                        l_color = (0, 255, 0) # Green: Connected
                    else:
                        l_color = (0, 165, 255) # Orange: Wait (2s)
                else:
                    proximity_timers.pop((sid, pid), None)
                    l_color = (150, 150, 150) # Gray: Out of Radius

                # DRAWING COMPONENTS
                # A. The Radius Circle
                cv2.circle(frame_vis, (pcx, pcy), int(R), l_color, 2)
                # B. The Connection Line
                cv2.line(frame_vis, (pcx, pcy), (scx, scy), l_color, 1)
                
                # C. Velocity Vector (Cyan Arrow)
                vel = rl.get_velocity_vec(p_history.get(pid, []))
                if np.linalg.norm(vel) > 2:
                    cv2.arrowedLine(frame_vis, (pcx, pcy), (int(pcx+vel[0]*2), int(pcy+vel[1]*2)), (255, 255, 0), 2)

                # D. Logic Dashboard (Text next to person)
                v_mag = np.linalg.norm(vel)
                txt = f"R:{int(R)} H:{int(p_info[pid]['max_h'])} V:{v_mag:.1f} N:{others}"
                cv2.putText(frame_vis, txt, (pcx - 40, pcy - int(R) - 10), 0, 0.4, (255, 255, 255), 1)

            # 3. ALARM LOGIC
            if guards_found == 0:
                abandon_timers.setdefault(sid, t_curr)
                elapsed = t_curr - abandon_timers[sid]
                if elapsed > TTA_THRESHOLD:
                    if first_alarm_time is None: first_alarm_time = t_curr
                    cv2.rectangle(frame_vis, (s_data['box'][0], s_data['box'][1]), (s_data['box'][2], s_data['box'][3]), (0,0,255), 3)
                    cv2.putText(frame_vis, f"!!! ALARM {elapsed:.1f}s !!!", (s_data['box'][0], s_data['box'][1]-10), 0, 0.7, (0,0,255), 2)
                else:
                    cv2.putText(frame_vis, f"ABANDONED: {elapsed:.1f}s", (s_data['box'][0], s_data['box'][1]-10), 0, 0.5, (0,165,255), 1)
            else:
                abandon_timers.pop(sid, None)

        # 4. OVERLAYS (GT and Information)
        cv2.rectangle(frame_vis, (10, 10), (300, 120), (0,0,0), -1)
        cv2.putText(frame_vis, f"Method: {METHOD_NAME}", (20, 40), 0, 0.6, (0, 255, 255), 2)
        cv2.putText(frame_vis, f"Video: {VIDEO_ID} | {t_curr:.2f}s", (20, 70), 0, 0.5, (255, 255, 255), 1)
        if gt:
            gt_col = (0, 255, 0) if gt['type'] == "SAFE" else (0, 200, 255)
            cv2.putText(frame_vis, f"GT: {gt['type']} @ {gt['start_time']}s", (20, 100), 0, 0.5, gt_col, 1)

        cv2.imshow("Scientific Visualization - Ultimate Dynamic V1.4", frame_vis)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_visual_test()