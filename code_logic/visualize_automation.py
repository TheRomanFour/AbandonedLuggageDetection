
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import os
import radius_logic as rl

# --- CONFIGURATION ---
VIDEO_ID = "0037"
TARGET_SAVE_TIME = 5.0  # <--- Change this for the specific "Golden Moment"
BASE_PATH = "/mnt/SamsungSSD/IvanVrsalovic/Prtljaga/Datasets/Uniri dataset/Dataset"
VIDEO_PATH = f"{BASE_PATH}/{VIDEO_ID}.mp4"
MODEL_PATH = "/mnt/SamsungSSD/IvanVrsalovic/Prtljaga/Datasets/Uniri dataset/yolov11/yolo11s/weights/best.pt"


# Methods to compare. Note: Velocity will ONLY show if the name contains 'DYNAMIC' or 'ULTIMATE'
METHODS_TO_RUN = [
    (rl.radius_strictly_fixed_150, "FIXED_150"),
    (rl.radius_height_adaptive, "HEIGHT_ADAPTIVE"),
    (rl.radius_velocity_dynamic, "VELOCITY_DYNAMIC"),
    (rl.radius_fixed_equivalent, "FIXED_EQUIVALENT"),
    (rl.radius_ultimate_dynamic_v1_3, "ULTIMATE_V1.3"),
    (rl.radius_ultimate_dynamic_v1_4, "ULTIMATE_V1.4")
]

SAVE_DIR = f"comparison_{VIDEO_ID}"
os.makedirs(SAVE_DIR, exist_ok=True)

def run_auto_comparison():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    p_history, s_history, p_info = {}, {}, {}
    saved_files = []
    frame_idx = 0
    captured = False

    print(f"--- STARTING COMPARISON WITH BOUNDING BOXES: {VIDEO_ID} ---")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or captured: break
        
        t_curr = frame_idx / fps
        frame_vis_base = cv2.resize(frame, (1280, 720))

        # 1. TRACKING
        if frame_idx % 2 == 0:
            res = model.track(frame_vis_base, persist=True, classes=[0, 1], verbose=False)
            curr_p, curr_s = {}, {}
            if res and res[0].boxes.id is not None:
                for box in res[0].boxes:
                    obj_id = int(box.id[0])
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    cx, cy = (coords[0] + coords[2]) // 2, (coords[1] + coords[3]) // 2
                    
                    if int(box.cls[0]) == 1: # PERSON
                        curr_p[obj_id] = {'cnt': (cx, cy), 'box': coords}
                        h_px = abs(coords[3] - coords[1])
                        p_info.setdefault(obj_id, {"max_h": 0})["max_h"] = max(p_info.get(obj_id, {}).get("max_h", 0), h_px)
                        p_history.setdefault(obj_id, deque(maxlen=30)).append((cx, cy))
                    else: # LUGGAGE
                        curr_s[obj_id] = {'cnt': (cx, cy), 'box': coords}
                        s_history.setdefault(obj_id, deque(maxlen=30)).append((cx, cy))
            last_res_p, last_res_s = curr_p, curr_s

        # 2. TRIGGER CAPTURE
        if abs(t_curr - TARGET_SAVE_TIME) < (1.0 / fps):
            for method_func, method_name in METHODS_TO_RUN:
                method_frame = frame_vis_base.copy()
                
                # A. DRAW BOUNDING BOXES FIRST (So they are in the background)
                for pid, p_data in last_res_p.items():
                    bx = p_data['box']
                    cv2.rectangle(method_frame, (bx[0], bx[1]), (bx[2], bx[3]), (255, 100, 0), 2) # Blueish-Cyan
                    cv2.putText(method_frame, f"P-{pid}", (bx[0], bx[1]-5), 0, 0.5, (255, 100, 0), 1)

                for sid, s_data in last_res_s.items():
                    bx = s_data['box']
                    cv2.rectangle(method_frame, (bx[0], bx[1]), (bx[2], bx[3]), (0, 165, 255), 2) # Orange
                    cv2.putText(method_frame, f"L-{sid}", (bx[0], bx[1]-5), 0, 0.5, (0, 165, 255), 1)

                # B. DRAW LOGIC (Radius, Lines, Vectors)
                for sid, s_data in last_res_s.items():
                    scx, scy = s_data['cnt']
                    for pid, p_data in last_res_p.items():
                        pcx, pcy = p_data['cnt']
                        others = sum(1 for opid, op in last_res_p.items() if opid != pid and np.linalg.norm(np.array(p_data['cnt'])-np.array(op['cnt'])) < 150)
                        dist = np.linalg.norm(np.array(s_data['cnt']) - np.array(p_data['cnt']))
                        
                        R = method_func(p_info=p_info, pid=pid, p_history=p_history, s_history=s_history, sid=sid, neighbors=others)
                        
                        # Color logic: Green if linked, Gray if searching
                        color = (0, 255, 0) if dist < R else (200, 200, 200)
                        cv2.circle(method_frame, (pcx, pcy), int(R), color, 2)
                        cv2.line(method_frame, (pcx, pcy), (scx, scy), color, 1)

                        # Selective Vector drawing
                        if "DYNAMIC" in method_name or "ULTIMATE" in method_name:
                            vel = rl.get_velocity_vec(p_history.get(pid, []))
                            if np.linalg.norm(vel) > 1.5:
                                end_pt = (int(pcx + vel[0] * 3), int(pcy + vel[1] * 3))
                                cv2.arrowedLine(method_frame, (pcx, pcy), end_pt, (255, 255, 0), 2, tipLength=0.3)

                # C. ANNOTATE METHOD NAME
                cv2.putText(method_frame, f"Method: {method_name}", (20, 50), 0, 1.0, (0, 255, 255), 2)
                
                save_path = f"{SAVE_DIR}/{method_name}_bbox_capture.png"
                cv2.imwrite(save_path, method_frame)
                saved_files.append(save_path)
            
            captured = True

        frame_idx += 1

    cap.release()

    # --- FINAL OUTPUT PRINT ---
    print("\n" + "="*60)
    print("SCIENTIFIC OUTPUT SUMMARY - READY FOR PUBLICATION")
    print("="*60)
    for i, file in enumerate(saved_files, 1):
        print(f"{i}. [SAVED] {file}")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_auto_comparison()