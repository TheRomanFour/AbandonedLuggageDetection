"""
Grid search za CA-DPF parametre: Velocity Multiplier (V_mul), Sticky Factor (S), Density Factor (D).

ARHITEKTURA (bitno za brzinu):
    FAZA 1 (skupo, jednom po videu): YOLO tracking prolazi kroz video JEDNOM i sprema
        pozicije/visine/brzine po frameu u memoriju (frame_records).
    FAZA 2 (jeftino, 60x po videu): za svaku kombinaciju (V_mul, S, D) samo "odigramo"
        keairane frame_records kroz radius formulu (cisti numpy, bez ponovnog pokretanja
        modela) i izracunamo alarm/no-alarm po videu.

Time izbjegavamo 60x ponavljanje YOLO inferencije - ona se radi samo jednom.

PRETPOSTAVKE (provjeri/prilagodi ako ne odgovaraju tvojoj definiciji):
    - "is_sticky" za par (sid, pid) = True ako je pid bio odabrani "vlasnik" (najbliza
      osoba koja zadovoljava R i vremenski prag) za taj sid u PRETHODNOM evaluiranom frameu.
      Tvoj originalni run_benchmark() nikad nije postavljao is_sticky, pa je S faktor
      efektivno bio neaktivan - ovdje ga aktiviram preko ove definicije.
    - STATISTICAL LAYER (mean + 2*std grana iz v1.4) je izostavljena radi
      jednostavnosti/reproducibilnosti grid searcha - uvijek se koristi "fallback" grana
      (baza 0.75*h + brzinski bonus). Ako ti treba i ta grana, javi pa dodajem.
    - Formula je identicna radius_ultimate_dynamic_v1_4 iz radius_logic.py, samo su
      0.5 (velocity), 1.3 (sticky) i 0.2 (density) izvuceni kao parametri.

POPRAVAK (bitno!): originalni run_benchmark() je imao "if alarm_time: break" - obrada
    videa je stala na PRVI alarm u cijelom videu, cak i ako se dogodio prije pravog GT
    eventa. Dijagnostikom (diagnose_early_break.py) je potvrdjeno da to gubi ~56% tocnih
    detekcija (originalni kod: 20/80 pogodaka, verzija bez prekidanja: 45/80). Ovdje se
    obrada VISE NE PREKIDA - biljezi se svaki alarm (rising edge, gasi se kad se vlasnik
    vrati), i video se racuna kao TP ako BILO KOJI od tih alarma pogodi GT prozor.

Pokretanje:
    python gridsearch_ca_dpf_params.py
"""
import itertools
import json
import os
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# KONFIGURACIJA - prilagodi po potrebi
# ---------------------------------------------------------------------------

BASE_DIR = "/mnt/SamsungSSD/IvanVrsalovic/Prtljaga/Datasets/Uniri dataset/Dataset/"
ANNOTATION_DIR = os.path.join(BASE_DIR, "annotations_update")

# Model koji se koristi za grid search parametara (fiksan - varira se samo logika, ne model)
MODEL_PATH = "/mnt/SamsungSSD/IvanVrsalovic/Prtljaga/Datasets/Uniri dataset/yolov11/yolo11s/weights/best.pt"

# Grid vrijednosti - sad testiramo SVA TRI parametra (V_mul vraen u grid, iako je
# ranije pokazao zanemariv efekt, radi potpunog CSV-a sa svim kombinacijama).
# V_mul: 5 vrijednosti (0.2-0.6, korak 0.1)
# S: 11 vrijednosti (0.40-0.90, korak 0.05) - pokriva okolinu pronadjenog vrha (S=0.6)
# D: 11 vrijednosti (0.00-0.25, korak 0.025) - pokriva okolinu pronadjenog platoa (D=0.10-0.20)
# Ukupno 5 x 11 x 11 = 605 kombinacija - i dalje brzo jer je Faza 2 (replay) jeftina,
# tracking (Faza 1) se i dalje radi samo jednom po videu.
V_MUL_GRID = [round(0.2 + 0.1 * i, 2) for i in range(5)]        # 0.2, 0.3, 0.4, 0.5, 0.6
STICKY_GRID = [round(0.40 + 0.05 * i, 2) for i in range(11)]     # 0.40, 0.45, ..., 0.90
DENSITY_GRID = [round(0.000 + 0.025 * i, 3) for i in range(11)]  # 0.000, 0.025, ..., 0.250

NEIGHBOR_RADIUS_PX = 150  # isto kao u originalnom run_benchmark()

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(SCRIPT_DIR, f"gridsearch_ca_dpf_results_{timestamp}.csv")


# ---------------------------------------------------------------------------
# Pomocne funkcije (identicne logici u radius_logic.py)
# ---------------------------------------------------------------------------

def get_velocity_vec(history):
    if len(history) < 10:
        return np.array([0.0, 0.0])
    return np.array(history[-1]) - np.array(history[-10])


def cosine_similarity(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-5 or n2 < 1e-5:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def radius_ca_dpf(h, v_mag, align_cosine, neighbors, is_sticky, v_mul, sticky_factor, density_coef):
    """Ista formula kao radius_ultimate_dynamic_v1_4, s V_mul/S/D kao parametrima."""
    if h <= 0:
        h = 100
    v_bonus = min((v_mag / h) * h * v_mul, h * 0.4)
    base_r = h * 0.75 + v_bonus

    a_bonus = 1.3 if align_cosine > 0.8 else 1.0
    density_factor = 1.0 / (1.0 + density_coef * neighbors)
    sticky_bonus = sticky_factor if is_sticky else 1.0

    final_r = base_r * a_bonus * density_factor * sticky_bonus
    return float(np.clip(final_r, h * 0.4, h * 2.5))


# ---------------------------------------------------------------------------
# FAZA 1: tracking pass - jednom po videu, keAira sve sto ne ovisi o parametrima
# ---------------------------------------------------------------------------

def run_tracking_pass(model, video_path, fps_fallback=30.0):
    """
    Vraca listu frame_records:
        {
          't': vrijeme u sekundama,
          'people': {pid: {'pos': (cx,cy), 'h': running_max_h, 'vel': np.array([vx,vy])}},
          'statics': {sid: {'pos': (cx,cy), 'vel': np.array([vx,vy])}},
        }
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or fps_fallback

    p_history, s_history, p_info = {}, {}, {}
    frame_records = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (800, 600))
        t_curr = frame_idx / fps

        res = model.track(frame, persist=True, classes=[0, 1], verbose=False, vid_stride=2)

        curr_people, curr_statics = {}, {}
        if res and res[0].boxes.id is not None:
            for box in res[0].boxes:
                obj_id, cls = int(box.id[0]), int(box.cls[0])
                coords = box.xyxy[0].cpu().numpy()
                cx = int((coords[0] + coords[2]) / 2)
                cy = int(coords[3]) if cls == 1 else int((coords[1] + coords[3]) / 2)
                h_px = abs(coords[3] - coords[1])

                if cls == 1:  # osoba
                    p_info.setdefault(obj_id, {"max_h": 0})
                    p_info[obj_id]["max_h"] = max(p_info[obj_id]["max_h"], h_px)
                    p_history.setdefault(obj_id, deque(maxlen=30)).append((cx, cy))
                    curr_people[obj_id] = {
                        "pos": (cx, cy),
                        "h": p_info[obj_id]["max_h"],
                        "vel": get_velocity_vec(list(p_history[obj_id])),
                    }
                else:  # staticni objekt (torba)
                    s_history.setdefault(obj_id, deque(maxlen=30)).append((cx, cy))
                    curr_statics[obj_id] = {
                        "pos": (cx, cy),
                        "vel": get_velocity_vec(list(s_history[obj_id])),
                    }

        frame_records.append({"t": t_curr, "people": curr_people, "statics": curr_statics})
        frame_idx += 2  # isti korak kao original (vid_stride=2, frame_idx += 2)

    cap.release()
    return frame_records


# ---------------------------------------------------------------------------
# FAZA 2: replay logike za jednu kombinaciju parametara (jeftino)
# ---------------------------------------------------------------------------

def replay_params(frame_records, t_own_thresh, t_abandon_thresh, v_mul, sticky_factor, density_coef):
    """Vraca (alarms, diag) za dani video i kombinaciju parametara.

    alarms = lista SVIH "rising edge" alarm-trenutaka (float sekunde), po sid-u -
    tj. svaki put kad abandon_timer[sid] prvi put prijedje prag. Alarm se "gasi"
    kad se guard vrati (guards>0), pa isti sid moze ponovno alarmirati kasnije.

    VAZNO: ovo VISE NE PREKIDA obradu na prvi alarm (originalni run_benchmark() je
    imao "if alarm_time: break", sto je dokazano gutalo ~56% tocnih detekcija jer bi
    rani/preuranjeni alarm zaustavio obradu prije pravog GT eventa - vidi dijagnostiku
    u chatu). Sad se video obradjuje do kraja i SVAKI alarm se biljezi.

    diag broji koliko puta je is_sticky bio True i koliko puta je bio PRESUDAN
    (bez sticky bonusa dist>=R, a sa sticky bonusom dist<R)."""
    proximity_timers, abandon_timers = {}, {}
    prev_owner = {}
    alarm_active = {}  # sid -> je li trenutno u stanju alarma (da znamo rising edge)
    alarms = []
    diag = {"pid_evals": 0, "sticky_active": 0, "sticky_decisive": 0}

    for rec in frame_records:
        t_curr = rec["t"]
        people = rec["people"]
        statics = rec["statics"]
        curr_owner_this_frame = {}

        for sid, sdata in statics.items():
            guards = 0
            best_pid, best_dist = None, None

            for pid, pdata in people.items():
                h = pdata["h"]
                v_mag = float(np.linalg.norm(pdata["vel"]))
                neighbors = sum(
                    1
                    for opid, opdata in people.items()
                    if opid != pid
                    and np.linalg.norm(np.array(pdata["pos"]) - np.array(opdata["pos"])) < NEIGHBOR_RADIUS_PX
                )
                is_sticky = prev_owner.get(sid) == pid
                align = cosine_similarity(pdata["vel"], sdata["vel"])

                diag["pid_evals"] += 1
                R = radius_ca_dpf(h, v_mag, align, neighbors, is_sticky, v_mul, sticky_factor, density_coef)
                dist = float(np.linalg.norm(np.array(sdata["pos"]) - np.array(pdata["pos"])))

                if is_sticky:
                    diag["sticky_active"] += 1
                    R_no_sticky = radius_ca_dpf(h, v_mag, align, neighbors, False, v_mul, sticky_factor, density_coef)
                    if (dist < R) and not (dist < R_no_sticky):
                        diag["sticky_decisive"] += 1

                if dist < R:
                    proximity_timers.setdefault((sid, pid), t_curr)
                    if (t_curr - proximity_timers[(sid, pid)]) >= t_own_thresh:
                        guards += 1
                        if best_dist is None or dist < best_dist:
                            best_dist, best_pid = dist, pid
                else:
                    proximity_timers.pop((sid, pid), None)

            curr_owner_this_frame[sid] = best_pid

            if guards == 0:
                abandon_timers.setdefault(sid, t_curr)
                if (t_curr - abandon_timers[sid]) > t_abandon_thresh:
                    if not alarm_active.get(sid, False):
                        alarms.append(t_curr)
                        alarm_active[sid] = True
            else:
                abandon_timers.pop(sid, None)
                alarm_active[sid] = False  # guard se vratio - alarm se gasi, moze ponovno okinuti

        prev_owner = curr_owner_this_frame

    return alarms, diag


# ---------------------------------------------------------------------------
# Evaluacija (Precision/Recall/F1) preko svih videa za jednu kombinaciju
# ---------------------------------------------------------------------------

def evaluate_combo(video_cache, v_mul, sticky_factor, density_coef):
    tp = fp = fn = tn = 0
    diag_total = {"pid_evals": 0, "sticky_active": 0, "sticky_decisive": 0}

    for entry in video_cache:
        alarms, diag = replay_params(
            entry["frame_records"],
            entry["t_own_thresh"],
            entry["t_abandon_thresh"],
            v_mul,
            sticky_factor,
            density_coef,
        )
        for k in diag_total:
            diag_total[k] += diag[k]

        is_gt_abandon = entry["is_gt_abandon"]

        if is_gt_abandon:
            # TP ako BILO KOJI alarm (ne samo prvi) pogodi GT prozor bilo kojeg eventa
            is_correct = any(
                e["start_time"] <= a <= (e["start_time"] + entry["t_abandon_thresh"] + e.get("tolerance", 15.0))
                for e in entry["gt_events"]
                for a in alarms
            )
            if is_correct:
                tp += 1
            else:
                fn += 1
        else:
            if len(alarms) == 0:
                tn += 1
            else:
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Precision": precision, "Recall": recall, "F1": f1,
        "pid_evals": diag_total["pid_evals"],
        "sticky_active_frames": diag_total["sticky_active"],
        "sticky_decisive_frames": diag_total["sticky_decisive"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Ucitavam model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    video_files = sorted([f for f in os.listdir(BASE_DIR) if f.endswith(".mp4")])
    print(f"Pronadjeno {len(video_files)} videa.")

    print("\n=== FAZA 1: tracking pass (jednom po videu) ===")
    video_cache = []
    for v_name in video_files:
        v_path = os.path.join(BASE_DIR, v_name)
        j_path = os.path.join(ANNOTATION_DIR, v_name.replace(".mp4", ".json"))
        if not os.path.exists(j_path):
            print(f"  [{v_name}] preskacem - nema anotacije")
            continue

        with open(j_path, "r") as f:
            gt_data = json.load(f)

        timers = gt_data.get("timers", {})
        t_abandon_thresh = timers.get("time_to_abandoned", 5.0)
        t_own_thresh = timers.get("time_to_own", 2.0)
        fps_json = gt_data.get("fps", 30.0)
        gt_events = [e for e in gt_data.get("events", []) if e.get("type") == "abandonment"]

        print(f"  [{v_name}] tracking...", end="\r")
        frame_records = run_tracking_pass(model, v_path, fps_fallback=fps_json)
        print(f"  [{v_name}] gotovo - {len(frame_records)} frameova")

        video_cache.append(
            {
                "video": v_name,
                "frame_records": frame_records,
                "t_own_thresh": t_own_thresh,
                "t_abandon_thresh": t_abandon_thresh,
                "gt_events": gt_events,
                "is_gt_abandon": len(gt_events) > 0,
            }
        )

    print("\n=== FAZA 2: grid search preko parametara (replay, bez ponovnog trackinga) ===")
    combos = list(itertools.product(V_MUL_GRID, STICKY_GRID, DENSITY_GRID))
    print(f"Ukupno kombinacija: {len(combos)}")

    results = []
    for i, (v_mul, sticky_factor, density_coef) in enumerate(combos, start=1):
        metrics = evaluate_combo(video_cache, v_mul, sticky_factor, density_coef)
        row = {"V_mul": v_mul, "S": sticky_factor, "D": density_coef, **metrics}
        results.append(row)
        print(
            f"  [{i}/{len(combos)}] V_mul={v_mul} S={sticky_factor} D={density_coef} "
            f"-> P={metrics['Precision']:.3f} R={metrics['Recall']:.3f} F1={metrics['F1']:.3f}"
        )

    df = pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 70)
    print("TOP 10 KOMBINACIJA PO F1")
    print("=" * 70)
    print(df.head(10).to_string(index=False))

    target = df[(df["V_mul"] == 0.4) & (df["S"] == 0.9) & (df["D"] == 0.3)]
    if not target.empty:
        rank = df.index[(df["V_mul"] == 0.4) & (df["S"] == 0.9) & (df["D"] == 0.3)][0] + 1
        print("\n" + "-" * 70)
        print(f"Tvoja odabrana kombinacija (V_mul=0.4, S=0.9, D=0.3) je na rangu #{rank} od {len(df)}:")
        print(target.to_string(index=False))

    print("\n" + "-" * 70)
    print("DIJAGNOSTIKA STICKY LOGIKE (agregirano preko svih kombinacija/videa)")
    print("-" * 70)
    total_active = df["sticky_active_frames"].sum()
    total_decisive = df["sticky_decisive_frames"].sum()
    total_evals = df["pid_evals"].sum()
    pct_active = (total_active / total_evals * 100) if total_evals else 0.0
    pct_decisive_of_active = (total_decisive / total_active * 100) if total_active else 0.0
    print(f"Ukupno (pid, frame) evaluacija (svi combos): {total_evals}")
    print(f"Sticky aktivan (is_sticky=True): {total_active}  ({pct_active:.2f}% svih evaluacija)")
    print(f"Sticky bio PRESUDAN (promijenio dist<R odluku): {total_decisive}  ({pct_decisive_of_active:.2f}% od aktivnih)")
    if total_decisive == 0:
        print(">> Sticky bonus NIJE NIJEDNOM promijenio ishod - zato S nema efekta na F1.")
        print(">> Vjerojatno treba redefinirati sto 'sticky' znaci (vidi napomenu u chatu).")

    print(f"\nSvih {len(df)} kombinacija spremljeno u: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
