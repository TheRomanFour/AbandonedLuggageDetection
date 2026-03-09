import numpy as np

def get_velocity_vec(history):
    if len(history) < 10: return np.array([0.0, 0.0])
    # Calculates velocity over the last 10 frames
    return np.array(history[-1]) - np.array(history[-10])

def cosine_similarity(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-5 or n2 < 1e-5: return 0.0
    return np.dot(v1, v2) / (n1 * n2)

# --- BENCHMARK METHODS ---


def radius_strictly_fixed_100(p_info, pid, **kwargs):
    """
    4.3.1 Baseline: Fixed Pixel Radius (FPR).
    The standard approach in legacy systems. Uses a constant 150px threshold.
    Highly sensitive to perspective effects.
    """
    return 100

def radius_strictly_fixed_150(p_info, pid, **kwargs):
    """
    4.3.1 Baseline: Fixed Pixel Radius (FPR).
    The standard approach in legacy systems. Uses a constant 150px threshold.
    Highly sensitive to perspective effects.
    """
    return 150

def radius_strictly_fixed_200(p_info, pid, **kwargs):
    """
    4.3.1 Baseline: Fixed Pixel Radius (FPR).
    The standard approach in legacy systems. Uses a constant 150px threshold.
    Highly sensitive to perspective effects.
    """
    return 200

def radius_strictly_fixed_250(p_info, pid, **kwargs):
    """
    4.3.1 Baseline: Fixed Pixel Radius (FPR).
    The standard approach in legacy systems. Uses a constant 150px threshold.
    Highly sensitive to perspective effects.
    """
    return 250

def radius_fixed_equivalent(p_info, pid, **kwargs):
    """Baseline: 80% of person height (Static approximation)."""
    h = p_info.get(pid, {}).get("max_h", 100)
    return int(h * 0.8)

def radius_height_adaptive(p_info, pid, **kwargs):
    """Adaptive: 120% of person height."""
    h = p_info.get(pid, {}).get("max_h", 100)
    return int(h * 1.2)

def radius_velocity_dynamic(p_info, pid, p_history, **kwargs):
    """Dynamic: Base 80% h + velocity bonus based on movement speed."""
    h = p_info.get(pid, {}).get("max_h", 100)
    v_mag = np.linalg.norm(get_velocity_vec(p_history.get(pid, [])))
    v_bonus = (v_mag / h) * h * 0.6
    return int(h * 0.8 + v_bonus)

def radius_ultimate_dynamic_v1_3(p_info, pid, p_history, s_history, sid, neighbors, **kwargs):
    """
    v1.3: Fully relative method. 
    Combines height, velocity, alignment (cosine similarity), and density.
    """
    h = p_info.get(pid, {}).get("max_h", 100)
    
    # 1. Base (75% h)
    base_r = h * 0.75
    
    # 2. Velocity Bonus
    v_mag = np.linalg.norm(get_velocity_vec(p_history.get(pid, [])))
    v_bonus = min((v_mag / h) * h * 0.5, h * 0.4)
    
    # 3. Alignment Bonus (Are they moving together?)
    a_bonus = 1.0
    if sid in s_history and pid in p_history:
        if cosine_similarity(get_velocity_vec(p_history[pid]), get_velocity_vec(s_history[sid])) > 0.8:
            a_bonus = 1.3
            
    # 4. Density Factor (Crowd control)
    density_factor = 1 / (1 + 0.2 * neighbors)
    
    final_r = (base_r + v_bonus) * a_bonus * density_factor
    return int(np.clip(final_r, h * 0.4, h * 2.5))

def radius_ultimate_dynamic_v1_4(p_info, pid, p_history, s_history, sid, neighbors, is_sticky=False, stats=None, **kwargs):
    """
    v1.4: Statistical & Temporal Layer.
    Adds 'Sticky' logic to prevent radius flickering and statistical mean/std adaptation.
    """
    h = p_info.get(pid, {}).get("max_h", 100)
    
    # --- 1. STATISTICAL LAYER ---
    if stats and stats.get('n', 0) > 15:
        # Learned habit: Mean distance + 2 standard deviations
        base_r = stats['mean'] + 2 * stats['std']
    else:
        # Fallback to Dynamic v1.3 logic
        v_mag = np.linalg.norm(get_velocity_vec(p_history.get(pid, [])))
        v_bonus = min((v_mag / h) * h * 0.5, h * 0.4)
        base_r = (h * 0.75) + v_bonus

    # --- 2. ENVIRONMENTAL LAYER ---
    a_bonus = 1.3 if (sid in s_history and pid in p_history and 
                      cosine_similarity(get_velocity_vec(p_history[pid]), 
                                        get_velocity_vec(s_history[sid])) > 0.8) else 1.0
    
    density_factor = 1 / (1 + 0.2 * neighbors)
    
    # --- 3. TEMPORAL LAYER ---
    # Sticky bonus prevents 'dropping' an owner due to a single frame of noise
    sticky_bonus = 1.3 if is_sticky else 1.0
    
    final_r = base_r * a_bonus * density_factor * sticky_bonus
    
    # Safety clip based on physical human proportions
    return int(np.clip(final_r, h * 0.4, h * 2.5))