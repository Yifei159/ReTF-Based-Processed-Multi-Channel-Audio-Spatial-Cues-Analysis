
import numpy as np

def get_mic_geometry_matrix() -> np.ndarray:
    R = 0.15
    theta_deg = -11.0
    angles_deg = {"Mic 2": 90.0, "Mic 3": 210.0, "Mic 4": 330.0}
    pts0 = {}
    for name, a in angles_deg.items():
        a_rad = np.deg2rad(a)
        x = R * np.cos(a_rad)
        y = 0.0
        z = R * np.sin(a_rad)
        pts0[name] = np.array([x, y, z])
    theta = np.deg2rad(theta_deg)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta), -np.sin(theta)],
                   [0, np.sin(theta),  np.cos(theta)]])
    pts_rot = {name: Rx @ v for name, v in pts0.items()}
    P = np.vstack([pts_rot["Mic 2"], pts_rot["Mic 3"], pts_rot["Mic 4"]])
    return P
