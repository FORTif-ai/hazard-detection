import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose


def get_ankle_coord(landmarks, side, axis="x"):
    idx = (
        mp_pose.PoseLandmark.LEFT_ANKLE
        if side == "left"
        else mp_pose.PoseLandmark.RIGHT_ANKLE
    )
    if landmarks[idx.value].visibility > 0.5:
        return getattr(landmarks[idx.value], axis)
    return None


def calculate_step_variation(step_lengths):
    if len(step_lengths) < 2:
        return None

    mean = np.mean(step_lengths)
    std_dev = np.std(step_lengths)
    coeff_var = std_dev / mean

    return {
        "mean": mean,
        "std_dev": std_dev,
        "coefficient_of_variation": coeff_var,
        "even_steps": coeff_var < 0.15,
    }
