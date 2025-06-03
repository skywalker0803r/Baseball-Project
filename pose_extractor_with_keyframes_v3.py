import sys
import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.framework.formats import landmark_pb2

mp_pose = mp.solutions.pose


def get_landmark_vector(lm, idx):
    return np.array([lm[idx].x, lm[idx].y, lm[idx].z])  # ✅ 改為 NumPy 陣列


def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def calc_stride_angle(lm):
    return calculate_angle(
        get_landmark_vector(lm, 24),
        get_landmark_vector(lm, 26),
        get_landmark_vector(lm, 23),
    )


def calc_throwing_angle(lm):
    return calculate_angle(
        get_landmark_vector(lm, 12),
        get_landmark_vector(lm, 14),
        get_landmark_vector(lm, 16),
    )


def calc_arm_symmetry(lm):
    return 1 - abs(lm[15].y - lm[16].y)


def calc_hip_rotation(lm):
    return abs(lm[23].z - lm[24].z)


def calc_elbow_height(lm):
    return lm[14].y


def calc_ankle_height(lm):
    return lm[28].y


def calc_shoulder_rotation(lm):
    return abs(lm[11].z - lm[12].z)


def calc_torso_tilt_angle(lm):
    return calculate_angle(
        get_landmark_vector(lm, 11),
        get_landmark_vector(lm, 23),
        get_landmark_vector(lm, 24),
    )


def calc_release_distance(lm):
    return np.linalg.norm(
        get_landmark_vector(lm, 16) - get_landmark_vector(lm, 12)
    )  # ✅ 修正 list 相減


def calc_shoulder_to_hip(lm):
    return abs(lm[12].x - lm[24].x)


def draw_pose(image, landmarks):
    annotated = image.copy()
    mp.solutions.drawing_utils.draw_landmarks(
        annotated, landmarks, mp_pose.POSE_CONNECTIONS
    )
    return annotated


def find_release_frame(landmarks_seq, target_z):
    z_list = [lm[16].z for lm in landmarks_seq]
    idx = int(np.argmin([abs(z - target_z) for z in z_list]))
    return idx


def extract_four_keyframes(video_path, output_dir, statcast_df, vis_dir="output_vis"):
    filename = os.path.splitext(os.path.basename(video_path))[0]
    row = statcast_df[statcast_df["Filename"].str.contains(filename)].iloc[0]
    pitch_type = row["pitch_type"]
    pitcher = row["player_name"] if "player_name" in row else row["pitcher"]
    is_strike = 1 if "strike" in row["description"].lower() else 0
    release_z = row["release_pos_z"] if "release_pos_z" in row else None

    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    frames, frame_landmarks = [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if results.pose_landmarks:
            frame_landmarks.append(results.pose_landmarks.landmark)
            frames.append(frame)

    cap.release()
    pose.close()

    total_frames = len(frame_landmarks)
    if total_frames < 60:
        print(f"❌ 幀數過少：{video_path}")
        return

    margin = min(50, total_frames // 3)
    release_idx = None
    if release_z:
        release_idx = find_release_frame(
            frame_landmarks, release_z / 10.0
        )  # 英呎 → 約 [0,1]
    if release_idx is None:
        velocities = [
            np.linalg.norm(
                get_landmark_vector(frame_landmarks[i], 16)
                - get_landmark_vector(frame_landmarks[i - 1], 16)
            )
            for i in range(1, total_frames)
        ]
        release_idx = int(np.argmax(velocities)) + 1

    indices = [release_idx - 20, release_idx - 5, release_idx, release_idx + 10]
    names = ["foot", "arm", "release", "hip"]

    features = []
    csv_rows = []

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    for name, idx in zip(names, indices):
        start = max(idx - margin, 0)
        end = min(idx + margin, total_frames)
        seq = frame_landmarks[start:end]
        pad_len = 100 - len(seq)
        if pad_len > 0:
            seq += [seq[-1]] * pad_len
        seq = seq[:100]

        f1 = [calc_stride_angle(lm) for lm in seq]
        f2 = [calc_throwing_angle(lm) for lm in seq]
        f3 = [calc_arm_symmetry(lm) for lm in seq]
        f4 = [calc_hip_rotation(lm) for lm in seq]
        f5 = [calc_elbow_height(lm) for lm in seq]
        f6 = [calc_ankle_height(lm) for lm in seq]
        f7 = [calc_shoulder_rotation(lm) for lm in seq]
        f8 = [calc_torso_tilt_angle(lm) for lm in seq]
        f9 = [calc_release_distance(lm) for lm in seq]
        f10 = [calc_shoulder_to_hip(lm) for lm in seq]

        features.append([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10])

        landmark_proto = landmark_pb2.NormalizedLandmarkList()
        landmark_proto.landmark.extend(frame_landmarks[idx])
        vis_img = draw_pose(frames[idx], landmark_proto)
        cv2.imwrite(os.path.join(vis_dir, f"{filename}_{name}.jpg"), vis_img)

        csv_rows.append(
            [filename, name]
            + [np.mean(f) for f in [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]]
            + [pitch_type, pitcher, is_strike]
        )

    features = np.transpose(features, (1, 2, 0))  # (4, 100, 10)
    features = features.reshape(4, -1)  # (4, 1000)
    np.save(os.path.join(output_dir, filename + ".npy"), features)

    df = pd.DataFrame(
        csv_rows,
        columns=[
            "影片",
            "動作",
            "stride_angle",
            "throwing_angle",
            "arm_symmetry",
            "hip_rotation",
            "elbow_height",
            "ankle_height",
            "shoulder_rotation",
            "torso_tilt_angle",
            "release_distance",
            "shoulder_to_hip",
            "pitch_type",
            "pitcher",
            "is_strike",
        ],
    )
    df.to_csv(os.path.join(output_dir, filename + ".csv"), index=False)
    print(f"✅ 儲存特徵：{filename}, shape={features.shape}")


if __name__ == "__main__":
    video_path = sys.argv[1]
    output_dir = sys.argv[2]
    statcast_path = sys.argv[3]
    statcast_df = pd.read_csv(statcast_path)
    extract_four_keyframes(video_path, output_dir, statcast_df)
