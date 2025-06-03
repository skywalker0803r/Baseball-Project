import os
import pandas as pd
from pose_extractor_with_keyframes_v3 import extract_four_keyframes

# 設定基底資料夾與輸出根路徑
base_dir = "data"
output_root = "features"

# 建立 features 根資料夾
os.makedirs(output_root, exist_ok=True)

# 遍歷所有子資料夾（每一位投手 × 球種）
for subdir in os.listdir(base_dir):
    subdir_path = os.path.join(base_dir, subdir)
    if not os.path.isdir(subdir_path):
        continue

    print(f"\n📂 處理資料夾：{subdir_path}")

    # 找出 CSV 檔案
    csv_files = [f for f in os.listdir(subdir_path) if f.endswith(".csv")]
    if len(csv_files) == 0:
        print("⚠️ 未找到 CSV，跳過此資料夾")
        continue
    csv_path = os.path.join(subdir_path, csv_files[0])
    statcast_df = pd.read_csv(csv_path)

    # 找出影片
    video_files = sorted([f for f in os.listdir(subdir_path) if f.endswith(".mp4")])
    if len(video_files) == 0:
        print("⚠️ 沒有影片檔案，跳過")
        continue

    # 計算分割點
    split_idx = int(0.8 * len(video_files))

    # 建立 train/test 子資料夾
    sub_output_dir = os.path.join(output_root, subdir)
    train_dir = os.path.join(sub_output_dir, "train")
    test_dir = os.path.join(sub_output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 依序處理每支影片
    for i, video in enumerate(video_files):
        video_path = os.path.join(subdir_path, video)
        filename = os.path.splitext(video)[0]

        # 分配至 train 或 test 子資料夾
        tag = "train" if i < split_idx else "test"
        output_subdir = train_dir if tag == "train" else test_dir

        # 防呆：若已存在 .npy 則略過
        if os.path.exists(os.path.join(output_subdir, filename + ".npy")):
            print(f"⚠️ 已存在，略過：{filename}")
            continue

        print(f"🎥 [{i+1}/{len(video_files)}] 處理影片：{video} → {tag}")

        try:
            extract_four_keyframes(
                video_path=video_path,
                output_dir=output_subdir,
                statcast_df=statcast_df,
                vis_dir="output_vis",
            )
        except Exception as e:
            print(f"❌  失敗：{video}，錯誤：{str(e)}")
