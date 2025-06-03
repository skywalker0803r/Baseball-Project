
# filter_clean_throws.py
# ✅ 專為結構：每資料夾對應一種球種（含 CSV + 多部影片）
# 篩選 description 為 "called_strike" 或 "swinging_strike" 的影片與資料

import os
import pandas as pd
import shutil

GOOD_DESCRIPTIONS = ["called_strike", "swinging_strike"]

def filter_good_throws(input_root, output_root):
    os.makedirs(output_root, exist_ok=True)

    for folder_name in os.listdir(input_root):
        folder_path = os.path.join(input_root, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # 尋找 CSV
        csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        if not csv_files:
            print(f"[!] 找不到 CSV：{folder_path}")
            continue

        csv_path = os.path.join(folder_path, csv_files[0])
        df = pd.read_csv(csv_path)

        # 篩選好球
        good_rows = df[df["description"].isin(GOOD_DESCRIPTIONS)]
        good_indices = good_rows.index

        # 建立輸出資料夾
        out_folder = os.path.join(output_root, folder_name)
        os.makedirs(out_folder, exist_ok=True)

        # 複製符合的影片
        for idx in good_indices:
            video_name = f"pitch_{idx+1:04d}.mp4"
            src_path = os.path.join(folder_path, video_name)
            dst_path = os.path.join(out_folder, video_name)
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)

        # 儲存對應的新 CSV
        new_csv_path = os.path.join(out_folder, csv_files[0])
        good_rows.to_csv(new_csv_path, index=False)

        print(f"[✓] {folder_name}：保留 {len(good_indices)} 筆好球")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("用法：python filter_clean_throws.py <input_data_dir> <output_clean_dir>")
        exit(1)

    input_root = sys.argv[1]
    output_root = sys.argv[2]
    filter_good_throws(input_root, output_root)
