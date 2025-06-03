# evaluate_model_v3.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from prepare_and_train_v3 import Classifier

# ✅ 讀取模型
model = Classifier()
model.load_state_dict(torch.load("model_strike_predictor.pth", map_location="cpu"))
model.eval()

# ✅ 設定資料路徑
feature_root = "features"
output_rows = []
correct = 0
total = 0

# ✅ 遍歷所有子資料夾
for subdir in os.listdir(feature_root):
    folder = os.path.join(feature_root, subdir)
    if not os.path.isdir(folder):
        continue

    for fname in os.listdir(folder):
        if not fname.endswith(".npy"):
            continue

        npy_path = os.path.join(folder, fname)
        csv_path = npy_path.replace(".npy", ".csv")
        if not os.path.exists(csv_path):
            continue

        # 讀取特徵
        data = np.load(npy_path)
        x = np.concatenate([data[i] for i in [0, 2, 3]], axis=0)

        # 預測
        with torch.no_grad():
            input_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            output = model(input_tensor)
            predicted = torch.argmax(output, dim=1).item()

        # 讀取標籤
        try:
            df = pd.read_csv(csv_path)
            true_label = df[df["動作"] == "release"]["is_strike"].values[0]
            true_label = int(true_label)
            match = predicted == true_label
            correct += int(match)
            total += 1
        except:
            true_label = "N/A"
            match = "N/A"

        output_rows.append(
            {"影片": fname, "預測": predicted, "實際": true_label, "是否正確": match}
        )

# ✅ 輸出成報表
result_df = pd.DataFrame(output_rows)
result_df.to_csv("inference_results.csv", index=False, encoding="utf-8-sig")

if total > 0:
    accuracy = correct / total
    print(f"✅ 準確率：{accuracy*100:.2f}%（{correct}/{total} 正確）")
    print(f"❌ 失敗率：{(1-accuracy)*100:.2f}%（{total - correct}/{total} 錯誤）")
else:
    print("⚠️ 無有效樣本可評估")

print("📄 已產出 inference_results.csv")
