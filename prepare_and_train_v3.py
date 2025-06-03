# prepare_and_train_v3.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import pandas as pd


class SimpleTCN(nn.Module):
    def __init__(self, input_size=400 * 12, num_classes=2):
        super(SimpleTCN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.fc(x)


def load_features_and_labels(features_dir):
    def collect_data(split):
        X, y = [], []
        for subdir in os.listdir(features_dir):
            split_path = os.path.join(features_dir, subdir, split)
            if not os.path.isdir(split_path):
                continue
            for file in os.listdir(split_path):
                if file.endswith(".npy"):
                    npy_path = os.path.join(split_path, file)
                    csv_path = npy_path.replace(".npy", ".csv")
                    try:
                        features = np.load(npy_path).flatten()
                        label_df = pd.read_csv(csv_path)
                        label = int(label_df.iloc[0]["is_strike"])
                        X.append(features)
                        y.append(label)
                    except Exception as e:
                        print(f"❌ 失敗讀取 {file}：{e}")
        return np.array(X), np.array(y)

    X_train, y_train = collect_data("train")
    X_test, y_test = collect_data("test")
    return X_train, y_train, X_test, y_test


def train_and_evaluate():
    X_train, y_train, X_test, y_test = load_features_and_labels("features")

    print(f"📊 訓練資料筆數：{len(X_train)}，測試資料筆數：{len(X_test)}")

    model = SimpleTCN(input_size=X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    model.train()
    for epoch in range(20):
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/20, Loss: {loss.item():.4f}")

    # 測試
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_pred = model(X_test_tensor).argmax(dim=1).numpy()
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ 測試正確率：{acc:.2%}")

    # 儲存模型
    torch.save(model.state_dict(), "model_strike_predictor_v5.pth")
    print("💾 模型已儲存為 model_strike_predictor_v5.pth")


if __name__ == "__main__":
    train_and_evaluate()
