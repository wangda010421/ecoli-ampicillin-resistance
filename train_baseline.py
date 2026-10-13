from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)


def parse_entries(text):
    if pd.isna(text) or not text.strip():
        raise ValueError("基因结果缺失")

    if text.strip() == "<NONE>":
        return []

    return sorted(set(
        entry.strip() for entry in text.split(",")
    ))


# 1. 读取数据与固定划分
data = pd.read_csv("data/processed/merged_audit.csv", dtype=str)
manifest = pd.read_csv("data/processed/split_manifest.csv", dtype=str)

data = data.merge(
    manifest[["BioSample", "split"]],
    on="BioSample",
    how="left",
    validate="one_to_one"
)

assert data["split"].isin(["train", "test"]).all(), "划分缺失或异常"

data["y"] = data["y"].astype(int)
data["entries"] = data["AMR genotypes"].apply(parse_entries)

train = data.loc[data["split"] == "train"]
test = data.loc[data["split"] == "test"]

# 2. 只用训练集建立特征列
encoder = MultiLabelBinarizer(sparse_output=True)
X_train = encoder.fit_transform(train["entries"])
X_test = encoder.transform(test["entries"])

# 3. 训练原始基线模型
model = LogisticRegression(
    solver="liblinear",
    max_iter=1000,
    random_state=42
)
model.fit(X_train, train["y"])

# 4. 评估并保存逐样本预测
pred = model.predict(X_test)

print("训练样本数：", len(train))
print("测试样本数：", len(test))
print("准确率：", round(accuracy_score(test["y"], pred), 4))
print("耐药精确率：",
      round(precision_score(test["y"], pred, zero_division=0), 4))
print("耐药召回率：",
      round(recall_score(test["y"], pred, zero_division=0), 4))
print("混淆矩阵：")
print(confusion_matrix(test["y"], pred, labels=[0, 1]))

output_dir = Path("results/baseline")
output_dir.mkdir(parents=True, exist_ok=True)

predictions = test[["BioSample", "y"]].copy()
predictions["pred"] = pred
predictions["p_resistant"] = model.predict_proba(X_test)[:, 1]
predictions.to_csv(output_dir / "predictions.csv", index=False)
import joblib

bundle = {
    "encoder": encoder,
    "model": model,
    "feature_format": "name=category",
    "label_meaning": {0: "susceptible", 1: "resistant"}
}

joblib.dump(bundle, output_dir / "model_bundle.joblib")

print("\n模型与编码器已保存")
