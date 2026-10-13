import joblib
import numpy as np
import pandas as pd

bundle = joblib.load("results/baseline/model_bundle.joblib")
encoder = bundle["encoder"]
model = bundle["model"]

saved = pd.read_csv("results/baseline/predictions.csv")
data = pd.read_csv("data/processed/merged_audit.csv", dtype=str)

# 选原测试集的第一条预测记录
previous = saved.iloc[0]
sample_id = previous["BioSample"]

rows = data.loc[data["BioSample"] == sample_id]
assert len(rows) == 1, "样本缺失或编号重复"

text = rows.iloc[0]["AMR genotypes"]

if pd.isna(text) or not text.strip():
    raise ValueError("基因结果缺失")
elif text.strip() == "<NONE>":
    entries = []
else:
    entries = sorted(set(
        entry.strip() for entry in text.split(",")
    ))

X = encoder.transform([entries])
pred = int(model.predict(X)[0])
probability = float(model.predict_proba(X)[0, 1])

print("样本：", sample_id)
print("预测类别：", bundle["label_meaning"][pred])
print("耐药概率：", round(probability, 6))
print("之前保存的概率：", round(previous["p_resistant"], 6))

assert pred == int(previous["pred"]), "预测类别不一致"
assert np.isclose(
    probability, previous["p_resistant"],
    rtol=0, atol=1e-10
), "预测概率不一致"

print("验证通过：加载后的预测与保存结果一致")
