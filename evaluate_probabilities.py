import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve
)

predictions = pd.read_csv(
    "results/baseline/predictions.csv"
)

y_true = predictions["y"]
probability = predictions["p_resistant"]

ap = average_precision_score(y_true, probability)

precision, recall, thresholds = precision_recall_curve(
    y_true, probability
)

print("测试集耐药比例：", round(y_true.mean(), 4))
print("平均精确率 AP：", round(ap, 4))
print("候选阈值数量：", len(thresholds))

curve = pd.DataFrame({
    "recall": recall,
    "precision": precision
})
curve.to_csv("results/baseline/pr_curve.csv", index=False)
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_score, recall_score

fig, ax = plt.subplots(figsize=(6, 5))

ax.step(
    recall,
    precision,
    where="post",
    label=f"Baseline (AP={ap:.4f})"
)

ax.axhline(
    y_true.mean(),
    color="gray",
    linestyle="--",
    label="Prevalence reference"
)

# 标出之前混淆矩阵对应的默认阈值
default_pred = predictions["pred"]

ax.scatter(
    recall_score(y_true, default_pred),
    precision_score(y_true, default_pred),
    color="red",
    zorder=3,
    label="Threshold = 0.5"
)

ax.set(
    xlabel="Recall (resistant)",
    ylabel="Precision (resistant)",
    title="Precision-recall: grouped test set",
    xlim=(0, 1),
    ylim=(0, 1.02)
)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()

Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/pr_curve.png", dpi=200)
plt.close(fig)

print("图已保存：figures/pr_curve.png")
