import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

predictions = pd.read_csv(
    "results/baseline/predictions.csv"
)

matrix = confusion_matrix(
    predictions["y"],
    predictions["pred"],
    labels=[0, 1]
)

fig, ax = plt.subplots(figsize=(5, 4))

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=["Susceptible", "Resistant"]
)
display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

ax.set_title("Baseline: grouped test set")
fig.tight_layout()

fig.savefig(
    "results/baseline/confusion_matrix.png",
    dpi=200
)
plt.close(fig)

print("混淆矩阵：")
print(matrix)
print("图已保存：results/baseline/confusion_matrix.png")
