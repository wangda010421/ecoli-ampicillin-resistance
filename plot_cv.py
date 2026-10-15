import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

scores = pd.read_csv(
    "results/cross_validation/fold_scores.csv"
)

fig, ax = plt.subplots(figsize=(7, 4))

for metric in ["accuracy", "precision", "recall"]:
    ax.plot(
        scores["fold"],
        scores[metric] * 100,
        marker="o",
        label=metric
    )

ax.set(
    xlabel="Validation fold",
    ylabel="Score (%)",
    title="Five-fold grouped cross-validation",
    xticks=[1, 2, 3, 4, 5],
    ylim=(0, 100)
)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()

fig.savefig(
    "results/cross_validation/fold_scores.png",
    dpi=200
)
plt.close(fig)

print("图已保存：results/cross_validation/fold_scores.png")
