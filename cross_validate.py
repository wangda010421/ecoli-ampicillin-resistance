import pandas as pd

data = pd.read_csv(
    "data/processed/merged_audit.csv", dtype=str
)
manifest = pd.read_csv(
    "data/processed/split_manifest.csv", dtype=str
)

data = data.merge(
    manifest[["BioSample", "split", "group_id"]],
    on="BioSample",
    how="left",
    validate="one_to_one"
)

assert data["split"].isin(["train", "test"]).all()
assert data["group_id"].notna().all()

train = data.loc[data["split"] == "train"].copy()
train["y"] = train["y"].astype(int)

print("交叉验证使用的样本数：", len(train))
print("组数：", train["group_id"].nunique())
print("类别数量：")
print(train["y"].value_counts())
from sklearn.model_selection import StratifiedGroupKFold

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

folds = list(cv.split(
    train,
    y=train["y"],
    groups=train["group_id"]
))

for fold, (fit_idx, valid_idx) in enumerate(folds, start=1):
    fit_data = train.iloc[fit_idx]
    valid_data = train.iloc[valid_idx]

    shared = (
        set(fit_data["group_id"])
        & set(valid_data["group_id"])
    )
    assert not shared, "训练部分与验证部分存在同组样本"

    print(f"\n第 {fold} 折")
    print("训练样本数：", len(fit_data))
    print("验证样本数：", len(valid_data))
    print("验证部分耐药比例：",
          round(valid_data["y"].mean(), 4))
    print("重叠组数：", len(shared))
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score


def parse_entries(text):
    if pd.isna(text) or not text.strip():
        raise ValueError("基因结果缺失")
    if text.strip() == "<NONE>":
        return []
    return sorted(set(
        entry.strip() for entry in text.split(",")
    ))


train["entries"] = train["AMR genotypes"].apply(parse_entries)

scores = []

for fold, (fit_idx, valid_idx) in enumerate(folds, start=1):
    fit_data = train.iloc[fit_idx]
    valid_data = train.iloc[valid_idx]

    encoder = MultiLabelBinarizer(sparse_output=True)
    X_fit = encoder.fit_transform(fit_data["entries"])
    X_valid = encoder.transform(valid_data["entries"])

    model = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        random_state=42
    )
    model.fit(X_fit, fit_data["y"])
    pred = model.predict(X_valid)

    result = {
        "fold": fold,
        "accuracy": accuracy_score(valid_data["y"], pred),
        "precision": precision_score(
            valid_data["y"], pred, zero_division=0
        ),
        "recall": recall_score(
            valid_data["y"], pred, zero_division=0
        )
    }
    scores.append(result)

    print(
        f"\n第 {fold} 折："
        f"准确率 {result['accuracy']:.4f}，"
        f"耐药精确率 {result['precision']:.4f}，"
        f"耐药召回率 {result['recall']:.4f}"
    )

scores_df = pd.DataFrame(scores)
print("\n各折结果：")
print(scores_df.to_string(index=False))
from pathlib import Path

metrics = ["accuracy", "precision", "recall"]

summary = scores_df[metrics].agg(["mean", "std"]).T

print("\n五折平均值和标准差：")
print(summary.to_string(float_format=lambda value: f"{value:.4f}"))

output_dir = Path("results/cross_validation")
output_dir.mkdir(parents=True, exist_ok=True)

scores_df.to_csv(output_dir / "fold_scores.csv", index=False)
summary.to_csv(output_dir / "summary.csv", index_label="metric")
