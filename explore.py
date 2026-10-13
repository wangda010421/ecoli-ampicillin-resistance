import pandas as pd

df = pd.read_csv(
    "data/asts_fixed.csv",
    dtype=str,
    on_bad_lines="error"
)

print("数据的行数和列数：", df.shape)
print("列名：", df.columns.tolist())
print(df.head(3).to_string(index=False))
print("\n药敏标签数量：")
print(df["Resistance phenotype"].value_counts(dropna=False))
keep = df["Resistance phenotype"].isin(
    ["resistant", "susceptible"]
)

binary_df = df.loc[keep].copy()

print("\n二分类记录数：", len(binary_df))
print(binary_df["Resistance phenotype"].value_counts())
label_counts = (
    binary_df.groupby("#BioSample")["Resistance phenotype"]
    .nunique()
)

print("\n不同 BioSample 数量：", len(label_counts))

conflict_ids = label_counts[label_counts > 1].index
print("标签冲突的 BioSample 数量：", len(conflict_ids))

conflict_rows = binary_df[
    binary_df["#BioSample"].isin(conflict_ids)
]

print(conflict_rows[
    ["#BioSample", "Resistance phenotype",
     "Measurement sign", "MIC (mg/L)", "Testing standard"]
].to_string(index=False))
from pathlib import Path

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

# 保存冲突记录，保留调查依据
conflict_rows.to_csv(
    output_dir / "label_conflicts.csv", index=False
)

# 排除冲突样本，只取样本编号和标签，再去重
labels = (
    binary_df.loc[
        ~binary_df["#BioSample"].isin(conflict_ids),
        ["#BioSample", "Resistance phenotype"]
    ]
    .drop_duplicates(subset="#BioSample")
    .copy()
)

labels["y"] = labels["Resistance phenotype"].map({
    "susceptible": 0,
    "resistant": 1
})

labels.to_csv(output_dir / "labels.csv", index=False)

print("\n最终候选样本数：", len(labels))
print(labels["y"].value_counts())
