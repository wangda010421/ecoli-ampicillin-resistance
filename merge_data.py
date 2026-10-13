import pandas as pd

labels = pd.read_csv("data/processed/labels.csv", dtype=str)
genes = pd.read_csv("data/raw/isolates_ampicillin.csv", dtype=str)

labels = labels.rename(columns={"#BioSample": "BioSample"})

merged = labels.merge(
    genes,
    on="BioSample",
    how="left",
    validate="one_to_one",
    indicator=True
)

print("匹配情况：")
print(merged["_merge"].value_counts())

missing = merged.loc[
    merged["_merge"] == "left_only", ["BioSample"]
]
print("\n未匹配样本，最多显示前 10 个：")
print(missing.head(10).to_string(index=False))

merged.to_csv("data/processed/merged_audit.csv", index=False)
