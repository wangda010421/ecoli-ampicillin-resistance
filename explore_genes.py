import pandas as pd

df = pd.read_csv("data/raw/isolates_example.csv", dtype=str)

sample = df.iloc[0]
gene_text = sample["AMR genotypes"]

entries = gene_text.split(",")

print("样本编号：", sample["BioSample"])
print("条目数量：", len(entries))

for entry in entries:
    name, category = entry.rsplit("=", 1)
    print(name, category)
names = [
    entry.rsplit("=", 1)[0].strip()
    for entry in entries
]

features = dict.fromkeys(names, 1)

feature_df = pd.DataFrame(
    [features],
    index=[sample["BioSample"]]
)
feature_df.index.name = "BioSample"

print("\n示例特征表：")
print(feature_df[["blaTEM", "blaCTX-M-55", "gyrA_S83L"]])
