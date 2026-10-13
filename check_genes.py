import pandas as pd
from collections import Counter

df = pd.read_csv("data/processed/merged_audit.csv", dtype=str)

categories = Counter()
missing = 0
none_found = 0

for text in df["AMR genotypes"]:
    if pd.isna(text) or not text.strip():
        missing += 1
        continue

    if text.strip() == "<NONE>":
        none_found += 1
        continue

    for entry in text.split(","):
        name, category = entry.rsplit("=", 1)
        categories[category.strip()] += 1

print("基因结果缺失的样本数：", missing)
print("明确未检出的样本数：", none_found)
print("各类别条目数：")
for category, count in categories.items():
    print(category, count)
def parse_entries(text):
    if pd.isna(text) or not text.strip():
        raise ValueError("基因结果缺失，不能当作未检出")

    if text.strip() == "<NONE>":
        return []

    return sorted(set(
        entry.strip() for entry in text.split(",")
    ))

df["entries"] = df["AMR genotypes"].apply(parse_entries)

print("\n第一个样本：", df.iloc[0]["BioSample"])
print("前 5 个条目：", df.iloc[0]["entries"][:5])
print("条目数：", len(df.iloc[0]["entries"]))
from sklearn.model_selection import train_test_split

df["y"] = df["y"].astype(int)

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["y"],
    random_state=42
)

print("\n训练集样本数：", len(train_df))
print("测试集样本数：", len(test_df))

print("\n训练集类别比例：")
print(train_df["y"].value_counts(normalize=True))

print("\n测试集类别比例：")
print(test_df["y"].value_counts(normalize=True))

overlap = set(train_df["BioSample"]) & set(test_df["BioSample"])
assert not overlap, "训练集和测试集出现重复样本"
from sklearn.preprocessing import MultiLabelBinarizer

encoder = MultiLabelBinarizer(sparse_output=True)

X_train = encoder.fit_transform(train_df["entries"])
X_test = encoder.transform(test_df["entries"])

y_train = train_df["y"].to_numpy()
y_test = test_df["y"].to_numpy()

print("\n训练特征矩阵：", X_train.shape)
print("测试特征矩阵：", X_test.shape)
print("前 5 个特征名：", encoder.classes_[:5])

known = set(encoder.classes_)
test_entries = {
    entry
    for entries in test_df["entries"]
    for entry in entries
}
unknown = test_entries - known

print("测试集独有条目数：", len(unknown))
print("测试集独有条目示例：", sorted(unknown)[:5])
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 对照模型：无论基因如何，都预测训练集中最多的类别
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)

# 学习基因特征与耐药标签的关系
model = LogisticRegression(
    solver="liblinear",
    max_iter=1000,
    random_state=42
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\n多数类基线准确率：",
      round(accuracy_score(y_test, baseline_pred), 4))
print("逻辑回归准确率：",
      round(accuracy_score(y_test, pred), 4))
print("混淆矩阵：")
print(confusion_matrix(y_test, pred, labels=[0, 1]))
train_clusters = set(train_df["SNP cluster"].dropna())
test_clusters = set(test_df["SNP cluster"].dropna())

shared_clusters = train_clusters & test_clusters

test_has_cluster = test_df["SNP cluster"].notna()
test_shared = test_df["SNP cluster"].isin(shared_clusters)

print("\n训练与测试共有的 SNP 群数：", len(shared_clusters))
print("测试集中有 SNP 群信息的样本数：", test_has_cluster.sum())
print("测试集中与训练集同群的样本数：", test_shared.sum())

if test_has_cluster.sum() > 0:
    print("有群信息的测试样本中，同群比例：",
          round(test_shared.sum() / test_has_cluster.sum(), 4))
from sklearn.model_selection import GroupShuffleSplit

cluster = df["SNP cluster"].fillna("").str.strip()
has_cluster = cluster.ne("")

groups = "sample:" + df["BioSample"]
groups.loc[has_cluster] = "cluster:" + cluster.loc[has_cluster]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(df, df["y"], groups=groups)
)

group_train = df.iloc[train_idx]
group_test = df.iloc[test_idx]

assert not (
    set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
), "分组出现重叠"

print("\n分组训练样本数：", len(group_train))
print("分组测试样本数：", len(group_test))
print("分组测试类别比例：")
print(group_test["y"].value_counts(normalize=True))

group_encoder = MultiLabelBinarizer(sparse_output=True)
GX_train = group_encoder.fit_transform(group_train["entries"])
GX_test = group_encoder.transform(group_test["entries"])

group_model = LogisticRegression(
    solver="liblinear",
    max_iter=1000,
    random_state=42
)
group_model.fit(GX_train, group_train["y"])
group_pred = group_model.predict(GX_test)

print("分组测试准确率：",
      round(accuracy_score(group_test["y"], group_pred), 4))
print("分组测试混淆矩阵：")
print(confusion_matrix(group_test["y"], group_pred, labels=[0, 1]))
weights = pd.Series(
    group_model.coef_[0],
    index=group_encoder.classes_
)

print("\n与耐药预测正向关联最大的 10 个特征：")
print(weights.sort_values(ascending=False).head(10))

print("\n与耐药预测负向关联最大的 10 个特征：")
print(weights.sort_values().head(10))
selected = [
    "blaCMY-2=COMPLETE",
    "blaTEM-1=COMPLETE",
    "mcr-1.1=COMPLETE",
    "blaNDM-5=COMPLETE"
]

for feature in selected:
    present = group_train["entries"].apply(
        lambda entries: feature in entries
    )
    count = int(present.sum())

    print("\n特征：", feature)
    print("训练集中检出样本数：", count)

    if count > 0:
        resistance_rate = group_train.loc[present, "y"].mean()
        print("检出样本中的耐药比例：",
              round(resistance_rate, 4))
errors = group_test[
    ["BioSample", "Resistance phenotype", "y",
     "AMR genotypes", "SNP cluster",
     "AMRFinderPlus analysis type"]
].copy()

errors["pred"] = group_pred
errors["p_resistant"] = group_model.predict_proba(GX_test)[:, 1]

missed = errors.loc[
    (errors["y"] == 1) & (errors["pred"] == 0)
].sort_values("p_resistant")

missed.to_csv(
    "data/processed/missed_resistant.csv",
    index=False
)

print("\n实际耐药但预测敏感的样本数：", len(missed))
print(missed[
    ["BioSample", "y", "pred", "p_resistant"]
].head(5).to_string(index=False))
sample_id = missed.iloc[0]["BioSample"]

print("\n检查样本：", sample_id)

sample_row = group_test.loc[
    group_test["BioSample"] == sample_id
].iloc[0]

print("分析类型：", sample_row["AMRFinderPlus analysis type"])
print("SNP 群：", sample_row["SNP cluster"])

print("\n报告的基因／突变条目：")
known_group = set(group_encoder.classes_)
ast = pd.read_csv("data/asts_fixed.csv", dtype=str)

records = ast.loc[
    ast["#BioSample"] == sample_id,
    ["#BioSample", "Antibiotic", "Resistance phenotype",
     "Measurement sign", "MIC (mg/L)",
     "Disk diffusion (mm)", "Testing standard"]
]

print("\n原始药敏记录：")
print(records.to_string(index=False))
for entry in sample_row["entries"]:
    status = "模型已知" if entry in known_group else "训练集未见"
    print(entry, status)
sample_entries = set(sample_row["entries"])

contributions = pd.Series({
    entry: weights[entry]
    for entry in sample_entries
    if entry in weights.index
}).sort_values()

intercept = group_model.intercept_[0]
score = intercept + contributions.sum()

print("\n该样本已知特征的贡献：")
print(contributions.to_string())

print("\n截距：", round(intercept, 4))
print("特征贡献总和：", round(contributions.sum(), 4))
print("最终预测分数：", round(score, 4))
has_ndm5 = group_train["entries"].apply(
    lambda entries: "blaNDM-5=COMPLETE" in entries
)

ndm5_ids = group_train.loc[has_ndm5, "BioSample"]

ndm5_records = ast.loc[
    ast["#BioSample"].isin(ndm5_ids),
    ["#BioSample", "Resistance phenotype",
     "Measurement sign", "MIC (mg/L)", "Testing standard"]
]

print("\nblaNDM-5 训练样本的药敏记录：")
print(ndm5_records.to_string(index=False))
train_ast = ast.loc[
    ast["#BioSample"].isin(group_train["BioSample"])
].copy()

mic = pd.to_numeric(train_ast["MIC (mg/L)"], errors="coerce")

suspect = train_ast.loc[
    (train_ast["Resistance phenotype"] == "susceptible")
    & (train_ast["Measurement sign"] == ">")
    & (mic == 32)
]

print("\n训练集中：敏感标签且 MIC >32 的记录数：", len(suspect))
print("对应 BioSample 数：", suspect["#BioSample"].nunique())

print("\n这些记录的检测标准：")
print(suspect["Testing standard"].value_counts(dropna=False))

suspect.to_csv(
    "data/processed/suspect_sensitive_gt32.csv",
    index=False
)
genes = pd.read_csv(
    "data/raw/isolates_ampicillin.csv",
    dtype=str
)
suspect_ids = set(suspect["#BioSample"])

source_rows = genes.loc[
    genes["BioSample"].isin(suspect_ids),
    ["BioSample", "Strain", "Location", "Create date"]
].copy()

source_rows["Create day"] = source_rows["Create date"].str[:10]

print("\n可疑样本的地点分布：")
print(source_rows["Location"].value_counts(dropna=False))

print("\n可疑样本的收录日期分布：")
print(source_rows["Create day"].value_counts().head(10))

print("\n前 5 个样本的来源信息：")
print(source_rows.head(5).to_string(index=False))
# 只从训练集排除可疑样本
filtered_train = group_train.loc[
    ~group_train["BioSample"].isin(suspect_ids)
].copy()

print("\n过滤后训练样本数：", len(filtered_train))
print("排除样本数：", len(group_train) - len(filtered_train))

# 沿用原编码器，保持特征列一致
FX_train = group_encoder.transform(filtered_train["entries"])

filtered_model = LogisticRegression(
    solver="liblinear",
    max_iter=1000,
    random_state=42
)
filtered_model.fit(FX_train, filtered_train["y"])

# 使用同一个测试集
filtered_pred = filtered_model.predict(GX_test)

print("\n原模型准确率：",
      round(accuracy_score(group_test["y"], group_pred), 4))
print("过滤后模型准确率：",
      round(accuracy_score(group_test["y"], filtered_pred), 4))

print("过滤后混淆矩阵：")
print(confusion_matrix(
    group_test["y"], filtered_pred, labels=[0, 1]
))

# 检查此前反常的权重如何变化
feature = "blaNDM-5=COMPLETE"
feature_idx = list(group_encoder.classes_).index(feature)

print("\nblaNDM-5 原权重：",
      round(group_model.coef_[0][feature_idx], 4))
print("blaNDM-5 过滤后权重：",
      round(filtered_model.coef_[0][feature_idx], 4))
comparison = group_test[["BioSample", "y"]].copy()

comparison["original_pred"] = group_pred
comparison["filtered_pred"] = filtered_pred

comparison["original_correct"] = (
    comparison["original_pred"] == comparison["y"]
)
comparison["filtered_correct"] = (
    comparison["filtered_pred"] == comparison["y"]
)

changed = comparison.loc[
    comparison["original_pred"] != comparison["filtered_pred"]
]

improved = (
    ~comparison["original_correct"]
    & comparison["filtered_correct"]
).sum()

worsened = (
    comparison["original_correct"]
    & ~comparison["filtered_correct"]
).sum()

comparison.to_csv(
    "data/processed/model_comparison.csv", index=False
)
changed.to_csv(
    "data/processed/changed_predictions.csv", index=False
)

print("\n预测发生变化的样本数：", len(changed))
print("由错误变正确：", improved)
print("由正确变错误：", worsened)
split_manifest = df[["BioSample", "y"]].copy()
split_manifest["group_id"] = groups
split_manifest["split"] = "train"
split_manifest.loc[group_test.index, "split"] = "test"

split_manifest["excluded_from_filtered_train"] = (
    split_manifest["split"].eq("train")
    & split_manifest["BioSample"].isin(suspect_ids)
)

split_manifest.to_csv(
    "data/processed/split_manifest.csv",
    index=False
)

print("\n实验划分已保存")
print(split_manifest["split"].value_counts())
