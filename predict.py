import argparse
import joblib
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("input_csv", help="包含基因结果的CSV")
parser.add_argument("output_csv", help="预测结果保存位置")
args = parser.parse_args()

bundle = joblib.load("results/baseline/model_bundle.joblib")
encoder = bundle["encoder"]
model = bundle["model"]

data = pd.read_csv(args.input_csv, dtype=str)

required = {"BioSample", "AMR genotypes"}
if not required.issubset(data.columns):
    raise ValueError(f"输入必须包含这些列：{required}")


def parse_entries(text):
    if pd.isna(text) or not text.strip():
        raise ValueError("存在缺失基因结果，不能直接预测")
    if text.strip() == "<NONE>":
        return []
    return sorted(set(
        item.strip() for item in text.split(",")
    ))


entries = data["AMR genotypes"].apply(parse_entries)

known = set(encoder.classes_)
unknown_counts = entries.apply(
    lambda items: len(set(items) - known)
)

X = encoder.transform(entries)
pred = model.predict(X)
probability = model.predict_proba(X)[:, 1]

output = data[["BioSample"]].copy()
output["prediction"] = [
    bundle["label_meaning"][int(value)] for value in pred
]
output["p_resistant"] = probability
output["unknown_entry_count"] = unknown_counts.to_numpy()

output.to_csv(args.output_csv, index=False)
print(f"已预测 {len(output)} 个样本")
print(output.head().to_string(index=False))
