import pandas as pd
import numpy as np

data = pd.read_csv("data/processed/merged_audit.csv", dtype=str)
saved = pd.read_csv("results/baseline/predictions.csv")

# 按保存的测试样本编号提取输入
test_input = saved[["BioSample"]].merge(
    data[["BioSample", "AMR genotypes"]],
    on="BioSample",
    validate="one_to_one"
)
test_input.to_csv("results/test_input.csv", index=False)
print("已生成测试输入，样本数：", len(test_input))
new = pd.read_csv("results/test_predictions.csv")

check = saved.merge(
    new,
    on="BioSample",
    how="outer",
    validate="one_to_one",
    indicator=True
)

assert check["_merge"].eq("both").all(), "样本名单不一致"

new_labels = check["prediction"].map({
    "susceptible": 0,
    "resistant": 1
})
assert (check["pred"] == new_labels).all(), "类别不一致"

assert np.allclose(
    check["p_resistant_x"],
    check["p_resistant_y"],
    rtol=0,
    atol=1e-10
), "概率不一致"

print("批量接口验证通过：全部测试样本预测一致")
