# 大肠杆菌氨苄西林耐药预测

## 目标
根据公开的基因与突变检出条目，预测氨苄西林耐药或敏感。

## 数据处理顺序
1. fix_csv.py：修复原始CSV的异常引号。
2. explore.py：检查药敏数据，生成标签表。
3. merge_data.py：按BioSample合并标签与基因信息。
4. check_genes.py：特征编码、模型评估和探索性分析。

## 当前实验
- 候选样本：8146。
- 分组训练集：6506；测试集：1640。
- 基因条目保留名称和检出类别。
- 原分组模型准确率：95.43%。
- 排除89个可疑训练样本后：95.98%。
- 当前测试集已用于探索，不是独立外部验证。

## 复现资料
- requirements.txt：Python包版本。
- data/processed/split_manifest.csv：固定实验划分。
- data_quality_notes.md：标签质量问题与处理依据。
- data/processed/model_comparison.csv：逐样本预测对照。
## 运行基线模型

python train_baseline.py

## 使用已保存模型预测

python predict.py 输入.csv 输出.csv

输入必须包含：
- BioSample
- AMR genotypes

基因条目格式示例：
blaTEM=COMPLETE,gyrA_S83L=POINT

## 已完成的验证
- 独立训练脚本复现了原分组实验结果。
- 保存并加载模型后，单样本预测一致。
- 批量接口对1640个测试样本的预测全部一致。
- 接口一致性验证不等于独立数据上的性能验证。
