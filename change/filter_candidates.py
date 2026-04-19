import pandas as pd

# 请根据你的实际路径修改
PRED_FILE = 'C:\\Users\\22616\\PycharmProjects\\cgcnn\\test_results.csv'                     # 预测结果
ATTR_FILE = 'C:\\Users\\22616\\PycharmProjects\cgcnn\\cgcnn\\data\\tmc_data\\tmc_all_materials.csv'       # TMCs 原始属性
OUTPUT_FILE = 'final_candidates.csv'

# 读取预测结果（无表头，列名自定义）
pred_df = pd.read_csv(PRED_FILE, header=None, names=['material_id', 'target', 'predicted_bandgap'])

# 读取 TMCs 属性
attr_df = pd.read_csv(ATTR_FILE)

# 合并
merged = pred_df.merge(attr_df, left_on='material_id', right_on='material_id', how='inner')

# 筛选
# 放宽稳定性要求（允许亚稳态）
filtered = merged[
    (merged['predicted_bandgap'] >= 1.0) &
    (merged['predicted_bandgap'] <= 3.0) &
    (merged['formation_energy_per_atom'] <= 0.1)  # 仍要求形成能≤0，但不再强制 is_stable
]

# 保存结果
filtered[['material_id', 'formula', 'predicted_bandgap', 'formation_energy_per_atom', 'crystal_system']].to_csv(OUTPUT_FILE, index=False)

print(f"筛选完成！最终候选材料数: {len(filtered)}")
print(f"结果保存至: {OUTPUT_FILE}")