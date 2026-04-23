import pandas as pd
"""
筛选结果中符合条件的项
"""
PRED_FILE = 'test_results_final.csv'
ATTR_FILE = 'C:\\Users\\22616\\PycharmProjects\\cgcnn\\cgcnn\\data\\tmc_data\\tmc_all_materials.csv'   # 请确认实际路径
OUTPUT_FILE = 'final_candidates.csv'

# 读取预测结果
pred_df = pd.read_csv(PRED_FILE)
pred_df = pred_df.rename(columns={'id': 'material_id'})  # 统一列名

# 读取属性表
attr_df = pd.read_csv(ATTR_FILE)

# 检查列名
print("属性表列名:", attr_df.columns.tolist())

# 确定形成能列名（自动匹配常见命名）
form_e_col = None
for col in attr_df.columns:
    if 'formation' in col.lower() and 'energy' in col.lower():
        form_e_col = col
        break
if form_e_col is None:
    raise ValueError("未找到形成能列，请手动指定")

print(f"使用形成能列: {form_e_col}")

# 合并
merged = pred_df.merge(attr_df, on='material_id', how='inner')
print(f"预测材料总数: {len(pred_df)}")
print(f"匹配成功数量: {len(merged)}")

if len(merged) == 0:
    print("警告：预测结果与属性表无匹配！请检查ID格式。")
    # 打印前几个ID对比
    print("预测ID示例:", pred_df['material_id'].head().tolist())
    print("属性表ID示例:", attr_df['material_id'].head().tolist())
    exit()

# 筛选条件
cond1 = merged['predicted_bandgap_eV'].between(1.6, 2.8)
cond2 = merged[form_e_col] <= 0.0
cond3 = merged['is_stable'] == True

print(f"带隙 1.6-2.8 eV: {cond1.sum()} 个")
print(f"形成能 <= 0: {cond2.sum()} 个")
print(f"同时满足: {(cond1 & cond2).sum()} 个")

filtered = merged[cond1 & cond2 & cond3]

# 保存结果
if len(filtered) > 0:
    filtered[['material_id', 'formula', 'predicted_bandgap_eV', form_e_col, 'crystal_system']].to_csv(OUTPUT_FILE, index=False)
    print(f"筛选完成！最终候选材料数: {len(filtered)}")
else:
    # 尝试放宽条件以排查问题
    print("无同时满足条件的材料，尝试放宽带隙范围...")
    cond1_loose = merged['predicted_bandgap_eV'].between(1.0, 3.0)
    filtered_loose = merged[cond1_loose & cond2]
    print(f"放宽至1.0-3.0 eV后数量: {len(filtered_loose)}")
    if len(filtered_loose) > 0:
        print("部分候选材料示例:")
        print(filtered_loose[['material_id', 'formula', 'predicted_bandgap_eV', form_e_col]].head())