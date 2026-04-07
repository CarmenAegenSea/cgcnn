import os
import pandas as pd
import shutil

#==========================================================
# 遍历data/catalysis文件夹中的所有文件夹，生成id_prop.csv文件
#==========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MASTER_CSV = os.path.join(PROJECT_ROOT, "data", "catalysis", "id_prop.csv")
CV_BASE_DIR = os.path.join(PROJECT_ROOT, "data", "catalysis")
ATOM_INIT_SRC = os.path.join(PROJECT_ROOT, "data", "catalysis", "atom_init.json")

print(f"脚本目录: {SCRIPT_DIR}")
print(f"项目根目录: {PROJECT_ROOT}")
print(f"总表路径: {MASTER_CSV}")
print(f"总表是否存在: {os.path.exists(MASTER_CSV)}")

if not os.path.exists(MASTER_CSV):
    raise FileNotFoundError(f"找不到总表文件: {MASTER_CSV}")
if not os.path.exists(ATOM_INIT_SRC):
    raise FileNotFoundError(f"找不到原子特征文件: {ATOM_INIT_SRC}")

master_df = pd.read_csv(MASTER_CSV, header=None, names=['id', 'target'])
master_df['id'] = master_df['id'].astype(str)
data_map = dict(zip(master_df['id'], master_df['target']))
print(f"已加载总表，共 {len(data_map)} 条数据。")

# 遍历数字文件夹 1~5
for i in range(1, 6):
    fold_dir = os.path.join(CV_BASE_DIR, str(i))
    if not os.path.isdir(fold_dir):
        print(f"警告：文件夹 {fold_dir} 不存在，跳过。")
        continue

    cif_files = [f[:-4] for f in os.listdir(fold_dir) if f.endswith('.cif')]
    if not cif_files:
        print(f"警告：{fold_dir} 中没有 .cif 文件，跳过。")
        continue

    fold_data = []
    for cid in cif_files:
        if cid in data_map:
            fold_data.append([cid, data_map[cid]])
        else:
            print(f"错误：在总表中找不到 {cid} 的数据")

    # 生成 id_prop.csv
    fold_df = pd.DataFrame(fold_data)
    fold_csv_path = os.path.join(fold_dir, "id_prop.csv")
    fold_df.to_csv(fold_csv_path, index=False, header=False)

    # 复制 atom_init.json
    shutil.copy(ATOM_INIT_SRC, os.path.join(fold_dir, "atom_init.json"))

    print(f"Fold {i} 处理完成：匹配到 {len(fold_data)} 个文件")

print("\n所有文件夹已修复完毕！")