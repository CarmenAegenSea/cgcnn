import os
import random
import shutil
import pandas as pd

#================================================
# 将data/yuan中的素材均匀且随机的分配到5个文件夹内
#================================================

# ===== 路径配置 =====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录：向上两级，即 C:\Users\22616\PycharmProjects\cgcnn
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# yuan 文件夹的实际位置：项目根目录/cgcnn/data/yuan
YUAN_DIR = os.path.join(PROJECT_ROOT, "cgcnn", "data", "yuan")
MASTER_CSV = os.path.join(YUAN_DIR, "id_prop.csv")          # 总表也在 yuan 内

# 目标根目录：cgcnn/data/catalysis（存放 1~5 子文件夹）
DATA_CATALYSIS = os.path.join(PROJECT_ROOT, "cgcnn", "data", "catalysis")

# atom_init.json 的位置（从 pre-trained 目录复制到 catalysis 目录）
ATOM_INIT_SRC = os.path.join(PROJECT_ROOT, "pre-trained", "atom_init.json")

# 输出路径信息供确认
print(f"脚本目录: {SCRIPT_DIR}")
print(f"项目根目录: {PROJECT_ROOT}")
print(f"yuan 文件夹: {YUAN_DIR}")
print(f"总表文件: {MASTER_CSV}")
print(f"目标 catalysis 目录: {DATA_CATALYSIS}")
print(f"原子特征文件源路径: {ATOM_INIT_SRC}")

# ===== 检查必要文件 =====
if not os.path.exists(YUAN_DIR):
    raise FileNotFoundError(f"找不到 yuan 文件夹: {YUAN_DIR}")
if not os.path.exists(MASTER_CSV):
    raise FileNotFoundError(f"找不到总表文件: {MASTER_CSV}")
if not os.path.exists(ATOM_INIT_SRC):
    print(f"警告: 原子特征文件 {ATOM_INIT_SRC} 不存在，请确保项目中有 pre-trained 目录。")
    print("如果缺少，可以从 CGCNN 官方仓库的 pre-trained 目录下载 atom_init.json。")
    # 不强制退出，只是无法复制

# ===== 读取总表 =====
master_df = pd.read_csv(MASTER_CSV, header=None, names=["id", "target"])
master_df["id"] = master_df["id"].astype(str)
data_map = dict(zip(master_df["id"], master_df["target"]))
print(f"已从总表读取 {len(data_map)} 条数据。")

# ===== 收集 yuan 下的 .cif 文件 =====
cif_files = [f[:-4] for f in os.listdir(YUAN_DIR) if f.endswith(".cif")]
print(f"在 yuan 文件夹中找到 {len(cif_files)} 个 .cif 文件。")

valid_ids = [cid for cid in cif_files if cid in data_map]
print(f"其中有效（在总表中有记录）的材料数: {len(valid_ids)}")

if len(valid_ids) == 0:
    raise RuntimeError("没有找到任何有效的材料，请检查 yuan 中的 .cif 文件名是否与 id_prop.csv 中的 id 列匹配。")

# ===== 随机打乱并均匀分成 5 份 =====
random.seed(42)   # 可删除该行以获得完全随机分配
random.shuffle(valid_ids)

n = len(valid_ids)
splits = []
for i in range(5):
    start = i * n // 5
    end = (i + 1) * n // 5
    splits.append(valid_ids[start:end])

# ===== 创建 1~5 文件夹并写入数据 =====
for i in range(1, 6):
    fold_dir = os.path.join(DATA_CATALYSIS, str(i))
    # 删除旧文件夹并重建
    if os.path.exists(fold_dir):
        shutil.rmtree(fold_dir)
    os.makedirs(fold_dir)

    ids_this_fold = splits[i-1]
    print(f"\nFold {i}: {len(ids_this_fold)} 个材料")

    fold_data = []
    for cid in ids_this_fold:
        src_cif = os.path.join(YUAN_DIR, cid + ".cif")
        dst_cif = os.path.join(fold_dir, cid + ".cif")
        shutil.copy2(src_cif, dst_cif)
        fold_data.append([cid, data_map[cid]])

    # 写入 id_prop.csv（无表头）
    fold_csv = os.path.join(fold_dir, "id_prop.csv")
    pd.DataFrame(fold_data).to_csv(fold_csv, index=False, header=False)

    # 复制 atom_init.json（如果源文件存在）
    if os.path.exists(ATOM_INIT_SRC):
        shutil.copy2(ATOM_INIT_SRC, os.path.join(fold_dir, "atom_init.json"))
    else:
        print(f"  警告: 未复制 atom_init.json，因为源文件不存在。")

print("\n随机分配完成！生成的文件夹：")
for i in range(1, 6):
    print(f"  - {os.path.join(DATA_CATALYSIS, str(i))}")