import os
import subprocess
import shutil
import pandas as pd
from sklearn.model_selection import KFold

# 执行交叉验证

# 原始数据目录：存放所有 .cif 文件、id_prop.csv 和 atom_init.json
ORIGIN_DIR = "data/yuan"

# 临时工作根目录（用于存放每一折的临时数据）
WORK_BASE = "data/kfold_temp"

# 折数
N_SPLITS = 5

# CGCNN 主程序路径
CGCNN_MAIN = "main.py"

# 是否使用 GPU
USE_CUDA = True

# 训练额外参数（可根据需要调整）
EXTRA_ARGS = [
    "--epochs", "200",
    "--batch-size", "32",
    "--lr", "0.01",
    "--optim", "SGD",
    # "--disable-cuda",    # 如果 USE_CUDA = False，取消这行注释
]

def prepare_fold_data(fold_idx, train_ids, val_ids, df, work_dir):
    fold_dir = os.path.join(work_dir, str(fold_idx))
    # 如果目录已存在，先删除
    if os.path.exists(fold_dir):
        shutil.rmtree(fold_dir)
    os.makedirs(fold_dir)

    # 写入 id_prop.csv
    train_df = df.iloc[train_ids]
    val_df = df.iloc[val_ids]
    # 训练集 + 验证集合并写入（根据 --train-size 和 --val-size 按顺序切分）
    combined = pd.concat([train_df, val_df], ignore_index=True)
    combined.to_csv(os.path.join(fold_dir, "id_prop.csv"), index=False, header=False)

    # 复制 atom_init.json
    shutil.copy(os.path.join(ORIGIN_DIR, "atom_init.json"), fold_dir)

    # 复制所有 .cif 文件
    for fname in os.listdir(ORIGIN_DIR):
        if fname.endswith(".cif"):
            shutil.copy(os.path.join(ORIGIN_DIR, fname), fold_dir)

    return fold_dir

def run_cgcnn(data_dir, train_size, val_size):
    """调用 CGCNN 训练"""
    cmd = ["python", CGCNN_MAIN,
           "--train-size", str(train_size),
           "--val-size", str(val_size),
           data_dir]
    if not USE_CUDA:
        cmd.append("--disable-cuda")
    cmd.extend(EXTRA_ARGS)
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    # 1. 读取总表
    csv_path = os.path.join(ORIGIN_DIR, "id_prop.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到总表文件: {csv_path}")
    df = pd.read_csv(csv_path, header=None, names=["id", "target"])
    print(f"总样本数: {len(df)}")

    # 2. 初始化 KFold（随机打乱，固定种子以保证可重复）
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # 3. 创建临时工作目录
    if os.path.exists(WORK_BASE):
        shutil.rmtree(WORK_BASE)
    os.makedirs(WORK_BASE)

    # 4. 循环每一折
    for fold, (train_idx, val_idx) in enumerate(kf.split(df), start=1):
        print(f"\n{'='*50}")
        print(f"开始第 {fold} 折训练")
        print(f"训练集大小: {len(train_idx)}, 验证集大小: {len(val_idx)}")
        print(f"{'='*50}")

        # 准备该折的数据目录
        fold_dir = prepare_fold_data(fold, train_idx, val_idx, df, WORK_BASE)

        # 运行 CGCNN 训练
        run_cgcnn(fold_dir, len(train_idx), len(val_idx))

        print(f"第 {fold} 折训练完成\n")

    print("所有折训练完毕！")
    print(f"临时数据保存在: {WORK_BASE}")
    print("你可以手动收集各折的验证集最佳误差（查看 log.csv）")

if __name__ == "__main__":
    main()