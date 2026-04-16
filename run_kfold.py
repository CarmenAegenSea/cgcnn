import os
import subprocess
import shutil
import pandas as pd
import torch
from sklearn.model_selection import KFold

# ========== 配置参数 ==========
ORIGIN_DIR = "data/yuan"                # 原始数据目录
WORK_BASE = "data/kfold_temp"           # 临时工作根目录
N_SPLITS = 5                            # 折数
CGCNN_MAIN = "main.py"                  # CGCNN 训练脚本
CGCNN_PREDICT = "predict.py"            # CGCNN 预测脚本
USE_CUDA = True                         # 是否使用 GPU
EXTRA_ARGS = [
    "--epochs", "200",
    "--batch-size", "32",
    "--lr", "0.01",
    "--optim", "SGD",
]

# ========== 获取脚本绝对路径（关键修改） ==========
CGCNN_MAIN_ABS = os.path.abspath(CGCNN_MAIN)
CGCNN_PREDICT_ABS = os.path.abspath(CGCNN_PREDICT)

# ========== 辅助函数 ==========
def prepare_fold_data(fold_idx, train_ids, val_ids, df, work_dir):
    """准备第 fold_idx 折的数据目录"""
    fold_dir = os.path.join(work_dir, str(fold_idx))
    if os.path.exists(fold_dir):
        shutil.rmtree(fold_dir)
    os.makedirs(fold_dir)

    # 写入 id_prop.csv（先 train 后 val）
    train_df = df.iloc[train_ids]
    val_df = df.iloc[val_ids]
    combined = pd.concat([train_df, val_df], ignore_index=True)
    combined.to_csv(os.path.join(fold_dir, "id_prop.csv"), index=False, header=False)

    # 复制 atom_init.json
    shutil.copy(os.path.join(ORIGIN_DIR, "atom_init.json"), fold_dir)

    # 复制所有 .cif 文件（若文件多可改用符号链接）
    for fname in os.listdir(ORIGIN_DIR):
        if fname.endswith(".cif"):
            shutil.copy(os.path.join(ORIGIN_DIR, fname), fold_dir)

    return fold_dir

def run_cgcnn(data_dir, train_size, val_size, fold_dir):
    """训练 CGCNN，并将工作目录切换到折目录以保存输出"""
    # 关键修改：使用绝对路径指定 main.py 和数据目录
    cmd = ["python", CGCNN_MAIN_ABS,
           "--train-size", str(train_size),
           "--val-size", str(val_size),
           os.path.abspath(data_dir)]   # 数据目录也转为绝对路径
    if not USE_CUDA:
        cmd.append("--disable-cuda")
    cmd.extend(EXTRA_ARGS)
    print("Running command:", " ".join(cmd))
    # 在折目录内运行，模型文件和日志会保存在该目录下
    subprocess.run(cmd, check=True, cwd=fold_dir)

def run_predict_for_val(fold_dir, val_ids, df, result_list):
    """对验证集进行预测，将结果追加到 result_list 中"""
    model_path = os.path.join(fold_dir, "model_best.pth.tar")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

    pred_out = os.path.join(fold_dir, "val_pred.csv")

    # 根据 predict.py 的实际参数格式构建命令
    cmd = ["python", CGCNN_PREDICT_ABS,
           os.path.abspath(model_path),      # 位置参数 1：模型文件
           os.path.abspath(fold_dir),        # 位置参数 2：数据目录
           "--csv-output", os.path.abspath(pred_out)]  # 指定输出文件
    # 如果 predict.py 支持 --device cpu 或其他 GPU 相关参数，可在此添加
    # 例如：if not USE_CUDA: cmd.extend(["--device", "cpu"])
    print("Predict command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # 读取预测结果（假设输出 CSV 无表头，三列为 id, target, prediction）
    pred_df = pd.read_csv(pred_out, header=None, names=["id", "target", "prediction"])
    # 过滤只保留当前验证集样本
    val_ids_set = set(df.iloc[val_ids]["id"].values)
    val_pred = pred_df[pred_df["id"].isin(val_ids_set)].copy()
    result_list.append(val_pred)

def patch_model_checkpoint(fold_dir):
    """修补模型参数以兼容 predict.py 的命名"""
    model_path = os.path.join(fold_dir, "model_best.pth.tar")
    checkpoint = torch.load(model_path, map_location='cpu')
    args = checkpoint['args']
    if hasattr(args, 'atom_fea_len') and not hasattr(args, 'orig_atom_fea_len'):
        args.orig_atom_fea_len = args.atom_fea_len
    torch.save(checkpoint, model_path)
    print(f"模型已修补: 添加 orig_atom_fea_len = {args.atom_fea_len}")

def main():
    # 1. 读取总表
    csv_path = os.path.join(ORIGIN_DIR, "id_prop.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到总表文件: {csv_path}")
    df = pd.read_csv(csv_path, header=None, names=["id", "target"])
    print(f"总样本数: {len(df)}")

    # 2. 初始化 KFold
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # 3. 创建临时工作目录
    if os.path.exists(WORK_BASE):
        shutil.rmtree(WORK_BASE)
    os.makedirs(WORK_BASE)

    all_results = []   # 收集所有折的预测结果

    # 4. 循环每一折
    for fold, (train_idx, val_idx) in enumerate(kf.split(df), start=1):
        print(f"\n{'='*50}")
        print(f"开始第 {fold} 折训练")
        print(f"训练集大小: {len(train_idx)}, 验证集大小: {len(val_idx)}")
        print(f"{'='*50}")

        fold_dir = prepare_fold_data(fold, train_idx, val_idx, df, WORK_BASE)
        run_cgcnn(fold_dir, len(train_idx), len(val_idx), fold_dir)
        run_predict_for_val(fold_dir, val_idx, df, all_results)

        print(f"第 {fold} 折训练及预测完成\n")

    # 5. 汇总所有折的预测结果
    final_df = pd.concat(all_results, ignore_index=True)
    final_df = final_df.sort_values("id")
    final_df.to_csv("test_results.csv", index=False)
    print("所有折训练完毕！汇总预测结果已保存至: test_results.csv")
    print(f"临时数据保存在: {WORK_BASE}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(df), start=1):
        # ...
        run_cgcnn(fold_dir, len(train_idx), len(val_idx), fold_dir)
        patch_model_checkpoint(fold_dir)  # 新增修补步骤
        run_predict_for_val(fold_dir, val_idx, df, all_results)

if __name__ == "__main__":
    main()