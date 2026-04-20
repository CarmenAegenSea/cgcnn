"""
CGCNN 预测脚本 - 自动读取标准化参数，失败则使用手动指定值
"""

import os
import csv
import torch
import argparse
from cgcnn.data import CIFData, collate_pool
from cgcnn.model import CrystalGraphConvNet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('modelpath')
    parser.add_argument('datapath')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--disable-cuda', action='store_true')
    args = parser.parse_args()

    checkpoint = torch.load(args.modelpath, map_location='cpu')
    model_args = checkpoint.get('model_args') or checkpoint.get('args')

    #从 checkpoint 读取标准化参数
    normalizer = checkpoint.get('normalizer', {})
    mean = normalizer.get('mean')
    std = normalizer.get('std')

    if mean is None or std is None:
        print("Checkpoint 中未找到 normalizer，使用手动指定值。")
        MANUAL_MEAN = 1.5972 #替换为获取的mean
        MANUAL_STD = 1.2327   #替换为获取的std
        mean = MANUAL_MEAN
        std = MANUAL_STD
    else:
        print(f"从 checkpoint 读取: mean = {mean:.4f}, std = {std:.4f}")

    dataset = CIFData(args.datapath)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pool)

    model = CrystalGraphConvNet(
        orig_atom_fea_len=getattr(model_args, 'orig_atom_fea_len', 92),
        nbr_fea_len=getattr(model_args, 'nbr_fea_len', 41),
        atom_fea_len=getattr(model_args, 'atom_fea_len', 64),
        n_conv=getattr(model_args, 'n_conv', 3),
        h_fea_len=getattr(model_args, 'h_fea_len', 128),
        n_h=getattr(model_args, 'n_h', 1),
        classification=getattr(model_args, 'classification', False)
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    use_cuda = not args.disable_cuda and torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    predictions = []
    cif_ids = []
    with torch.no_grad():
        for batch in data_loader:
            inputs, _, batch_ids = batch
            atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs
            if use_cuda:
                atom_fea = atom_fea.cuda()
                nbr_fea = nbr_fea.cuda()
                nbr_fea_idx = nbr_fea_idx.cuda()
                crystal_atom_idx = [idx.cuda() for idx in crystal_atom_idx]
            output = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
            predictions.extend(output.cpu().numpy().flatten().tolist())
            cif_ids.extend(batch_ids)

    #逆标准化
    predictions = [p * std + mean for p in predictions]

    output_file = 'test_results_final.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'predicted_bandgap_eV'])
        for mid, pred in zip(cif_ids, predictions):
            writer.writerow([mid, f"{pred:.4f}"])

    print(f"预测完成，结果保存至 {output_file}")

if __name__ == '__main__':
    main()