from __future__ import print_function, division

import csv
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from cgcnn.data import CIFData, collate_pool
from cgcnn.model import CrystalGraphConvNet

def main():
    parser = argparse.ArgumentParser(description='CGCNN prediction')
    parser.add_argument('model_path', help='path to the trained model file')
    parser.add_argument('data_path', help='path to the dataset directory')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size for testing')
    parser.add_argument('--num-workers', type=int, default=0, help='number of workers for data loading')
    parser.add_argument('--device', help='device to use for computation')
    parser.add_argument('--csv-output', default='predictions.csv',
                        help='path to save predictions as CSV (default: predictions.csv)')
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据集
    test_dataset = CIFData(args.data_path)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_pool)

    # 加载模型
    checkpoint = torch.load(args.model_path, map_location=device)
    model_args = argparse.Namespace(**checkpoint['args'])
    model = CrystalGraphConvNet(model_args.orig_atom_fea_len, model_args.nbr_fea_len,
                                atom_fea_len=model_args.atom_fea_len, n_conv=model_args.n_conv,
                                h_fea_len=model_args.h_fea_len, n_h=model_args.n_h,
                                classification=False).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    test_ids = []
    test_targets = []
    test_preds = []

    # 进行预测
    for i, (atom_fea, nbr_fea, nbr_fea_idx, target, batch_cif_ids) in enumerate(test_loader):
        atom_fea = atom_fea.to(device)
        nbr_fea = nbr_fea.to(device)
        nbr_fea_idx = nbr_fea_idx.to(device)

        with torch.no_grad():
            pred, _ = model(atom_fea, nbr_fea, nbr_fea_idx)
            test_preds.extend(pred.cpu().numpy().flatten())
            test_targets.extend(target.numpy().flatten())
            test_ids.extend(batch_cif_ids)

        # 可选：显示进度
        print(f'Processed batch {i+1}/{len(test_loader)}', end='\r')

    # 计算并打印 MAE
    mae = np.mean(np.abs(np.array(test_targets) - np.array(test_preds)))
    print(f'\nTest MAE: {mae:.6f}')

    # 将预测结果写入 CSV 文件
    with open(args.csv_output, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['cif_id', 'target', 'prediction'])
        for cif_id, target, pred in zip(test_ids, test_targets, test_preds):
            writer.writerow([cif_id, target, pred])

    print(f"Predictions saved to '{args.csv_output}'")

if __name__ == '__main__':
    main()