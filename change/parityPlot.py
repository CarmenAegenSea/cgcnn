import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#===========================================
# 最终数据处理 需要test_results.csv(predict.py)
#===========================================

# 读取预测结果（假设无表头）
df = pd.read_csv('test_results.csv', names=['ID', 'Target', 'Prediction'])

# 计算评估指标
y_true = df['Target']
y_pred = df['Prediction']
mae = np.mean(np.abs(y_true - y_pred))
rmse = np.sqrt(np.mean((y_true - y_pred)**2))
r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))

print(f"MAE  = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R²   = {r2:.4f}")

# 绘图
plt.figure(figsize=(8, 6))
plt.scatter(y_true, y_pred, alpha=0.6, color='blue', edgecolors='white')
min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

plt.xlabel('Experimental/DFT Formation Energy (eV/atom)', fontsize=12)
plt.ylabel('CGCNN Predicted Formation Energy (eV/atom)', fontsize=12)
plt.title('Prediction Accuracy for Environmental Catalysts', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# 在图中添加文本
plt.annotate(f'$R^2$ = {r2:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}',
             xy=(0.05, 0.85), xycoords='axes fraction', fontsize=12,
             bbox=dict(boxstyle="round", fc="white"))

plt.tight_layout()
plt.savefig('prediction_plot.png', dpi=300)
plt.show()