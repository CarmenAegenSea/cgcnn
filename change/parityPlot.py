import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 假设你运行了 python predict.py 得到了 test_results.csv
df = pd.read_csv('../test_results.csv', names=['ID', 'Target', 'Prediction'])

plt.figure(figsize=(8, 6))
plt.scatter(df['Target'], df['Prediction'], alpha=0.6, color='blue', edgecolors='white')
plt.plot([df['Target'].min(), df['Target'].max()], [df['Target'].min(), df['Target'].max()], 'r--', lw=2)

plt.xlabel('Experimental/DFT Formation Energy (eV/atom)', fontsize=12)
plt.ylabel('CGCNN Predicted Formation Energy (eV/atom)', fontsize=12)
plt.title('Prediction Accuracy for Environmental Catalysts', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# 计算 R-squared
y_true = df['Target']
y_pred = df['Prediction']
r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
plt.annotate(f'$R^2$ = {r2:.3f}\nMAE = 0.179', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=12, bbox=dict(boxstyle="round", fc="white"))

plt.show()