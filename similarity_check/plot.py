import pandas as pd
import matplotlib.pyplot as plt


data_path = "C:/Users/Owner/Desktop/plot2.xlsx"
sheet_name = 'Sheet3'  # 替換成您想要讀取的工作表名

# 加載Excel文件
data = pd.read_excel(data_path, sheet_name=sheet_name)

# 設置行名作為x軸標籤
data.set_index('Unnamed: 0', inplace=True)

# 創建圖表
fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(18, 6))

data[['1-PRED', '1-COT']].plot(kind='bar', ax=axes[0], color=['skyblue', 'orange'])
axes[0].set_title('1-PRED vs 1-COT')
axes[0].set_xlabel('Metrics')
axes[0].set_ylabel('Values')
axes[0].legend(title='1-Experiment')
axes[0].grid(True, linestyle='--', alpha=0.6)

# 繪製'2-GEN'和'2-COT'
data[['2-PRED', '2-COT']].plot(kind='bar', ax=axes[1], color=['skyblue', 'orange'])
axes[1].set_title('2-PRED vs 2-COT')
axes[1].set_xlabel('Metrics')
axes[1].set_ylabel('Values')
axes[1].legend(title='2-Experiment')
axes[1].grid(True, linestyle='--', alpha=0.6)

# 繪製'3-GEN'和'3-COT'
data[['3-PRED', '3-COT']].plot(kind='bar', ax=axes[2], color=['green', 'red'])
axes[2].set_title('3-PRED vs 3-COT')
axes[2].set_xlabel('Metrics')
axes[2].set_ylabel('Values')
axes[2].legend(title='3-Experiment')
axes[2].grid(True, linestyle='--', alpha=0.6)


# 繪製'5-GEN'和'5-COT'
data[['5-PRED', '5-COT']].plot(kind='bar', ax=axes[3], color=['purple', 'yellow'])
axes[3].set_title('5-PRED vs 5-COT')
axes[3].set_xlabel('Metrics')
axes[3].set_ylabel('Values')
axes[3].legend(title='5-Experiment')
axes[3].grid(True, linestyle='--', alpha=0.6)

data[['6-PRED', '6-COT']].plot(kind='bar', ax=axes[4], color=['purple', 'yellow'])
axes[4].set_title('6-PRED vs 6-COT')
axes[4].set_xlabel('Metrics')
axes[4].set_ylabel('Values')
axes[4].legend(title='6-Experiment')
axes[4].grid(True, linestyle='--', alpha=0.6)

data[['8-PRED', '8-COT']].plot(kind='bar', ax=axes[5], color=['purple', 'yellow'])
axes[5].set_title('8-PRED vs 8-COT')
axes[5].set_xlabel('Metrics')
axes[5].set_ylabel('Values')
axes[5].legend(title='8-Experiment')
axes[5].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
