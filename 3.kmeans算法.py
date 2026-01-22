import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 加载数据
data = pd.read_csv('proecssed_data.csv')  # 假设文件名为 proecssed_data.csv

# 检查数据
print("Data columns:", data.columns)

# 提取特征
# 假设数据集中包含 'wind_Speed', 'Pressure', 'Moving_Speed' 这样的列
features = ['wind_Speed', 'Pressure', 'Moving_Speed']
X = data[[f for f in features if f in data.columns]]

# 标准化处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 确定K值
# 假定已经通过肘部法则确定了K=3
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=0)

# 模型拟合
kmeans.fit(X_scaled)

# 预测标签
labels = kmeans.predict(X_scaled)

# 将聚类标签添加回原始数据
data['cluster'] = labels

# 输出结果
print(data[['Chinese_Name', 'Wind_Speed', 'Pressure', 'Movement_Speed', 'cluster']])

# 绘制时间序列图
plt.figure(figsize=(14, 8))

# 定义类别颜色
colors = {0: 'blue', 1: 'green', 2: 'red'}

# 对每个台风绘制时间序列图
for name, group in data.groupby('Chinese_Name'):
    plt.plot(group['Current_Time'], group['cluster'], marker='o', linestyle='-', color=colors[group['cluster'].iloc[0]], label=f"{name} ({group['cluster'].iloc[0]})")

# 设置图例和标签
plt.xlabel('Current_Time')
plt.ylabel('Cluster')
plt.title('Typhoon Classification Over Time')
plt.legend(title='Typhoon Name and Cluster')

# 显示图表
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()