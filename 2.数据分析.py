import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import mpl
import geopandas as gpd
mpl.rcParams['font.sans-serif'] = ['SimHei']
# 读取预处理后的数据文件
file_path = 'proecssed_data.csv'  # 请将此路径替换为您的数据文件路径
typhoon_data = pd.read_csv(file_path)

# 1. 总体统计信息
print("数据集描述性统计信息：")
print(typhoon_data.describe())

# 2. 特征分布分析

# 风速分布
plt.figure(figsize=(10, 6))
sns.histplot(typhoon_data['Wind_Speed'], bins=30, kde=True, color='skyblue')
plt.title("风速分布")
plt.xlabel("风速 (标准化)")
plt.ylabel("频数")
plt.show()

# 气压分布
plt.figure(figsize=(10, 6))
sns.histplot(typhoon_data['Pressure'], bins=30, kde=True, color='salmon')
plt.title("气压分布")
plt.xlabel("气压 (标准化)")
plt.ylabel("频数")
plt.show()

# 等级分布
plt.figure(figsize=(10, 6))
sns.histplot(typhoon_data['Grade'], bins=30, kde=True, color='lightgreen')
plt.title("等级分布")
plt.xlabel("等级 (标准化)")
plt.ylabel("频数")
plt.show()

# 3. 时间特征分析

# 台风月份分布
plt.figure(figsize=(12, 6))
sns.countplot(x='Month', data=typhoon_data, palette='coolwarm')
plt.title("台风月份分布")
plt.xlabel("月份")
plt.ylabel("台风数量")
plt.show()

# 台风年份分布
plt.figure(figsize=(12, 6))
sns.countplot(x='Year', data=typhoon_data, palette='viridis')
plt.xticks(rotation=90)
plt.title("台风年份分布")
plt.xlabel("年份")
plt.ylabel("台风数量")
plt.show()

# 4. 空间特征分析

# 经纬度分布
plt.figure(figsize=(10, 8))
plt.scatter(typhoon_data['Longitude'], typhoon_data['Latitude'], alpha=0.5, s=10, c='blue')
plt.title("台风经纬度分布")
plt.xlabel("经度")
plt.ylabel("纬度")
plt.grid(True)
plt.show()

# 5. 台风路径示例 (选取特定台风的轨迹)
# 示例：根据 Typhoon_ID 绘制某个台风的路径
typhoon_id = typhoon_data['Typhoon_ID'].iloc[0]  # 示例选择第一个台风ID
selected_typhoon = typhoon_data[typhoon_data['Typhoon_ID'] == typhoon_id]

plt.figure(figsize=(10, 8))
plt.plot(selected_typhoon['Longitude'], selected_typhoon['Latitude'], marker='o', color='orange')
plt.title(f"台风 {typhoon_id} 的路径")
plt.xlabel("经度")
plt.ylabel("纬度")
plt.grid(True)
plt.show()

# 6 台风位置地理图
# 创建一个 GeoDataFrame
gdf = gpd.GeoDataFrame(
    typhoon_data,
    geometry=gpd.points_from_xy(typhoon_data['经度'], typhoon_data['纬度']),
    crs="EPSG:4326"  # 使用 WGS 84 坐标系统
)

# 加载世界地图数据 (请将此路径替换为解压后的 .shp 文件路径)
world = gpd.read_file('ne_110m_admin_0_countries.shp')

# 创建绘图
plt.figure(figsize=(15, 10))
ax = world.plot(color='lightgrey', edgecolor='black')

# 在地图上绘制所有台风的位置
gdf.plot(ax=ax, marker='o', color='red', markersize=5, alpha=0.6, label='Typhoon Locations')

plt.title("台风经纬度分布")
plt.xlabel("经度")
plt.ylabel("纬度")
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.grid(True)
plt.legend()
plt.show()