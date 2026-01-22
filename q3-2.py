import netCDF4 as nc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from datetime import datetime, timedelta  # 确保导入 datetime 模块
import geopandas as gpd
import contextily as ctx
import os  # 导入 os 模块

# 设置绘图字体和正确显示负号
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 设置 PROJ_LIB 环境变量
proj_lib_path = r'D:\Anaconda\envs\DL\Library\share\proj'
os.environ['PROJ_LIB'] = proj_lib_path

# 读取贝碧嘉台风数据
filename = '问题三预测数据.xlsx'
typhoon_data_bebinca = pd.read_excel(filename)

# 将时间格式 YYYYMMDDHH 转换为 datetime 格式
typhoon_data_bebinca['Time'] = pd.to_datetime(typhoon_data_bebinca['当前台风时间'].astype(str), format='%Y%m%d%H')

# 过滤出 2024 年 9 月 16 日至 18 日的数据
date_filter = (typhoon_data_bebinca['Time'] >= datetime(2024, 9, 16)) & (typhoon_data_bebinca['Time'] <= datetime(2024, 9, 18))
typhoon_data_filtered = typhoon_data_bebinca[date_filter]

# 提取数据
latitudes = typhoon_data_filtered['纬度'].values
longitudes = typhoon_data_filtered['经度'].values
P_c = typhoon_data_filtered['气压'].values  # 单位为 hPa
V_max = typhoon_data_filtered['风速'].values  # 最大风速 (单位：m/s)
lat = typhoon_data_filtered['纬度'].values  # 纬度

# 定义空气密度 rho 和环境气压 P_0
rho = 1.15  # kg/m^3
P_0 = 1013  # 标准大气压 hPa

# 计算最大风速半径 R_m 和 Holland 参数 B
dp = 1010 - P_c
ln_Rmax = 3.015 - 6.291e-5 * (dp ** 2) + 0.0337 * lat
R_m = np.exp(ln_Rmax)  # 最大风速半径 (单位：km)
B = 1.4 + 0.01 * dp - 0.0026 * R_m

# 使用 Holland 公式计算中心风速
V_max_calc = np.sqrt((B / rho) * ((P_0 - P_c) * 100) * np.exp(1))  # 将气压差转换为 Pa

# 使用之前拟合的降水公式计算降水量
# 拟合的参数: A, alpha, beta
A = 0.0070
alpha = 0.0106
beta = 0.9089

# 计算距离台风中心的降水量（假设某固定距离，例如 R_m）
D = R_m  # 使用最大风速半径作为距离
V = V_max_calc  # 使用 Holland 公式计算得到的风速
P_ref = 1013  # 标准大气压 hPa
P_obs = A * np.exp(-alpha * D) * (1 / V) * (1 / (P_c - P_ref)) * (V_max ** beta)

# 创建一个新的 DataFrame 来存放计算结果
result_data = typhoon_data_filtered.copy()
result_data['Calculated_Wind_Speed'] = V_max_calc
result_data['Predicted_Precipitation'] = P_obs

# 保存新的 DataFrame 到 CSV 文件
output_filename = 'typhoon_bebinca_prediction_results.csv'
result_data.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f'New data has been saved to {output_filename}')

# 绘制结果可视化
plt.figure(figsize=(10, 10))

plt.subplot(2, 1, 1)
plt.plot(typhoon_data_filtered['Time'], V_max, 'r', linewidth=2, label='Observed Wind Speed')
plt.plot(typhoon_data_filtered['Time'], V_max_calc, 'b--', linewidth=2, label='Calculated Wind Speed')
plt.xlabel('Time')
plt.ylabel('Wind Speed (m/s)')
plt.title('Typhoon Bebinca Central Wind Speed (2024-09-16 to 2024-09-18)')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(typhoon_data_filtered['Time'], P_obs, 'b', linewidth=2, label='Predicted Precipitation')
plt.xlabel('Time')
plt.ylabel('Precipitation (mm)')
plt.title('Predicted Precipitation for Typhoon Bebinca (2024-09-16 to 2024-09-18)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# 创建 GeoDataFrame 用于地图绘制
gdf = gpd.GeoDataFrame(
    typhoon_data_filtered,
    geometry=gpd.points_from_xy(typhoon_data_filtered['经度'], typhoon_data_filtered['纬度']),
    crs="EPSG:4326"  # WGS84坐标系
)
# 地图显示台风路径
fig, ax = plt.subplots(figsize=(10, 6))
gdf.plot(ax=ax, color='blue', markersize=50, label='Typhoon Path', alpha=0.7)
ctx.add_basemap(ax, crs=gdf.crs, source=ctx.providers.OpenStreetMap.Mapnik)  # 添加 OpenStreetMap 作为底图
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Typhoon Bebinca Path (2024-09-16 to 2024-09-18)')
ax.grid(True)
ax.legend()
plt.show()