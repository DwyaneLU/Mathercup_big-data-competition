import os
import pyproj
import netCDF4 as nc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from datetime import datetime, timedelta
import geopandas as gpd
import contextily as ctx
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from xyzservices import TileProvider
from PIL import Image, ImageCms
import io

# 设置 PROJ_LIB 环境变量
proj_lib_path = r'D:\Anaconda\envs\DL\Library\share\proj'
os.environ['PROJ_LIB'] = proj_lib_path

# 或者使用 pyproj.datadir.set_data_dir 方法
# pyproj.datadir.set_data_dir(proj_lib_path)

# 设置绘图字体和正确显示负号
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 读取 NetCDF 文件路径（使用已经提取的文件）
file_path = 'extracted_precipitation_2022.nc'

# 读取经度和纬度数据
dataset = nc.Dataset(file_path)
longitude = dataset.variables['longitude'][:]
latitude = dataset.variables['latitude'][:]
time = dataset.variables['time'][:]

# 将时间单位转换为日期，原时间是 "hours since 1961-01-01 00:00:00"
base_date = datetime(1961, 1, 1)
dates = [base_date + timedelta(hours=int(t)) for t in time]

# 读取降水量数据
precipitation_data = dataset.variables['precipitation'][:]

# 将小于0的降水量数据替换为0
precipitation_data = np.maximum(precipitation_data, 0)

# 显示提取的数据的基本信息
print('提取的数据维度 (时间 x 纬度 x 经度)：', precipitation_data.shape)

# 读取台风路径数据（从上传的 CSV 文件中读取并保留原始列标题）
filename = 'processed_typhoon_data.csv'
typhoon_data = pd.read_csv(filename)

# 将时间格式 YYYY/MM/DD HH:MM 转换为 datetime 格式
typhoon_data['Time'] = pd.to_datetime(typhoon_data['当前台风时间'].astype(str), format='%Y/%m/%d %H:%M')

# 过滤出 2022 年的台风数据
typhoon_data_2022 = typhoon_data[typhoon_data['Time'].dt.year == 2022].copy()

# 使用 .loc 来避免设置副本的警告
typhoon_data_2022.loc[:, 'Time'] = pd.to_datetime(typhoon_data_2022['当前台风时间'].astype(str), format='%Y/%m/%d %H:%M')

# 计算台风登陆后，降水量与台风中心的距离关系
distance_precipitation = []
for _, row in typhoon_data_2022.iterrows():
    typhoon_lat = row['纬度']
    typhoon_lon = row['经度']
    typhoon_time = row['Time']

    # 找到最接近的时间索引
    time_idx = np.argmin([abs((d - typhoon_time).total_seconds()) for d in dates])

    # 计算降水场中每个点与台风中心的距离
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    distances = np.sqrt((lat_grid - typhoon_lat) ** 2 + (lon_grid - typhoon_lon) ** 2)

    # 提取对应时间的降水量数据
    precipitation_at_time = precipitation_data[time_idx, :, :]

    # 将距离和降水量对应起来，计算与台风中心的距离和降水量关系
    distances = distances.flatten()
    precipitation_at_time = precipitation_at_time.flatten()
    valid_indices = ~np.isnan(precipitation_at_time)

    # 确保有效索引与距离数据匹配
    valid_distances = distances[valid_indices]
    valid_precipitation = precipitation_at_time[valid_indices]

    # 保存距离最小的降水量（即台风中心附近的降水量）
    if len(valid_distances) > 0:
        min_idx = np.argmin(valid_distances)
        distance_precipitation.append({
            'Time': typhoon_time,
            'Distance': valid_distances[min_idx],
            'Precipitation': valid_precipitation[min_idx]
        })

# 转换为 DataFrame
distance_precipitation_df = pd.DataFrame(distance_precipitation)

# 将降水量和距离关系写入 Excel 表格
distance_precipitation_df.to_excel('typhoon_distance_precipitation_2022.xlsx', index=False)

# 使用待定系数公式拟合降水量数据
P_ref = 1013  # 参考气压，标准大气压 (hPa)

# 提取所需数据
D = distance_precipitation_df['Distance'].values
P_obs = distance_precipitation_df['Precipitation'].values

# 获取其他台风参数
V = typhoon_data_2022['移动速度'].values  # 假设单位为 km/h
P_c = typhoon_data_2022['气压'].values  # 修正为表中的列名 '气压'
V_max = typhoon_data_2022['风速'].values  # 修正为表中的列名 '风速'

# 删除无效数据（如 D, V, P_c, V_max, P_obs 中包含 NaN 或 Inf 的值）
valid_indices = ~(np.isnan(D) | np.isnan(V) | np.isnan(P_c) | np.isnan(V_max) | np.isnan(P_obs) |
                  np.isinf(D) | np.isinf(V) | np.isinf(P_c) | np.isinf(V_max) | np.isinf(P_obs))

# 输出各个条件下被筛除的数据数量
print(f'总数据点数: {len(D)}')
print(f'NaN 或 Inf 值数量（D, V, P_c, V_max, P_obs）: {np.sum(~valid_indices)}')
print(f'有效数据点数量: {np.sum(valid_indices)}')

# 确保至少有一些有效数据点
if np.sum(valid_indices) < 10:
    raise ValueError('有效数据点不足，无法进行拟合，请检查数据预处理步骤。')

D = D[valid_indices]
V = V[valid_indices]
P_c = P_c[valid_indices]
V_max = V_max[valid_indices]
P_obs = P_obs[valid_indices]

# 对数据进行标准化以提高拟合的稳定性
D_norm = (D - np.mean(D)) / np.std(D)
V_norm = (V - np.mean(V)) / np.std(V)
P_c_norm = (P_c - np.mean(P_c)) / np.std(P_c)
V_max_norm = (V_max - np.mean(V_max)) / np.std(V_max)

# 定义待定系数拟合的目标函数
def model_fun(params, X):
    A, alpha, beta = params
    D, V, P_c, V_max = X.T
    return A * np.exp(-alpha * D) * (1 / (V + np.finfo(float).eps)) * (1 / np.maximum(P_c - P_ref, 1)) * (np.abs(V_max) ** beta)

# 定义误差函数
def residuals(params, X, y):
    return model_fun(params, X) - y

# 初始猜测参数 A, alpha, beta
initial_params = [0.5, 0.5, 0.5]  # 修改初始参数，使其不至于过小

# 定义拟合数据矩阵
X_data = np.vstack((D_norm, V_norm, P_c_norm, V_max_norm)).T

# 确保残差函数初始值是有限的
y_initial = model_fun(initial_params, X_data)
if not np.all(np.isfinite(y_initial)):
    raise ValueError('Initial residuals are not finite, please adjust initial parameters or check input data.')

# 进行拟合
result = least_squares(residuals, initial_params, bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]), args=(X_data, P_obs))

# 检查拟合是否成功
if not result.success:
    print('拟合未能成功收敛，请检查数据或尝试不同的初始参数。')

# 显示拟合结果
b_est = result.x
print('拟合的参数 (A, alpha, beta)：', b_est)

# 使用拟合的模型绘制拟合曲线和观测数据的对比
P_fit = model_fun(b_est, X_data)

plt.figure(figsize=(10, 6))
plt.scatter(D, P_obs, color='b', label='Observed Precipitation')
plt.plot(D, P_fit, color='r', linewidth=2, label='Fitted Model')
plt.xlabel('Distance to Typhoon Center (km)')
plt.ylabel('Precipitation (mm)')
plt.title('Observed vs Fitted Precipitation vs Distance to Typhoon Center')
plt.legend()
plt.grid(True)
plt.show()

# 将拟合结果保存到 CSV
fitted_results = pd.DataFrame({'Distance': D, 'Observed_Precipitation': P_obs, 'Fitted_Precipitation': P_fit})
fitted_results.to_csv('fitted_typhoon_precipitation_2022.csv', index=False)

# 创建 GeoDataFrame 用于地图绘制
# 确保两个 DataFrame 有共同的时间索引
distance_precipitation_df['Time'] = pd.to_datetime(distance_precipitation_df['Time'])
typhoon_data_2022['Time'] = pd.to_datetime(typhoon_data_2022['Time'])

# 合并数据
merged_data = pd.merge(typhoon_data_2022, distance_precipitation_df[['Time', 'Precipitation']], on='Time')

# 创建 GeoDataFrame
gdf = gpd.GeoDataFrame(
    merged_data,
    geometry=gpd.points_from_xy(merged_data['经度'], merged_data['纬度']),
    crs="EPSG:4326"  # WGS84坐标系
)

# 自定义请求会话，增加重试次数
session = requests.Session()
retry = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

# 创建自定义瓦片提供者
class CustomTileProvider(TileProvider):
    def __init__(self, session):
        super().__init__(
            name='OpenStreetMap',
            url='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        )
        self.session = session

    def get_image(self, extent, zoom):
        url = self.build_url(extent, zoom)
        response = self.session.get(url)
        if response.status_code == 200:
            # 使用 PIL 处理图像并忽略颜色配置文件
            image = Image.open(io.BytesIO(response.content))
            image = ImageCms.profileToProfile(image, ImageCms.createProfile('sRGB'), ImageCms.createProfile('sRGB'))
            return ctx.utils.imread(np.array(image))
        else:
            raise Exception(f"Failed to download tile from {url}")

# 使用自定义瓦片提供者添加底图
fig, ax = plt.subplots(figsize=(12, 8))
gdf.plot(ax=ax, column='风速', cmap='viridis', markersize=50, legend=True, legend_kwds={'label': 'Wind Speed (m/s)'})
ctx.add_basemap(ax, crs=gdf.crs, source=CustomTileProvider(session), attribution=None)  # 使用 OpenStreetMap 作为底图
ax.set_title('Typhoon Path with Wind Speed (2022)')
plt.show()

# 绘制台风路径及降水量图
fig, ax = plt.subplots(figsize=(12, 8))
gdf.plot(ax=ax, column='Precipitation', cmap='Blues', markersize=50, legend=True, legend_kwds={'label': 'Precipitation (mm)'})
ctx.add_basemap(ax, crs=gdf.crs, source=CustomTileProvider(session), attribution=None)  # 使用 OpenStreetMap 作为底图
ax.set_title('Typhoon Path with Precipitation (2022)')
plt.show()

# 绘制某一天的降水量分布
selected_day_index = 0  # 选择2022年的第一天
precipitation_at_day = precipitation_data[selected_day_index, :, :]

plt.figure(figsize=(10, 6))
plt.imshow(precipitation_at_day, extent=(longitude.min(), longitude.max(), latitude.min(), latitude.max()), origin='lower', aspect='auto')
plt.colorbar(label='Precipitation (mm)')
plt.title('Daily Precipitation on 2022-01-01')
plt.xlabel('Longitude (degrees)')
plt.ylabel('Latitude (degrees)')
plt.grid(True)
plt.show()

# 绘制台风路径及降水量分布图
fig, ax = plt.subplots(figsize=(12, 8))
cs = ax.contourf(longitude, latitude, precipitation_at_day, cmap='Blues')
gdf.plot(ax=ax, column='风速', cmap='viridis', markersize=50, legend=True, legend_kwds={'label': 'Wind Speed (m/s)'})
ctx.add_basemap(ax, crs=gdf.crs, source=CustomTileProvider(session), attribution=None)  # 使用 OpenStreetMap 作为底图
cbar = fig.colorbar(cs, ax=ax)
cbar.set_label('Precipitation (mm)')
ax.set_title('Typhoon Path with Precipitation and Wind Speed (2022)')
plt.show()