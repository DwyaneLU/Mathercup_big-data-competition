import pandas as pd
import numpy as np
import math

# 读取Excel文件并保留原始列标题
filename = '1945-2023.xlsx'
data = pd.read_excel(filename)

# 重置索引，确保索引是从0到N-1连续的整数
data.reset_index(drop=True, inplace=True)

# 1. 缺失值处理
# 删除气压、台风等级、台风强度中存在缺失值的行
data.dropna(subset=['气压', '台风等级', '台风强度'], inplace=True)

# 重置索引，确保删除缺失值后索引是从0到N-1连续的整数
data.reset_index(drop=True, inplace=True)

# 处理时间列的格式
# 将“当前台风时间”列转换为 datetime 类型
try:
    data['当前台风时间'] = pd.to_datetime(data['当前台风时间'], format='%Y%m%d%H:%M:%S', errors='coerce')
except ValueError:
    data['当前台风时间'] = pd.to_datetime(data['当前台风时间'], errors='coerce')

# 删除无法解析的时间行
data.dropna(subset=['当前台风时间'], inplace=True)

# 重置索引，确保删除无效时间数据后索引是从0到N-1连续的整数
data.reset_index(drop=True, inplace=True)

# 2. 移动方向计算（基于同一台风编号）
for i in range(len(data) - 1):
    if data.at[i, '台风编号'] == data.at[i + 1, '台风编号']:
        lat1, lon1 = data.at[i, '纬度'], data.at[i, '经度']
        lat2, lon2 = data.at[i + 1, '纬度'], data.at[i + 1, '经度']
        delta_lon = lon2 - lon1
        y = math.sin(math.radians(delta_lon)) * math.cos(math.radians(lat2))
        x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(delta_lon))
        angle = math.degrees(math.atan2(y, x))
        if angle < 0:
            angle += 360
        data.at[i, '移动方向'] = angle
    else:
        data.at[i, '移动方向'] = np.nan

# 最后一行的移动方向无法计算，填充为 NaN
data.at[len(data) - 1, '移动方向'] = np.nan

# 3. 移动速度计算（基于同一台风编号）
radius_earth = 6371000  # 地球半径，单位为米
for i in range(len(data) - 1):
    if data.at[i, '台风编号'] == data.at[i + 1, '台风编号']:
        lat1, lon1 = math.radians(data.at[i, '纬度']), math.radians(data.at[i, '经度'])
        lat2, lon2 = math.radians(data.at[i + 1, '纬度']), math.radians(data.at[i + 1, '经度'])
        delta_lat = lat2 - lat1
        delta_lon = lon2 - lon1
        a = math.sin(delta_lat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = radius_earth * c  # 两个点之间的距离，单位为米
        # 确保时间列是datetime类型，并计算时间差
        time1 = data.at[i, '当前台风时间']
        time2 = data.at[i + 1, '当前台风时间']
        time_diff = (time2 - time1).total_seconds()  # 时间差，单位为秒
        if time_diff > 0:
            data.at[i, '移动速度'] = distance / time_diff  # 移动速度，单位为m/s
        else:
            data.at[i, '移动速度'] = np.nan
    else:
        data.at[i, '移动速度'] = np.nan

# 最后一行的移动速度无法计算，填充为 NaN
data.at[len(data) - 1, '移动速度'] = np.nan

# 4. 异常值处理
# 处理极端值，例如风速高于110或气压低于50
extremeWindSpeed = 110
lowPressure = 50

# 将风速超过110或气压低于50的行标记为NaN，进行进一步处理
data.loc[data['风速'] > extremeWindSpeed, '风速'] = np.nan
data.loc[data['气压'] < lowPressure, '气压'] = np.nan

# 对异常值进行插值处理
data = data.assign(
    风速=data['风速'].interpolate(method='linear'),
    气压=data['气压'].interpolate(method='linear')
)

# 5. 删除所有包含 NaN 的行
data.dropna(inplace=True)

# 重置索引，确保索引是从0到N-1连续的整数
data.reset_index(drop=True, inplace=True)

# 处理结果保存
# 将处理后的数据保存为新的Excel文件
data.to_csv('processed_data1.csv', index=False)
