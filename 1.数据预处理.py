import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 读取数据文件（确保路径正确）
file_path = '1945-2023.csv'  # 请将此路径替换为您的数据文件路径
typhoon_data = pd.read_csv(file_path, encoding='UTF-8')

# 修正列名
column_names = [
    '台风编号','台风中文名称','台风英文名称','台风起始时间','台风结束时间',
    '当前台风时间','经度','纬度','台风强度','台风等级','风速','气压','移动方向','移动速度'
]
typhoon_data.columns = column_names

# 1. 缺失值处理
# 删除气压、台风等级、台风强度中存在缺失值的行
typhoon_data.dropna(subset=['气压', '台风等级', '台风强度'], inplace=True)

# 处理时间列的格式
# 将“当前台风时间”列转换为 datetime 类型，并尝试指定自定义格式
try:
    typhoon_data['当前台风时间'] = pd.to_datetime(typhoon_data['当前台风时间'], format='%Y%m%d%H:%M:%S', errors='coerce')
except ValueError:
    typhoon_data['当前台风时间'] = pd.to_datetime(typhoon_data['当前台风时间'], errors='coerce')

# 删除无法解析的时间行
typhoon_data.dropna(subset=['当前台风时间'], inplace=True)

# 2. 移动方向计算（基于同一台风编号）
for i in range(len(typhoon_data) - 1):
    if typhoon_data.at[i, '台风编号'] == typhoon_data.at[i + 1, '台风编号']:
        lat1, lon1 = typhoon_data.at[i, '纬度'], typhoon_data.at[i, '经度']
        lat2, lon2 = typhoon_data.at[i + 1, '纬度'], typhoon_data.at[i + 1, '经度']
        delta_lon = lon2 - lon1
        y = math.sin(math.radians(delta_lon)) * math.cos(math.radians(lat2))
        x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(delta_lon))
        angle = math.degrees(math.atan2(y, x))
        if angle < 0:
            angle += 360
        if pd.isna(data.at[i, '移动方向']):
            data.at[i, '移动方向'] = angle
    else:
        data.at[i, '移动方向'] = np.nan

# 最后一行的移动方向无法计算，填充为 NaN
data.at[len(data) - 1, '移动方向'] = np.nan

# 删除包含缺失值的行
typhoon_data.dropna(inplace=True)

# 标准化数值列
scaler = MinMaxScaler()
typhoon_data[['Longitude', 'Latitude', 'Wind_Speed', 'Pressure', 'Grade']] = scaler.fit_transform(
    typhoon_data[['Longitude', 'Latitude', 'Wind_Speed', 'Pressure', 'Grade']]
)

# 保存处理后的数据为新的CSV文件
output_file_path = 'C:\\Users\Lenovo\PycharmProjects\luodi\mathorcup\proecssed_data.csv'  # 请将此路径替换为您的存储路径
typhoon_data.to_csv(output_file_path, index=False)

print(f"缺失值已删除，处理后的数据已保存到 {output_file_path}")