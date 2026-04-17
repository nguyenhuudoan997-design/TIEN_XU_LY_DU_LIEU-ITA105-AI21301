import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels
from statsmodels.tsa.seasonal import seasonal_decompose
# BÀI1:
df = pd.read_csv("ITA105_Lab_5_Supermarket.csv")

df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace = True)

print("Số lượng giá trị thiếu trước khi xử lý:", df['revenue'].isnull().sum())

df['revenue'] = df['revenue'].interpolate(method='linear')
df['revenue'] = df['revenue'].ffill().bfill()
df['year'] = df.index.year
df['month'] = df.index.month
df['quarter'] = df.index.quarter
df['day_of_week'] = df.index.dayofweek
df['weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
df.resample('ME')['revenue'].sum().plot(kind='line', marker='o', color='blue')
plt.title('Tổng doanh thu theo Tháng')
plt.ylabel('Doanh thu')
plt.subplot(1, 2, 2)
df.resample('W')['revenue'].sum().plot(kind='line', color='green')
plt.title('Tổng doanh thu theo Tuần')
plt.show()

result = seasonal_decompose(df['revenue'], model='additive', period=30) 

result.plot()
plt.suptitle('Phân rã chuỗi thời gian: Trend, Seasonality, Residual', fontsize=10)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(df['revenue'], alpha=0.3, label='Doanh thu hàng ngày')
plt.plot(df['revenue'].rolling(window=30).mean(), color='red', label='Xu hướng (Rolling Mean 30 ngày)')
plt.title('Xu hướng doanh thu dài hạn')
plt.legend()
plt.show()

# BÀI2:
df = pd.read_csv('ITA105_Lab_5_Web_traffic.csv')

df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

df = df.asfreq('h') 

print("Số lượng giá trị thiếu phát hiện:", df['visits'].isnull().sum())

df['visits'] = df['visits'].interpolate(method='linear')

df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek # 0: Thứ 2, 6: Chủ nhật
df['day_name'] = df.index.day_name()

plt.figure(figsize=(10, 5))

hourly_avg = df.groupby('hour')['visits'].mean()
hourly_avg.plot(kind='bar', color='orange', edgecolor='black')
plt.title('Trung bình lượt truy cập theo từng khung giờ (Peak/Trough)')
plt.xlabel('Giờ trong ngày')
plt.ylabel('Lượt truy cập (Visits)')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

result_daily = seasonal_decompose(df['visits'], model='additive', period=24)

plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
result_daily.seasonal[:72].plot()
plt.title('Tính mùa vụ hàng ngày (Daily Seasonality - 72 giờ đầu)')

result_weekly = seasonal_decompose(df['visits'], model='additive', period=168)

plt.subplot(2, 1, 2)
result_weekly.seasonal[:504].plot()
plt.title('Tính mùa vụ hàng tuần (Weekly Seasonality - 3 tuần đầu)')

plt.tight_layout()
plt.show()

# BÀI3:
df = pd.read_csv('ITA105_Lab_5_Stock.csv')

df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

print("Giá trị thiếu ban đầu:", df['close_price'].isnull().sum())

df['close_price'] = df['close_price'].ffill()

plt.figure(figsize=(10, 5))
plt.plot(df['close_price'], label='Giá đóng cửa (Gốc)', color='gray', alpha=0.5)
plt.title('Biểu đồ giá đóng cửa cổ phiếu')
plt.xlabel('Thời gian')
plt.ylabel('Giá (Close Price)')
plt.legend()
plt.show()

df['SMA_7'] = df['close_price'].rolling(window=7).mean()
df['SMA_30'] = df['close_price'].rolling(window=30).mean()

plt.figure(figsize=(10, 5))
plt.plot(df['close_price'], label='Giá gốc', alpha=0.3)
plt.plot(df['SMA_7'], label='Trend 7 ngày (Ngắn hạn)', color='blue')
plt.plot(df['SMA_30'], label='Trend 30 ngày (Trung hạn)', color='red', linewidth=2)
plt.title('Nhận diện xu hướng bằng Rolling Mean')
plt.legend()
plt.show()

df['month'] = df.index.month
monthly_pattern = df.groupby('month')['close_price'].mean()

plt.figure(figsize=(10, 5))
monthly_pattern.plot(kind='bar', color='teal', edgecolor='black')
plt.title('Mẫu hình biến động giá theo Tháng (Seasonality)')
plt.xlabel('Tháng trong năm')
plt.ylabel('Giá trung bình')
plt.xticks(rotation=0)
plt.show()

result = seasonal_decompose(df['close_price'].resample('D').ffill(), model='additive', period=252)
result.seasonal.plot(figsize=(12, 4))
plt.title('Thành phần Mùa vụ (Seasonality) dài hạn')
plt.show()

# BÀI4:

df = pd.read_csv('ITA105_Lab_5_Production.csv')

df['week_start'] = pd.to_datetime(df['week_start'])
df.set_index('week_start', inplace=True)

print("Số lượng giá trị thiếu:", df['production'].isnull().sum())
df['production'] = df['production'].interpolate(method='linear').ffill().bfill()

df['week'] = df.index.isocalendar().week # Số tuần trong năm
df['quarter'] = df.index.quarter
df['year'] = df.index.year

df['rolling_trend'] = df['production'].rolling(window=12).mean()

plt.figure(figsize=(10, 5))
plt.plot(df['production'], label='Sản lượng hàng tuần', alpha=0.4)
plt.plot(df['rolling_trend'], label='Xu hướng (12 tuần)', color='magenta', linewidth=2)
plt.title('Xu hướng sản xuất công nghiệp dài hạn')
plt.legend()
plt.show()

quarterly_avg = df.groupby('quarter')['production'].mean()
plt.figure(figsize=(10, 5))
quarterly_avg.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Sản lượng trung bình theo Quý (Seasonality)')
plt.xlabel('Quý')
plt.ylabel('Sản lượng')
plt.xticks(rotation=0)
plt.show()

result = seasonal_decompose(df['production'], model='additive', period=52)

fig = result.plot()
fig.set_size_inches(10, 5)
plt.tight_layout()
plt.show()