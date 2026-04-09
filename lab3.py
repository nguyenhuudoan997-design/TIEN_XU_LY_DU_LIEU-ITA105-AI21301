import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import zscore
#BÀI1:
# 1. Load dữ liệu & Kiểm tra
df = pd.read_csv("ITA105_Lab_3_Sports.csv")
print("Kích thước:", df.shape)
print("\nThống kê mô tả:\n", df.describe())

# 2. Chọn cột numeric
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# 3. Thực hiện chuẩn hóa
# Min-Max Scaling [0, 1]
scaler_mm = MinMaxScaler()
df_minmax = pd.DataFrame(scaler_mm.fit_transform(df[numeric_cols]), columns=numeric_cols)

# Z-score Normalization (Mean=0, Std=1)
scaler_std = StandardScaler()
df_zscore = pd.DataFrame(scaler_std.fit_transform(df[numeric_cols]), columns=numeric_cols)

# 4. So sánh phân phối bằng Histogram 
for col in numeric_cols:
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    
    # Gốc
    axes[0].hist(df[col], bins=10, color='skyblue', edgecolor='black')
    axes[0].set_title(f"Gốc - {col}")
    
    # Min-Max
    axes[1].hist(df_minmax[col], bins=10, color='salmon', edgecolor='black')
    axes[1].set_title(f"Min-Max (0-1) - {col}")
    
    # Z-score
    axes[2].hist(df_zscore[col], bins=10, color='lightgreen', edgecolor='black')
    axes[2].set_title(f"Z-score (Standard) - {col}")
    
    plt.tight_layout()
    plt.show()

# 5. So sánh bằng Boxplot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

df[numeric_cols].boxplot(ax=axes[0])
axes[0].set_title("Boxplot Gốc")

df_minmax[numeric_cols].boxplot(ax=axes[1])
axes[1].set_title("Boxplot Min-Max")

df_zscore[numeric_cols].boxplot(ax=axes[2])
axes[2].set_title("Boxplot Z-score")

plt.tight_layout()
plt.show()

#BÀI2:
# 1. Load & Thống kê
df = pd.read_csv("ITA105_Lab_3_Health.csv")
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# 2. Thực hiện tính toán Min-Max và Z-score 
scaler_mm = MinMaxScaler()
df_minmax = pd.DataFrame(scaler_mm.fit_transform(df[numeric_cols]), columns=numeric_cols)

scaler_std = StandardScaler()
df_zscore = pd.DataFrame(scaler_std.fit_transform(df[numeric_cols]), columns=numeric_cols)

# 3. So sánh Histogram 
for col in numeric_cols:
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    
    axes[0].hist(df[col], bins=10, color='skyblue', edgecolor='black')
    axes[0].set_title(f"Gốc - {col}")
    
    axes[1].hist(df_minmax[col], bins=10, color='salmon', edgecolor='black')
    axes[1].set_title(f"Min-Max - {col}")
    
    axes[2].hist(df_zscore[col], bins=10, color='lightgreen', edgecolor='black')
    axes[2].set_title(f"Z-score - {col}")
    
    plt.tight_layout()
    plt.show()

# 4. So sánh Boxplot sau chuẩn hóa
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
df_minmax.boxplot(ax=axes[0])
axes[0].set_title("Boxplot Sau Min-Max")
df_zscore.boxplot(ax=axes[1])
axes[1].set_title("Boxplot Sau Z-score")
plt.show()

#BÀI3:
sns.set_theme(style="whitegrid")

# 1. Đọc dữ liệu
df = pd.read_csv('ITA105_Lab_3_Finance.csv')

print("--- Thống kê mô tả dữ liệu gốc ---")
print(df.describe())

# 2. Kiểm tra Outlier doanh thu bằng IQR 
Q1 = df['doanh_thu_musd'].quantile(0.25)
Q3 = df['doanh_thu_musd'].quantile(0.75)
IQR = Q3 - Q1
upper_limit = Q3 + 1.5 * IQR

outliers = df[df['doanh_thu_musd'] > upper_limit]
print(f"\nSố lượng công ty 'khủng' (Outliers): {len(outliers)}")
print(outliers[['doanh_thu_musd', 'loi_nhuan_musd']])

# 3. Chuẩn hóa dữ liệu
cols_to_scale = ['doanh_thu_musd', 'loi_nhuan_musd', 'so_nhan_vien', 'EPS']

# Khởi tạo Scaler
mm_scaler = MinMaxScaler()
z_scaler = StandardScaler()

# Tạo các bản sao DataFrame để so sánh
df_minmax = df.copy()
df_zscore = df.copy()

# Thực hiện chuyển đổi
df_minmax[cols_to_scale] = mm_scaler.fit_transform(df[cols_to_scale])
df_zscore[cols_to_scale] = z_scaler.fit_transform(df[cols_to_scale])

# 4. Vẽ Scatterplot so sánh (Doanh thu vs Lợi nhuận)
fig, axes = plt.subplots(1, 3, figsize=(12, 5))

# Đồ thị gốc
sns.scatterplot(x='doanh_thu_musd', y='loi_nhuan_musd', data=df, ax=axes[0], color='blue')
axes[0].set_title("1. Original Data (Scale lớn)")

# Đồ thị Min-Max
sns.scatterplot(x='doanh_thu_musd', y='loi_nhuan_musd', data=df_minmax, ax=axes[1], color='orange')
axes[1].set_title("2. Min-Max Scaling (0 to 1)")

# Đồ thị Z-Score
sns.scatterplot(x='doanh_thu_musd', y='loi_nhuan_musd', data=df_zscore, ax=axes[2], color='green')
axes[2].set_title("3. Z-Score Normalization (Standard)")

plt.tight_layout()
plt.show()

# 5. Boxplot so sánh tổng thể các biến sau chuẩn hóa
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

sns.boxplot(data=df_minmax[cols_to_scale], ax=axes[0])
axes[0].set_title("Boxplot sau Min-Max (Bị nén bởi ngoại lệ)")

sns.boxplot(data=df_zscore[cols_to_scale], ax=axes[1])
axes[1].set_title("Boxplot sau Z-Score (Giữ được độ phân tán)")

plt.show()

#BÀI4:
sns.set_theme(style="whitegrid")

# 1. Đọc dữ liệu & Kiểm tra
df = pd.read_csv('ITA105_Lab_3_Gaming.csv')

print("--- Thông tin dữ liệu ---")
print(f"Kích thước: {df.shape}")
print("\nGiá trị thiếu:\n", df.isnull().sum())
print("\nThống kê mô tả:\n", df.describe())

# 2. Chọn các cột numeric để chuẩn hóa
numeric_cols = df.select_dtypes(include=[np.number]).columns

# 3. Thực hiện chuẩn hóa
# Min-Max Scaling (Đưa về khoảng [0, 1])
mm_scaler = MinMaxScaler()
df_minmax = pd.DataFrame(mm_scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)

# Z-Score Normalization (Mean=0, Std=1)
z_scaler = StandardScaler()
df_zscore = pd.DataFrame(z_scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)

# 4. Vẽ Histogram so sánh phân phối trước và sau chuẩn hóa
for col in numeric_cols:
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    
    # Dữ liệu gốc
    sns.histplot(df[col], kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title(f"Gốc - {col}")
    
    # Sau Min-Max
    sns.histplot(df_minmax[col], kde=True, ax=axes[1], color='salmon')
    axes[1].set_title(f"Min-Max (0-1) - {col}")
    
    # Sau Z-Score
    sns.histplot(df_zscore[col], kde=True, ax=axes[2], color='green')
    axes[2].set_title(f"Z-Score (Standard) - {col}")
    
    plt.tight_layout()
    plt.show()

# 5. Boxplot để quan sát ngoại lệ sau chuẩn hóa
plt.figure(figsize=(13, 5))
sns.boxplot(data=df_zscore)
plt.title("Boxplot các chỉ số Game sau khi chuẩn hóa Z-Score")
plt.show()
