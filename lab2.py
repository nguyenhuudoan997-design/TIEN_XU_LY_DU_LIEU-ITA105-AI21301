import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
import seaborn as sns
#BÀI1:
# 1. Nạp dữ liệu & kiểm tra shape, missing values
df = pd.read_csv("ITA105_Lab_2_Housing.csv")
print("1. Kích thước dữ liệu:", df.shape)
print("Số lượng giá trị thiếu:\n", df.isnull().sum())

# 2. Thống kê mô tả (mean, median, std, min, max)
numeric_cols = df.select_dtypes(include=[np.number]).columns
stats = df[numeric_cols].describe().loc[['mean', 'std', 'min', 'max']]
stats.loc['median'] = df[numeric_cols].median()
print("\n2. Thống kê mô tả:\n", stats)

# 3. Vẽ boxplot cho từng biến numeric
plt.figure(figsize=(10, 6))
df[numeric_cols].boxplot()
plt.title("3. Boxplot các biến số (Phát hiện ngoại lệ)")
plt.show()

# 4. Vẽ scatterplot diện tích và giá
plt.figure(figsize=(8, 5))
plt.scatter(df["dien_tich"], df["gia"], alpha=0.6, edgecolors='w')
plt.title("4. Scatterplot: Diện tích vs Giá")
plt.xlabel("Diện tích")
plt.ylabel("Giá")
plt.grid(True)
plt.show()

# 5. Tính IQR và xác định ngoại lệ cho cột Giá
Q1 = df["gia"].quantile(0.25)
Q3 = df["gia"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_iqr = df[(df["gia"] < lower_bound) | (df["gia"] > upper_bound)]
print(f"\n5. IQR: {IQR}, Ngưỡng: [{lower_bound}, {upper_bound}]")
print(f"Số lượng ngoại lệ theo IQR: {len(outliers_iqr)}")

# 6. Tính Z-score và xác định ngoại lệ (|Z| > 3)
df_z = df.copy()
for col in numeric_cols:
    df_z["z_" + col] = zscore(df[col])

# Lọc ngoại lệ nếu bất kỳ cột nào có |Z| > 3
outliers_z = df_z[(abs(df_z["z_dien_tich"]) > 3) | 
(abs(df_z["z_gia"]) > 3) | 
(abs(df_z["z_so_phong"]) > 3)]
print(f"6. Số lượng ngoại lệ theo Z-score: {len(outliers_z)}")

# 7. So sánh số lượng ngoại lệ
print(f"\n7. So sánh số lượng ngoại lệ:")
print(f"- IQR: {len(outliers_iqr)}")
print(f"- Z-score: {len(outliers_z)}")


# 8. Áp dụng xử lý ngoại lệ: dùng Clip (giới hạn giá trị trong khoảng IQR)
df_processed = df.copy()
df_processed["gia"] = df_processed["gia"].clip(lower=lower_bound, upper=upper_bound)
# 9. Vẽ lại boxplot sau xử lý
plt.figure(figsize=(8, 5))
plt.boxplot([df["gia"], df_processed["gia"]], tick_labels=["Trước xử lý", "Sau xử lý"])
plt.title("9. So sánh Boxplot Giá nhà Trước và Sau xử lý")
plt.ylabel("Giá")
plt.show()

#BÀI2:
# 1. Load & Set Index
df = pd.read_csv("ITA105_Lab_2_Iot.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp").sort_index()

# 2. Line plot theo từng sensor
plt.figure(figsize=(12, 5))
for sensor in df["sensor_id"].unique():
    data = df[df["sensor_id"] == sensor]
    plt.plot(data.index, data["temperature"], label=sensor, alpha=0.7)
plt.title("Temperature theo thời gian cho từng sensor")
plt.legend()
plt.show()

# 3 & 4. Phát hiện ngoại lệ (Rolling vs Z-score) 
s1 = df[df["sensor_id"] == "S1"].copy()

# Rolling mean
s1["rolling_mean"] = s1["temperature"].rolling(window=10).mean()
s1["rolling_std"] = s1["temperature"].rolling(window=10).std()
s1["upper"] = s1["rolling_mean"] + 3 * s1["rolling_std"]
s1["lower"] = s1["rolling_mean"] - 3 * s1["rolling_std"]

# Z-score
s1["z_temp"] = zscore(s1["temperature"])

# Lọc ngoại lệ 
outliers_rolling = s1.dropna(subset=['rolling_mean'])
outliers_rolling = outliers_rolling[(outliers_rolling["temperature"] > outliers_rolling["upper"]) | 
(outliers_rolling["temperature"] < outliers_rolling["lower"])]

outliers_z = s1[abs(s1["z_temp"]) > 3]

# 5. Scatter Plot với HIGHLIGHT điểm bất thường
plt.figure(figsize=(10, 6))

plt.scatter(s1["temperature"], s1["pressure"], alpha=0.3, label="Normal")

plt.scatter(outliers_z["temperature"], outliers_z["pressure"], color='red', label="Outliers (Z>3)")
plt.xlabel("Temperature")
plt.ylabel("Pressure")
plt.title("Scatter Plot: Temperature vs Pressure (Highlighted Outliers)")
plt.legend()
plt.show()

# 7. Xử lý bằng Interpolation 
s1["temp_cleaned"] = s1["temperature"].copy()

s1.loc[abs(s1["z_temp"]) > 3, "temp_cleaned"] = None 
s1["temp_cleaned"] = s1["temp_cleaned"].interpolate(method='time')

# Vẽ lại để so sánh
plt.figure(figsize=(12, 5))
plt.plot(s1.index, s1["temperature"], label="Original (Nhiễu)", alpha=0.5)
plt.plot(s1.index, s1["temp_cleaned"], label="After Interpolation", color='green')
plt.title("So sánh dữ liệu Sensor trước và sau khi xử lý (Interpolation)")
plt.legend()
plt.show()

#BÀI3:
# 1. Load dữ liệu & Thống kê
df = pd.read_csv("ITA105_Lab_2_Ecommerce.csv")
print("Kích thước:", df.shape)
print("\nGiá trị thiếu:\n", df.isnull().sum())
print("\nThống kê mô tả:\n", df.describe())

# 2. Boxplot ban đầu
plt.figure(figsize=(10, 5))
df[["price", "quantity", "rating"]].boxplot()
plt.title("Boxplot: Price, Quantity, Rating (Trước xử lý)")
plt.show()

# 

# 3. IQR và Z-score
# Tính IQR cho Price
Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)
IQR = Q3 - Q1
duoi = Q1 - 1.5 * IQR
tren = Q3 + 1.5 * IQR

# Xác định ngoại lệ theo Z-score cho cả 3 cột
cols = ["price", "quantity", "rating"]
for cot in cols:
    df["z_" + cot] = zscore(df[cot])

ngoai_le_z = df[(abs(df["z_price"]) > 3) | (abs(df["z_quantity"]) > 3) | (abs(df["z_rating"]) > 3)]
print(f"\nSố lượng ngoại lệ Z-score: {len(ngoai_le_z)}")

# 4. Scatter plot Price vs Quantity
plt.figure(figsize=(8, 5))
plt.scatter(df["price"], df["quantity"], alpha=0.5, label="Bình thường")
plt.scatter(ngoai_le_z["price"], ngoai_le_z["quantity"], color="red", label="Ngoại lệ (Z>3)")
plt.xlabel("Price")
plt.ylabel("Quantity")
plt.title("Scatter Plot: Price vs Quantity")
plt.legend()
plt.show()

# 

# 5. Phân tích nguyên nhân 
print("\n--- PHÂN TÍCH BẤT THƯỜNG ---")
print(f"Số đơn hàng giá 0: {len(df[df['price'] == 0])}")
print(f"Số đơn hàng rating > 5: {len(df[df['rating'] > 5])}")

# 6. Xử lý ngoại lệ
df_clean = df[(df["price"] > 0) & (df["rating"] >= 1) & (df["rating"] <= 5)].copy()

df_clean["price_final"] = df_clean["price"].clip(lower=duoi, upper=tren)

# 7. Vẽ lại sau xử lý
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Boxplot so sánh Price
axes[0].boxplot([df["price"], df_clean["price_final"]], tick_labels=["Trước", "Sau (Clip)"])
axes[0].set_title("So sánh Boxplot Price")

# Scatter sau xử lý
axes[1].scatter(df_clean["price_final"], df_clean["quantity"], alpha=0.5, color='green')
axes[1].set_title("Scatter: Price (Clipped) vs Quantity")
axes[1].set_xlabel("Price")
axes[1].set_ylabel("Quantity")

plt.tight_layout()
plt.show()

#BÀI4:
sns.set_theme(style="whitegrid")

# PHẦN 1: HOUSING (Diện tích + Giá)

print("\n===== HOUSING =====")
housing = pd.read_csv("ITA105_Lab_2_Housing.csv")

# Tính Z-score cho các biến quan tâm
housing["z_dien_tich"] = zscore(housing["dien_tich"])
housing["z_gia"] = zscore(housing["gia"])

# Xác định ngoại lệ đa biến
housing_outlier = housing[(abs(housing["z_dien_tich"]) > 3) | (abs(housing["z_gia"]) > 3)]

print(f"Số lượng ngoại lệ phát hiện: {len(housing_outlier)}")

# Vẽ Scatter plot highlight ngoại lệ
plt.figure(figsize=(8, 5))
sns.scatterplot(data=housing, x="dien_tich", y="gia", alpha=0.5, label="Bình thường")
sns.scatterplot(data=housing_outlier, x="dien_tich", y="gia", color="red", label="Ngoại lệ", s=100, edgecolor='black')
plt.title("Housing: Multivariate Outlier (Area vs Price)")
plt.legend()
plt.show()

# PHẦN 2: IOT (Temperature + Pressure)

print("\n===== IOT =====")
iot = pd.read_csv("ITA105_Lab_2_Iot.csv")

# Tính Z-score
iot["z_temp"] = zscore(iot["temperature"])
iot["z_pressure"] = zscore(iot["pressure"])

# Xác định ngoại lệ
iot_outlier = iot[(abs(iot["z_temp"]) > 3) | (abs(iot["z_pressure"]) > 3)]

print(f"Số lượng ngoại lệ phát hiện: {len(iot_outlier)}")

# Vẽ Scatter plot
plt.figure(figsize=(8, 5))
sns.scatterplot(data=iot, x="temperature", y="pressure", alpha=0.5, label="Bình thường")
sns.scatterplot(data=iot_outlier, x="temperature", y="pressure", color="red", label="Ngoại lệ", s=100, edgecolor='black')
plt.title("IoT: Multivariate Outlier (Temp vs Pressure)")
plt.legend()
plt.show()

# PHẦN 3: E-COMMERCE (Price + Quantity + Rating)

print("\n===== E-COMMERCE =====")
eco = pd.read_csv("ITA105_Lab_2_Ecommerce.csv")

# Tính Z-score cho 3 biến
cols_eco = ["price", "quantity", "rating"]
for col in cols_eco:
    eco["z_" + col] = zscore(eco[col])

# Xác định ngoại lệ đa biến (kết hợp 3 điều kiện)
eco_outlier = eco[(abs(eco["z_price"]) > 3) | 
                  (abs(eco["z_quantity"]) > 3) | 
                  (abs(eco["z_rating"]) > 3)]

print(f"Số lượng ngoại lệ phát hiện: {len(eco_outlier)}")

eco["is_outlier"] = "Normal"
eco.loc[eco_outlier.index, "is_outlier"] = "Outlier"

pair_plot = sns.pairplot(eco, vars=cols_eco, hue="is_outlier", palette={"Normal": "skyblue", "Outlier": "red"}, diag_kind="kde")
pair_plot.fig.suptitle("E-commerce: Scatter Matrix Outlier Detection", y=1.02)
plt.show()

