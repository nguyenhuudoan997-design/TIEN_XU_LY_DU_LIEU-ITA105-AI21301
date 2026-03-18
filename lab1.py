# 1. khám phá dữ liệu
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("ITA105_Lab_1.csv")

print("5 dòng đầu:")
print(df.head())

# kiểm tra kích thước dữ liệu
print("- kiểm tra kích thước dữ liệu:")
print(df.shape)

# thống kê mô tả
print("- thống kê mô tả dữ liệu:")
print(df.describe())

# kiểm tra kiểu dữ liệu
print("- kiểm tra kiểu dữ liệu:")
df.info()

# kiểm tra dữ liệu thiếu
print("- kiểm tra dữ liệu thiếu:")
print(df.isnull().sum())

# 2. xử lý dữ liệu thiếu
print("- điền dữ liệu thiếu:")
df["Price"] = df["Price"].fillna(df["Price"].mean())
df["StockQuantity"] = df["StockQuantity"].fillna(df["StockQuantity"].median())
df["Category"] = df["Category"].fillna(df["Category"].mode()[0])

# so sánh với dropna
df_drop = df.dropna()
print("Kích thước sau dropna:")
print(df_drop.shape)

# 3. xử lý dữ liệu lỗi
print("Price âm:")
print(df[df["Price"] < 0])

print("StockQuantity âm:")
print(df[df["StockQuantity"] < 0])

print("Rating sai:")
print(df[(df["Rating"] < 1) | (df["Rating"] > 5)])

# loại bỏ dữ liệu sai
df = df[df["Price"] > 0]
df = df[df["StockQuantity"] >= 0]
df = df[(df["Rating"] >= 1) & (df["Rating"] <= 5)]

print("Kích thước sau xử lý lỗi:")
print(df.shape)

# 4. làm mượt dữ liệu
df["Price_Smooth"] = df["Price"].rolling(window=5, min_periods=1).mean()

# biểu đồ
plt.figure(figsize=(10,5))
plt.plot(df["Price"], label="Giá gốc", marker='o')
plt.plot(df["Price_Smooth"], label="Giá sau làm mượt", linewidth=3)

plt.title("So sánh giá trước và sau làm mượt")
plt.xlabel("Dòng dữ liệu")
plt.ylabel("Giá")

plt.grid(True)
plt.legend()
plt.show()

# 5. chuẩn hóa dữ liệu
df["Category"] = df["Category"].str.lower()
df["Description"] = df["Description"].str.replace("[^a-zA-Z0-9 ]","", regex=True)

df["Price_VND"] = df["Price"] * 24000

scaler = MinMaxScaler()
df["Price_Normalized"] = scaler.fit_transform(df[["Price"]])

print(df.head())

