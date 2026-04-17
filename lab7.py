import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
import numpy as np
from scipy import stats
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# BÀI1:
df = pd.read_csv('ITA105_Lab_7.csv')
numeric_cols = df.select_dtypes(include=['number']).columns

skew_values = df[numeric_cols].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("--- Top 10 cột lệch nhất ---")
print(skew_values.head(10))

top_3_skew = skew_values.index[:3]
plt.figure(figsize=(10, 5))
for i, col in enumerate(top_3_skew):
    plt.subplot(1, 3, i+1)
    sns.histplot(df[col], kde=True, color='blue')
    plt.title(f"{col}\nSkew: {skew_values[col]:.2f}")
plt.tight_layout()
plt.show()

# BÀI2:
col_pos1, col_pos2, col_neg = 'SalePrice', 'LotArea', 'NegSkewIncome'

df_trans = df.copy()
df_trans['Log_SalePrice'] = np.log1p(df[col_pos1])
df_trans['BoxCox_LotArea'], lmbda = stats.boxcox(df[col_pos2])
pt = PowerTransformer(method='yeo-johnson')
df_trans['Power_NegIncome'] = pt.fit_transform(df[[col_neg]])

results = pd.DataFrame({
    'Phương pháp': ['Gốc', 'Log', 'Box-Cox', 'Yeo-Johnson'],
    'SalePrice': [skew(df[col_pos1]), skew(df_trans['Log_SalePrice']), '-', '-'],
    'LotArea': [skew(df[col_pos2]), '-', skew(df_trans['BoxCox_LotArea']), '-'],
    'NegIncome': [skew(df[col_neg]), '-', '-', skew(df_trans['Power_NegIncome'])]
})
print(results)

# BÀI3:
X = df[['LotArea', 'HouseAge', 'Rooms']].fillna(0)
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_a = LinearRegression().fit(X_train, y_train)
pred_a = model_a.predict(X_test)

y_train_log = np.log1p(y_train)
model_b = LinearRegression().fit(X_train, y_train_log)
pred_b_log = model_b.predict(X_test)
pred_b = np.expm1(pred_b_log) 

print(f"RMSE Gốc: {np.sqrt(mean_squared_error(y_test, pred_a)):.2f}")
print(f"RMSE sau Log: {np.sqrt(mean_squared_error(y_test, pred_b)):.2f}")

# BÀI4:
df['log_LotArea'] = np.log1p(df['LotArea'])
df['log_SalePrice'] = np.log1p(df['SalePrice'])

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(data=df, x='LotArea', y='SalePrice', alpha=0.5, color='red')
plt.title("Version A: Dữ liệu gốc (Bị nhiễu bởi Outliers)")
plt.xlabel("Diện tích (LotArea)")
plt.ylabel("Giá bán (SalePrice)")

plt.subplot(1, 2, 2)
sns.scatterplot(data=df, x='log_LotArea', y='log_SalePrice', alpha=0.5, color='green')
plt.title("Version B: Dữ liệu đã Log-transform (Rõ xu hướng)")
plt.xlabel("Log của Diện tích")
plt.ylabel("Log của Giá bán")
plt.tight_layout()
plt.show()

df['log_price_index'] = (df['log_SalePrice'] - df['log_SalePrice'].mean()) / df['log_SalePrice'].std()

print("--- 5 dòng đầu với chỉ số Log-Price-Index ---")
print(df[['SalePrice', 'log_SalePrice', 'log_price_index']].head())

high_value_houses = df[df['log_price_index'] > 2] 
print(f"\nSố lượng căn nhà có giá cao bất thường: {len(high_value_houses)}")