import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import joblib

df = pd.read_csv('ITA105_Lab_8.csv')

num_features = ['LotArea', 'Rooms', 'NoiseFeature']
cat_features = ['Neighborhood', 'Condition']
text_feature = 'Description'

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('power', PowerTransformer(method='yeo-johnson'))
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

text_transformer = Pipeline(steps=[
    ('tfidf', TfidfVectorizer(max_features=50, stop_words='english'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features),
        ('text', text_transformer, text_feature)
    ])

test_data = df.head(10)
output = preprocessor.fit_transform(test_data)
print(f"Hình dạng dữ liệu sau Pipeline: {output.shape}")

bad_data = pd.DataFrame({
    'LotArea': [2500, np.nan],
    'Rooms': [3, 4],
    'NoiseFeature': [0.5, -1.2],
    'Neighborhood': ['Z', 'B'],
    'Condition': ['Good', 'Excellent'],
    'Description': ['luxury house with garden', 'small cozy room']
})

try:
    bad_output = preprocessor.transform(bad_data)
    print("Pipeline xử lý thành công dữ liệu có Unseen Category!")
except Exception as e:
    print(f"Pipeline lỗi: {e}")

plt.hist(df['LotArea'], bins=20, alpha=0.5, label='Trước')
plt.hist(output[:, 0], bins=20, alpha=0.5, label='Sau (Scaled)')
plt.legend()
plt.title("Phân phối LotArea Trước và Sau Pipeline")
plt.show()

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

X = df.drop(columns=['SalePrice', 'SaleDate', 'ImagePath'])
y = df['SalePrice']

cv_scores = cross_val_score(full_pipeline, X, y, cv=5, scoring='r2')
print(f"R2 Score trung bình (5-fold): {cv_scores.mean():.4f}")

full_pipeline.fit(X, y)
joblib.dump(full_pipeline, 'house_price_model.pkl')

def predict_price(new_data_path):
    model = joblib.load('house_price_model.pkl')
    new_data = pd.read_csv(new_data_path)
    new_data = new_data[X.columns]
    predictions = model.predict(new_data)
    return predictions
