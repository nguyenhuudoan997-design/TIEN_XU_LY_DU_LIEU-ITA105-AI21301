import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#BÀI1:
# 1. Nạp dữ liệu & kiểm tra
df = pd.read_csv('ITA105_Lab_4_Hotel_reviews.csv')
# Kiểm tra và xử lý giá trị thiếu (Missing values)
print("Số lượng giá trị thiếu mỗi cột:\n", df.isnull().sum())

df = df.dropna(subset=['review_text'])

# 2. Encoding biến categorical
le = LabelEncoder()
df['customer_type_encoded'] = le.fit_transform(df['customer_type'].astype(str))

print("\nDanh sách các loại khách hàng đã mã hóa:")
print(dict(zip(le.classes_, le.transform(le.classes_))))

# 3. Tiền xử lý văn bản
stop_words = ["nhưng", "và", "là", "của", "có", "trong"]

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return tokens

df['tokens'] = df['review_text'].apply(preprocess_text)
df['clean_review'] = df['tokens'].apply(lambda x: " ".join(x))

# 4. TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['clean_review'])

print("\nKích thước TF-IDF matrix:", tfidf_matrix.shape)

# 5. Tìm từ gần giống với từ 'sạch'
feature_names = tfidf.get_feature_names_out()

word_to_check = "sạch"

if word_to_check in feature_names:
    idx = list(feature_names).index(word_to_check)

    similarities = cosine_similarity(
        tfidf_matrix.T[idx],
        tfidf_matrix.T
    )

    similar_indices = similarities[0].argsort()[-6:-1][::-1]

    print(f"\n5 từ gần với '{word_to_check}':")
    for i in similar_indices:
        print("-", feature_names[i])
else:
    print(f"\nTừ '{word_to_check}' không có trong từ điển.")
    
#BÀI2:
# 1. Nạp dữ liệu
df = pd.read_csv('ITA105_Lab_4_Match_comments.csv')

# Kiểm tra missing values
print("Số lượng giá trị thiếu mỗi cột:\n", df.isnull().sum())

# Xử lý missing values
df['team'] = df['team'].fillna('Unknown')
df['author'] = df['author'].fillna('Unknown')
df = df.dropna(subset=['comment_text'])

# 2. Label Encoding riêng từng cột
le_team = LabelEncoder()
le_author = LabelEncoder()

df['team_encoded'] = le_team.fit_transform(df['team'].astype(str))
df['author_encoded'] = le_author.fit_transform(df['author'].astype(str))

print("\nDanh sách team đã mã hóa:")
print(dict(zip(le_team.classes_, le_team.transform(le_team.classes_))))

# 3. Tiền xử lý văn bản
stop_words = ["và", "là", "của", "có", "nhưng", "đã", "đang", "với", "cho"]

def preprocess_comments(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return tokens

df['tokens'] = df['comment_text'].apply(preprocess_comments)
df['clean_comment'] = df['tokens'].apply(lambda x: " ".join(x))

# 4. TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['clean_comment'])

print("\nKích thước ma trận TF-IDF:", tfidf_matrix.shape)

# 5. Tìm từ gần giống từ 'xuất'
feature_names = tfidf_vectorizer.get_feature_names_out()
word_query = "xuất"

print(f"\n5 từ gần với '{word_query}':")

if word_query in feature_names:
    idx = list(feature_names).index(word_query)

    similarities = cosine_similarity(
        tfidf_matrix.T[idx],
        tfidf_matrix.T
    )

    similar_indices = similarities[0].argsort()[-6:-1][::-1]

    for i in similar_indices:
        print("-", feature_names[i])
else:
    print("Từ khóa không tồn tại trong tập dữ liệu.")
    
#BÀI3:
# 1. Nạp dữ liệu
df = pd.read_csv('ITA105_Lab_4_Player_feedback.csv')

# Kiểm tra missing values
print("Thông tin missing values:\n", df.isnull().sum())

# Xử lý missing values
df['device'] = df['device'].fillna('Unknown')
df['player_type'] = df['player_type'].fillna('Unknown')
df = df.dropna(subset=['feedback_text'])

# 2. Label Encoding riêng từng cột
le_player = LabelEncoder()
le_device = LabelEncoder()

df['player_type_encoded'] = le_player.fit_transform(df['player_type'].astype(str))
df['device_encoded'] = le_device.fit_transform(df['device'].astype(str))

print("\nVí dụ dữ liệu sau khi Encoding:")
print(df[['player_type', 'player_type_encoded', 'device', 'device_encoded']].head())

# 3. Tiền xử lý văn bản
stop_words = ["là", "của", "có", "trong", "và", "nhưng", "hơi", "rất", "với"]

def preprocess_feedback(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return tokens

df['tokens'] = df['feedback_text'].apply(preprocess_feedback)
df['clean_feedback'] = df['tokens'].apply(lambda x: " ".join(x))

# 4. TF-IDF matrix
tfidf_vec = TfidfVectorizer()
tfidf_matrix = tfidf_vec.fit_transform(df['clean_feedback'])

print("\nKích thước ma trận TF-IDF:", tfidf_matrix.shape)

# 5. Tìm từ gần với "đẹp" bằng cosine similarity
feature_names = tfidf_vec.get_feature_names_out()
word_target = "đẹp"

print(f"\n5 từ gần với '{word_target}':")

if word_target in feature_names:
    idx = list(feature_names).index(word_target)

    similarities = cosine_similarity(
        tfidf_matrix.T[idx],
        tfidf_matrix.T
    )

    similar_indices = similarities[0].argsort()[-6:-1][::-1]

    for i in similar_indices:
        print("-", feature_names[i])
else:
    print(f"Từ '{word_target}' không tồn tại trong tập dữ liệu.")
    
#BÀI4:
# 1. Nạp dữ liệu
df = pd.read_csv('ITA105_Lab_4_Album_reviews.csv')

# Kiểm tra missing values
print("Số lượng dòng trống:\n", df.isnull().sum())

# Xử lý missing values
df['genre'] = df['genre'].fillna('Unknown')
df['platform'] = df['platform'].fillna('Unknown')
df = df.dropna(subset=['review_text'])

# 2. Encoding categorical
le_genre = LabelEncoder()
le_platform = LabelEncoder()

df['genre_encoded'] = le_genre.fit_transform(df['genre'].astype(str))
df['platform_encoded'] = le_platform.fit_transform(df['platform'].astype(str))

print("\n--- Kết quả Encoding ---")
print(df[['genre', 'genre_encoded', 'platform', 'platform_encoded']].head())

# 3. Tiền xử lý văn bản
stop_words = ["là", "của", "có", "trong", "và", "nhưng", "rất", "với", "phần"]

def preprocess_album_review(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return tokens

df['tokens'] = df['review_text'].apply(preprocess_album_review)
df['clean_review'] = df['tokens'].apply(lambda x: " ".join(x))

# 4. TF-IDF
tfidf_vec = TfidfVectorizer()
tfidf_matrix = tfidf_vec.fit_transform(df['clean_review'])

print("\nKích thước ma trận TF-IDF:", tfidf_matrix.shape)

# 5. Tìm từ gần với "sáng"
feature_names = tfidf_vec.get_feature_names_out()
word_query = "sáng"

print(f"\n5 từ gần với '{word_query}' trong review album:")

if word_query in feature_names:
    idx = list(feature_names).index(word_query)

    similarities = cosine_similarity(
        tfidf_matrix.T[idx],
        tfidf_matrix.T
    )

    similar_indices = similarities[0].argsort()[-6:-1][::-1]

    for i in similar_indices:
        print("-", feature_names[i])
else:
    print("Từ không tồn tại trong dữ liệu.")