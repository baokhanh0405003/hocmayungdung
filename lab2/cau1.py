import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Đọc dữ liệu
data = pd.read_csv('Education.csv')

# 2. Tiền xử lý dữ liệu
# Chuyển các nhãn cảm xúc thành số: Positive -> 1, Negative -> 0
data['Label'] = data['Label'].apply(lambda x: 1 if x == 'Positive' else 0)

# 3. Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(data['Text'], data['Label'], test_size=0.2, random_state=42)

# 4. Vector hóa dữ liệu văn bản (sử dụng Bag of Words hoặc TF-IDF)
vectorizer = CountVectorizer()  # Bạn có thể thử TfidfVectorizer() ở đây để so sánh
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Áp dụng mô hình Naive Bayes với phân phối Bernoulli
bernoulli_nb = BernoulliNB()
bernoulli_nb.fit(X_train_vec, y_train)
y_pred_bernoulli = bernoulli_nb.predict(X_test_vec)

# 6. Đánh giá mô hình Bernoulli
print("Bernoulli Naive Bayes Classification Report:")
print(classification_report(y_test, y_pred_bernoulli))
print("Accuracy:", accuracy_score(y_test, y_pred_bernoulli))

# 7. Áp dụng mô hình Naive Bayes với phân phối Multinomial
multinomial_nb = MultinomialNB()
multinomial_nb.fit(X_train_vec, y_train)
y_pred_multinomial = multinomial_nb.predict(X_test_vec)

# 8. Đánh giá mô hình Multinomial
print("\nMultinomial Naive Bayes Classification Report:")
print(classification_report(y_test, y_pred_multinomial))
print("Accuracy:", accuracy_score(y_test, y_pred_multinomial))

# 9. So sánh kết quả giữa hai phân phối
