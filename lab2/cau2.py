import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

# 1. Đọc dữ liệu từ file CSV
data = pd.read_csv('Drug.csv')

# 2. Tiền xử lý dữ liệu
# Chuyển các biến phân loại (Sex, BP, Cholesterol) thành dạng số
le_sex = LabelEncoder()
data['Sex'] = le_sex.fit_transform(data['Sex'])

le_bp = LabelEncoder()
data['BP'] = le_bp.fit_transform(data['BP'])

le_cholesterol = LabelEncoder()
data['Cholesterol'] = le_cholesterol.fit_transform(data['Cholesterol'])

le_drug = LabelEncoder()
data['Drug'] = le_drug.fit_transform(data['Drug'])  # Encode labels for drugs (A, B, C, X, Y)

# 3. Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y = data['Drug']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Áp dụng Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

# 5. Đánh giá mô hình
print("Gaussian Naive Bayes Classification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))
