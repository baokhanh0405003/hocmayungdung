# hocmayungdung
Các công nghệ sử dụng
  1. Pandas:
  Chức năng: Được sử dụng để đọc và xử lý tập dữ liệu từ file CSV, quản lý và thao tác dữ liệu dưới dạng bảng (DataFrame).
  Ứng dụng trong bài: Đọc file Drug.csv, xử lý các cột dữ liệu và chuẩn bị đầu vào cho mô hình.
  2. Scikit-learn:
  Chức năng: Thư viện phổ biến nhất cho các thuật toán học máy trong Python. Nó cung cấp các công cụ mạnh mẽ để huấn luyện và đánh giá mô hình học máy.
  Ứng dụng trong bài:
  Label Encoding: Sử dụng LabelEncoder để chuyển đổi các biến phân loại như giới tính (Sex), huyết áp (BP), và cholesterol (Cholesterol) thành các số nguyên mà mô hình có thể xử lý.
  Chia tập dữ liệu: Sử dụng train_test_split để chia dữ liệu thành tập huấn luyện và kiểm tra.
  Thuật toán Gaussian Naive Bayes: Sử dụng GaussianNB để áp dụng mô hình Naive Bayes với phân phối Gaussian.
  Đánh giá mô hình: Sử dụng các công cụ như accuracy_score và classification_report để đánh giá hiệu suất mô hình.
  3. NumPy (implicit through scikit-learn):
  Chức năng: Thư viện cơ bản cho tính toán khoa học với Python, đặc biệt là các mảng đa chiều và các phép toán ma trận. Nhiều hàm trong scikit-learn sử dụng NumPy ngầm để thực hiện các tính toán.
  Ứng dụng trong bài: Không được gọi trực tiếp, nhưng scikit-learn sử dụng NumPy ngầm trong quá trình huấn luyện và dự đoán của mô hình.
  4. Học máy với Gaussian Naive Bayes:
  Chức năng: Gaussian Naive Bayes là một thuật toán học máy dựa trên định lý Bayes, giả định rằng các đặc trưng tuân theo phân phối chuẩn (Gaussian). Nó được sử dụng để phân loại các quan sát dựa trên xác suất của chúng.
  Ứng dụng trong bài: Dự đoán loại thuốc mà bệnh nhân nên dùng dựa trên các đặc trưng như tuổi, giới tính, huyết áp, mức cholesterol, và tỷ lệ Na_to_K.

Các thuật toán chủ yếu sử dụng
  1. Gaussian Naive Bayes:
Loại thuật toán: Thuật toán phân loại thuộc nhóm Naive Bayes.
Cách hoạt động: Gaussian Naive Bayes giả định rằng các đặc trưng liên quan trong tập dữ liệu tuân theo phân phối chuẩn (Gaussian). Thuật toán sử dụng định lý Bayes để tính xác suất của mỗi nhãn dựa trên các đặc trưng, sau đó chọn nhãn có xác suất cao nhất.
Ứng dụng trong bài: Được sử dụng để dự đoán loại thuốc (Drug) dựa trên các đặc trưng như tuổi, giới tính, huyết áp, cholesterol và tỷ lệ Na_to_K.
2. Label Encoding:
Loại thuật toán: Kỹ thuật tiền xử lý dữ liệu.
Cách hoạt động: Chuyển đổi các giá trị phân loại (categorical) thành các giá trị số nguyên. Ví dụ, giới tính có thể được chuyển thành 0 cho 'Male' và 1 cho 'Female'.
Ứng dụng trong bài: Chuyển đổi các thuộc tính dạng phân loại như Sex, BP (Huyết áp), và Cholesterol thành dạng số trước khi đưa vào mô hình Naive Bayes.
3. Train-Test Split:
Loại thuật toán: Kỹ thuật chia dữ liệu.
Cách hoạt động: Chia dữ liệu thành hai phần: một phần dùng để huấn luyện mô hình (train set) và một phần dùng để kiểm tra hiệu suất của mô hình (test set). Thông thường, dữ liệu được chia theo tỷ lệ như 80% dùng để huấn luyện và 20% dùng để kiểm tra.
Ứng dụng trong bài: Chia tập dữ liệu ban đầu thành hai phần để huấn luyện và đánh giá mô hình Gaussian Naive Bayes.
