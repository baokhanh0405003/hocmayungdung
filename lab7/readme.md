Phân Loại Ảnh CIFAR-10 với MLP Cơ Bản
Giới Thiệu Dự Án
Dự án này triển khai một mô hình MLP (Perceptron nhiều tầng) cơ bản để phân loại ảnh từ bộ dữ liệu CIFAR-10 vào một trong mười danh mục. Dự án thực hiện huấn luyện, đánh giá, và hiển thị kết quả của mô hình.

Các Công Nghệ Sử Dụng
PyTorch: Thư viện học sâu cho xây dựng và huấn luyện mô hình.
Torchvision: Cung cấp các tập dữ liệu hình ảnh phổ biến và các phép biến đổi ảnh hỗ trợ xử lý dữ liệu.
Matplotlib: Thư viện trực quan hóa dữ liệu để hiển thị kết quả.
Thuật Toán
Mô hình MLP sử dụng kiến trúc bao gồm các lớp tuyến tính (Fully Connected) với hàm kích hoạt ReLU và Dropout để giúp giảm thiểu hiện tượng quá khớp. Cấu trúc chính của mô hình:

Lớp Flatten: Chuyển ảnh 2D thành vector 1D.
Lớp Fully Connected (fc1, fc2): Lớp ẩn đầu tiên và thứ hai với kích thước đầu ra tương ứng là 512 và 256.
Dropout: Áp dụng sau mỗi lớp ẩn để giảm quá khớp.
Lớp Fully Connected cuối (fc3): Cho đầu ra với kích thước bằng số lớp (10) để dự đoán nhãn.
Mô hình được tối ưu bằng thuật toán Adam với CrossEntropyLoss làm hàm mất mát.

Kết Quả
Sau khi huấn luyện trong 20 epoch, kết quả về độ chính xác và mất mát trên tập huấn luyện và tập kiểm tra được hiển thị như sau:

Độ Chính Xác và Mất Mát Qua Các Epoch

Đánh Giá Cuối Cùng
Độ mất mát trên tập kiểm tra: Test Loss
Độ chính xác trên tập kiểm tra: Test Accuracy