Thư viện và công nghệ
PyTorch:

Thư viện chủ yếu được sử dụng là torch, một framework học sâu phổ biến, hỗ trợ phát triển các mô hình học máy và học sâu.
torch.nn để xây dựng các mô hình mạng nơ-ron.
torch.optim để tối ưu hóa các tham số của mô hình.
torch.utils.data.DataLoader để tải dữ liệu một cách linh hoạt và hiệu quả.
Torchvision:

Sử dụng torchvision.datasets để tải tập dữ liệu MNIST, một bộ dữ liệu nổi tiếng về các chữ số viết tay.
torchvision.transforms để tiền xử lý dữ liệu (biến đổi ảnh thành tensor và chuẩn hóa giá trị).
Các thuật toán và kỹ thuật
Mạng Nơ-ron Đa Lớp (MLP - Multilayer Perceptron):

Định nghĩa một mạng nơ-ron bao gồm các lớp fully connected (nn.Linear) với ba lớp chính:
Lớp đầu vào: Biến đổi ảnh 28x28 thành vector 128 chiều.
Lớp ẩn: Giảm chiều xuống 64.
Lớp đầu ra: Gồm 10 đầu ra, ứng với các nhãn số từ 0 đến 9.
Tối ưu hóa:

Sử dụng các phương pháp tối ưu hóa từ torch.optim, như SGD hoặc Adam, để điều chỉnh các tham số của mô hình.
Chuẩn hóa:

Ảnh được chuẩn hóa với giá trị trung bình và độ lệch chuẩn là 0.5 để cải thiện tốc độ huấn luyện và tính ổn định của mô hình.
