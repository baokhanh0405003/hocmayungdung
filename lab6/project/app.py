import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template

# Định nghĩa mô hình MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Khởi tạo Flask app
app = Flask(__name__)

# Tải mô hình đã huấn luyện
model = MLP()
model.load_state_dict(torch.load('mlp_mnist_model.pth', map_location=torch.device('cpu')))
model.eval()

# Định nghĩa các phép biến đổi ảnh
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Route để render trang HTML
@app.route('/')
def index():
    return render_template('index.html')

# Route để nhận ảnh và trả về kết quả dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    # Xử lý ảnh và chuyển đổi thành định dạng phù hợp
    try:
        img = Image.open(file).convert('L')
        img = transform(img)
        img = img.unsqueeze(0)  # Thêm batch dimension
    except Exception as e:
        return jsonify({'error': 'Invalid image format'}), 400
    
    # Thực hiện dự đoán
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output.data, 1)
        predicted_digit = predicted.item()
    
    # Trả về kết quả dự đoán
    return jsonify({'predicted_digit': predicted_digit})

# Chạy ứng dụng
if __name__ == '__main__':
    app.run(debug=True)
