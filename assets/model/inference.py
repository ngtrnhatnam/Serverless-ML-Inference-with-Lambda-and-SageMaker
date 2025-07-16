import torch    # Thư viện chạy model
import torchvision.transforms as transforms    # Hàm biến đổi ảnh
from PIL import Image   # Đọc file ảnh
import io   # Xử lý nhị phân

def model_fn(model_dir):    # SageMaker nạp hết file .tar.gz vào thư mục này khi container khởi chạy.
    model = torch.load(f"{model_dir}/model.pth", map_location=torch.device("cpu"))    # 	Mở file model.pth và nạp vào RAM, chạy bằng CPU
    model.eval()    # Bật “chế độ dự đoán” (tắt dropout, batchnorm training).
    return model    

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/x-image':
        image = Image.open(io.BytesIO(request_body)).convert("RGB")     # 	Đọc byte stream thành ảnh Pillow.
        transform = transforms.Compose([    
            transforms.Resize((224, 224)),      # ResNet18 cần input 224×224.
            transforms.ToTensor(),              # Đổi ảnh thành tensor (giá trị 0–1).
        ])
        return transform(image).unsqueeze(0)    # Thêm batch dimension -> shape [batch, channel, H, W].
    raise Exception(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    with torch.no_grad():   # Không cần tính gradient (tiết kiệm RAM/CPU)
        output = model(input_data)  #  Forward pass – model spit logits
        _, predicted = torch.max(output.data, 1)
        return "Dog" if predicted.item() == 1 else "Cat"

def output_fn(prediction, content_type):
    return prediction