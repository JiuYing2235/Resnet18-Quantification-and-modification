import torch
from torch import nn
from resnet18 import Resnet18  # 确保 Resnet18 正确导入
from PIL import Image
from torchvision import transforms
import os

def load_images_from_folder(folder):
    images = []
    labels = []
    for label in ['Defective', 'Non defective']:  # 假设您的标签为缺陷和无缺陷
        class_folder = os.path.join(folder, label)
        for filename in os.listdir(class_folder):
            if filename.endswith(".jpg"):
                img_path = os.path.join(class_folder, filename)
                images.append(img_path)
                labels.append(label)
    return images, labels

# 设定设备
device = "cuda" if torch.cuda.is_available() else 'cpu'

# 加载和准备模型
model = Resnet18(num_classes=2).to(device)  # 设置类别数为2
model.load_state_dict(torch.load("./save_model/quantized_m3_resnet18_epoch17_acc0.9562.pth"))
model.eval()

# 数据预处理
data_transform = transforms.Compose([
    transforms.Resize(224),  # 假设使用224x224作为输入尺寸
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化
])

# 加载测试数据集
folder_path = "./RailwayDefectDetectionDatabase_o/Test"  # 更新为您的文件夹路径
# folder_path = "./RailwayDefectDetectionDatabase/database/Test"  # 更新为您的文件夹路径
images, labels = load_images_from_folder(folder_path)

total_predictions, correct_predictions = 0, 0
# 进行预测
for img_path, true_label in zip(images, labels):
    img = Image.open(img_path).convert('RGB')
    img = data_transform(img)
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probabilities = nn.functional.softmax(outputs, dim=1)
        predicted = torch.argmax(probabilities, 1)
        predicted_label = 'Defective' if predicted == 0 else 'Non defective'
        predicted_probability = probabilities[0, predicted].item()

    # 累计统计
    correct_predictions += (predicted_label == true_label)
    total_predictions += 1

    print(f"Image: {os.path.basename(img_path)}, True label: {true_label}, Predicted label: {predicted_label}, Probability: {predicted_probability:.4f}")

# 输出总体预测的精准度
accuracy = correct_predictions / total_predictions
print(f'Overall Accuracy: {accuracy:.3f}')
