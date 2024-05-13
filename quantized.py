import torch
import torch.nn as nn
from torchvision import transforms
import torch.utils.data
from resnet18 import QuantizedResnet18  # Make sure your Resnet18 is imported correctly
from my_dataset import MyDataSet  # Import your custom dataset class
import os

# Data transformations
data_transform = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}

def read_data(data_path):
    images_path = []
    images_label = []
    for class_dir in os.listdir(data_path):
        class_dir_path = os.path.join(data_path, class_dir)
        for img_file in os.listdir(class_dir_path):
            images_path.append(os.path.join(class_dir_path, img_file))
            images_label.append(class_dir)
    return images_path, images_label

train_images_path, train_images_label = read_data('./RailwayDefectDetectionDatabase/database/train')
train_dataset = MyDataSet(images_path=train_images_path, images_class=train_images_label, transform=data_transform["train"])

# DataLoader for calibration
calibrate_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Device configuration
device = "cuda" if torch.cuda.is_available() else 'cpu'

# Load your trained ResNet18 model
model = QuantizedResnet18(num_classes=2).to(device)
model.load_state_dict(torch.load("save_model/m3_resnet18_epoch17_acc0.9562.pth"))
model.eval()

# Set the quantization configuration to default
model.qconfig = torch.quantization.default_qconfig
print("Model quantization config: ", model.qconfig)

# Prepare the model for quantization
torch.quantization.prepare(model, inplace=True)

# Calibrate with representative dataset
def calibrate_model(model, calibrate_loader):
    model.eval()
    with torch.no_grad():
        for batch, _ in calibrate_loader:
            batch = batch.to(device)
            model(batch)

calibrate_model(model, calibrate_loader)

# Convert the model to a quantized version
torch.quantization.convert(model, inplace=True)

# Save the quantized model
torch.save(model.state_dict(), "save_model/quantized_m3_resnet18_epoch17_acc0.9562.pth")
