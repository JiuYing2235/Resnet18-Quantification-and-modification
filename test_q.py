import torch
from torchvision import transforms
from torch import nn
from resnet18 import Resnet18  # Ensure Resnet18 is correctly imported
from PIL import Image
import os

def load_images_from_folder(folder):
    images = []
    labels = []
    for label in ['Defective', 'Non defective']:  # Example labels
        class_folder = os.path.join(folder, label)
        for filename in os.listdir(class_folder):
            if filename.endswith(".jpg"):
                img_path = os.path.join(class_folder, filename)
                images.append(img_path)
                labels.append(label)
    return images, labels

device = "cuda" if torch.cuda.is_available() else 'cpu'

# Load the quantized model structure
model = Resnet18(num_classes=2).to(device)

# Set the quantization configuration to per-tensor affine (widely supported)
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
print(model.qconfig)  # Print the quantization configuration

# Prepare the model for quantization
torch.quantization.prepare(model, inplace=True)

# Dummy calibration data loop (replace with your actual calibration dataset loader)
calibration_data_loader = torch.utils.data.DataLoader(
    # Your dataset here with correct transform
    batch_size=10  # Example batch size
)
# Calibrate the model with the data
for images, _ in calibration_data_loader:
    model(images.to(device))

# Convert the model to a quantized version
torch.quantization.convert(model, inplace=True)

# Load state dict (ensure this state dict is compatible with the quantized model)
model.load_state_dict(torch.load("./save_model/quantized_m3_resnet18_epoch17_acc0.9562.pth"))
model.eval()

# Data transformation setup
data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load images for testing
folder_path = "./RailwayDefectDetectionDatabase_o/Test"  # Adjust as needed
images, labels = load_images_from_folder(folder_path)

# Initialize counters for accuracy computation
total_predictions, correct_predictions = 0, 0

# Test the model with loaded images
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

    correct_predictions += (predicted_label == true_label)
    total_predictions += 1
    print(f"Image: {os.path.basename(img_path)}, True label: {true_label}, Predicted label: {predicted_label}, Probability: {predicted_probability:.4f}")

# Print overall accuracy
accuracy = correct_predictions / total_predictions
print(f'Overall Accuracy: {accuracy:.3f}')
