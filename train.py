import torch
from torch import nn
from resnet18 import Resnet18  # 确保 Resnet18 类正确导入
from torch.optim import lr_scheduler
from torchvision import transforms
import os
from tqdm import tqdm

from my_dataset import MyDataSet  # 导入自定义数据集类

# 更新数据预处理
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
val_images_path, val_images_label = read_data('./RailwayDefectDetectionDatabase/database/val')
# train_images_path, train_images_label = read_data('./RailwayDefectDetectionDatabase_o/train')
# val_images_path, val_images_label = read_data('./RailwayDefectDetectionDatabase_o/val')

train_dataset = MyDataSet(images_path=train_images_path,
                          images_class=train_images_label,
                          transform=data_transform["train"])
val_dataset = MyDataSet(images_path=val_images_path,
                        images_class=val_images_label,
                        transform=data_transform["val"])

train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=256, shuffle=False)

device = "cuda" if torch.cuda.is_available() else 'cpu'
model = Resnet18(num_classes=2).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

def train(dataloader, model, loss_fn, optimizer, epoch):
    model.train()
    loop = tqdm(dataloader, desc=f'\033[97m[Train epoch {epoch}]', total=len(dataloader), leave=True)
    for X, y in loop:
        X, y = X.to(device), y.to(device)
        output = model(X)
        loss = loss_fn(output, y)
        _, predicted = torch.max(output.data, 1)
        acc = (predicted == y).sum().item() / y.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item(), acc=acc)

def val(dataloader, model, loss_fn, epoch):
    model.eval()
    loop = tqdm(dataloader, desc=f'[Valid epoch {epoch}]', total=len(dataloader), leave=True)
    total_loss, total_acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for X, y in loop:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = loss_fn(output, y)
            _, predicted = torch.max(output.data, 1)
            acc = (predicted == y).sum().item() / y.size(0)

            total_loss += loss.item()
            total_acc += acc
            n += 1

            loop.set_postfix(loss=total_loss / n, acc=total_acc / n)

    return total_acc / n

epochs = 20
best_accuracy = 0.0
for epoch in range(epochs):
    train(train_dataloader, model, loss_fn, optimizer, epoch)
    accuracy = val(test_dataloader, model, loss_fn, epoch)
    lr_scheduler.step()
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        os.makedirs('save_model', exist_ok=True)
        torch.save(model.state_dict(), f'save_model/m7_resnet18_epoch{epoch}_acc{accuracy:.4f}.pth')
        tqdm.write('Saving best model')  # 使用 tqdm.write 来避免打乱进度条的显示
print('Training complete.')
