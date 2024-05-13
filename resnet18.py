import torch
import torch.nn as nn

#残差网络
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()

        if stride!=1 or in_channels!=out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out


# # m1
# class Resnet18(nn.Module):
#     def __init__(self, num_classes=2):
#         super(Resnet18, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         self.layer1 = self._make_layer(64, 128, 2, stride=2)
#         self.layer2 = self._make_layer(128, 256, 2, stride=2)
#         self.layer3 = self._make_layer(256, 512, 2, stride=2)
#         # self.layer4 = self._make_layer(256, 512, 2, stride=2)
#
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#
#         self.fc = nn.Linear(512, num_classes)
#
#     # 搭建层函数
#     def _make_layer(self, in_channels, out_channels, blocks, stride=1):
#         layer = []
#         layer.append(ResidualBlock(in_channels, out_channels, stride))
#         for _ in range(1, blocks):
#             layer.append(ResidualBlock(out_channels, out_channels))
#         return nn.Sequential(*layer)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.maxpool(out)
#
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#
#         out = self.avgpool(out)
#         out = torch.flatten(out, 1)
#         out = self.fc(out)
#
#         return out

# # m2
# class Resnet18(nn.Module):
#     def __init__(self, num_classes=2):
#         super(Resnet18, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         self.layer1 = self._make_layer(64, 128, 2, stride=2)
#         self.layer2 = self._make_layer(128, 256, 2, stride=2)
#         # self.layer3 = self._make_layer(128, 256, 2, stride=2)
#         # self.layer4 = self._make_layer(256, 512, 2, stride=2)
#
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#
#         self.fc = nn.Linear(256, num_classes)
#
#     # 搭建层函数
#     def _make_layer(self, in_channels, out_channels, blocks, stride=1):
#         layer = []
#         layer.append(ResidualBlock(in_channels, out_channels, stride))
#         for _ in range(1, blocks):
#             layer.append(ResidualBlock(out_channels, out_channels))
#         return nn.Sequential(*layer)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.maxpool(out)
#
#         out = self.layer1(out)
#         out = self.layer2(out)
#
#         out = self.avgpool(out)
#         out = torch.flatten(out, 1)
#         out = self.fc(out)
#
#         return out

# m3
class Resnet18(nn.Module):
    def __init__(self, num_classes=2):
        super(Resnet18, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=7, stride=1, padding=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(256, num_classes)

    # 搭建层函数
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layer = []
        layer.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layer.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layer)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

class QuantizedResnet18(nn.Module):
    def __init__(self, num_classes=2):
        super(QuantizedResnet18, self).__init__()
        self.quant = torch.quantization.QuantStub()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)

        self.conv2 = nn.Conv2d(128, 256, kernel_size=7, stride=1, padding=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

        self.dequant = torch.quantization.DeQuantStub()

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        x = self.dequant(x)
        return x

# # m4
# class Resnet18(nn.Module):
#     def __init__(self, num_classes=2):
#         super(Resnet18, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         self.layer1 = self._make_layer(64, 128, 2, stride=2)
#         self.layer2 = self._make_layer(128, 256, 2, stride=2)
#         # self.layer3 = self._make_layer(128, 256, 2, stride=2)
#         # self.layer4 = self._make_layer(256, 512, 2, stride=2)
#
#         # self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=7, stride=1, padding=1, bias=False)
#         # self.relu2 = nn.ReLU(inplace=True)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#
#         self.fc = nn.Linear(256, num_classes)
#
#     # 搭建层函数
#     def _make_layer(self, in_channels, out_channels, blocks, stride=1):
#         layer = []
#         layer.append(ResidualBlock(in_channels, out_channels, stride))
#         for _ in range(1, blocks):
#             layer.append(ResidualBlock(out_channels, out_channels))
#         return nn.Sequential(*layer)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.maxpool(out)
#
#         out = self.layer1(out)
#         out = self.layer2(out)
#         # out = self.layer3(out)
#         # out = self.layer4(out)
#
#         # out = self.conv2(out)
#         # out = self.relu2(out)
#         out = self.avgpool(out)
#         out = torch.flatten(out, 1)
#         out = self.fc(out)
#
#         return out

# # m5
# class Resnet18(nn.Module):
#     def __init__(self, num_classes=2):
#         super(Resnet18, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         self.layer1 = self._make_layer(64, 128, 2, stride=2)
#
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#
#         self.fc = nn.Linear(128, num_classes)
#
#     # 搭建层函数
#     def _make_layer(self, in_channels, out_channels, blocks, stride=1):
#         layer = []
#         layer.append(ResidualBlock(in_channels, out_channels, stride))
#         for _ in range(1, blocks):
#             layer.append(ResidualBlock(out_channels, out_channels))
#         return nn.Sequential(*layer)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.maxpool(out)
#
#         out = self.layer1(out)
#
#         out = self.avgpool(out)
#         out = torch.flatten(out, 1)
#         out = self.fc(out)
#
#         return out

# # m6
# class Resnet18(nn.Module):
#     def __init__(self, num_classes=2):
#         super(Resnet18, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         self.layer1 = self._make_layer(64, 64, 1, stride=1)
#         self.layer2 = self._make_layer(64, 128, 2, stride=2)
#
#         self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=7, stride=1, padding=1, bias=False)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#
#         self.fc = nn.Linear(256, num_classes)
#
#     # 搭建层函数
#     def _make_layer(self, in_channels, out_channels, blocks, stride=1):
#         layer = []
#         layer.append(ResidualBlock(in_channels, out_channels, stride))
#         for _ in range(1, blocks):
#             layer.append(ResidualBlock(out_channels, out_channels))
#         return nn.Sequential(*layer)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.maxpool(out)
#
#         out = self.layer1(out)
#         out = self.layer2(out)
#
#         out = self.conv2(out)
#         out = self.relu2(out)
#         out = self.avgpool(out)
#         out = torch.flatten(out, 1)
#         out = self.fc(out)
#
#         return out
