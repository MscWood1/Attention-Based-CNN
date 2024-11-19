import torch
import torch.nn as nn
from Pool_fuc import Ratio_Pool
from torchsummary import summary
from heatmap import get_heatmap, get_heatmap_att
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()

        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1) # 32 * 81 * 16
        key = self.key_conv(x).view(batch_size, -1, height * width)                      # 32 * 16 * 81
        value = self.value_conv(x).view(batch_size, -1, height * width)                  # 32 * 81 * 128

        attention_map = torch.bmm(query, key)                                            # 32 * 81 * 81
        attention_map = torch.softmax(attention_map, dim=-1)

        attention_features = torch.bmm(value, attention_map.permute(0, 2, 1))            # 32 * 81 * 128 -> 32 * 128 * 81
        attention_features = attention_features.view(batch_size, channels, height, width)# 32 * 128 * 9 * 9

        out = self.gamma * attention_features + x
        return out


class FourLayerCNN(nn.Module):
    def __init__(self):
        super(FourLayerCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.attention = SelfAttention(128)

        self.dropout = nn.Dropout(p=0.02)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 9 * 9, 512)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(512, 128)
        self.relu6 = nn.ReLU()
        self.fc3 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)

        get_heatmap(x)
        x = self.attention(x)
        get_heatmap_att(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu6(x)
        x = self.fc3(x)

        return x

if __name__ == '__main__':
    model = FourLayerCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    summary(model, (3, 150, 150))