import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(10, 5)  # 假设有一个线性层

    def forward(self, x):
        # 在这里对输入张量乘以一个系数
        x = x * 0.5
        x = self.fc(x)
        return x

# 使用示例
model = MyModel()
input_tensor = torch.randn(1, 10)
output_tensor = model(input_tensor)
print(output_tensor)