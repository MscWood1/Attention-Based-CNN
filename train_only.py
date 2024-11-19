import torch
import torch.nn as nn
import os
import csv
from time import time
from datetime import datetime
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

from CNN_only_num_4 import *

'''cfg'''

train_path = './datasets/Alzheimer_s Dataset/train'
val_path = './datasets/Alzheimer_s Dataset/test'
num_epochs = 20

'''cfg end'''


# set random seed
torch.manual_seed(42)

# data preprocess
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# load train and val data
train_dataset = ImageFolder(train_path, transform=transform)
val_dataset = ImageFolder(val_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = FourLayerCNN()

# optm
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

best_val_acc = 0.0
best_epoch = 0
train_loss = []
val_loss = []
train_acc = []
val_acc = []

for i in range(num_epochs):
    train_loss.append(0.0)
    val_loss.append(0.0)
    train_acc.append(0.0)
    val_acc.append(0.0)


current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
save_path = './model/only_' + current_time + '/'
model_path = save_path + 'models/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

if not os.path.exists(model_path):
    os.makedirs(model_path)

start = time()

for epoch in range(num_epochs):
    correct_train = 0
    total_train = 0

    correct_val = 0
    total_val = 0

    # train
    print(f'train: epoch {epoch+1}')

    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss[epoch] += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    print(f'end train')

    # eval
    print(f'eval: epoch {epoch+1}')

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss[epoch] += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    print(f'end eval')

    # calculate acc
    train_loss[epoch] = train_loss[epoch] / len(train_loader.dataset)
    val_loss[epoch] = val_loss[epoch] / len(val_loader.dataset)
    train_acc[epoch] = correct_train / total_train * 100
    val_acc[epoch] = correct_val / total_val * 100

    print(
        f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss[epoch]:.4f} - Val Loss: {val_loss[epoch]:.4f} - Train Acc: {train_acc[epoch]:.2f}% - Val Acc: {val_acc[epoch]:.2f}%')

    torch.save(model.state_dict(), model_path + 'model_epoch_' + str(epoch+1) + '_val_acc_' + str(val_acc[epoch]) + '.pth')

    #save model
    if val_acc[epoch] > best_val_acc:
        print(f"Saving model: best_val_acc is {val_acc[epoch]}")
        best_val_acc = val_acc[epoch]
        best_epoch = epoch + 1
        torch.save(model.state_dict(), save_path + 'best_model.pth')

    if val_acc[epoch] == 100 and train_acc[epoch] == 100:
        break

end = time()
print(f"Train end: the Best Val Acc is {best_val_acc:.2f}% in epoch {best_epoch}")
print("total train time: " + str(end - start))

# save train logs

os.makedirs(save_path + 'logs')

train_acc_file = save_path + 'logs/train_acc_logs.csv'
data = [[x] for x in train_acc]

with open(train_acc_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

print("Save train acc log in " + train_acc_file)

val_acc_file = save_path + 'logs/val_acc_logs.csv'
data = [[x] for x in val_acc]

with open(val_acc_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

print("Save val acc log in " + val_acc_file)

train_loss_file = save_path + 'logs/train_loss_logs.csv'
data = [[x] for x in train_loss]

with open(train_loss_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

print("Save train loss log in " + train_loss_file)

val_loss_file = save_path + 'logs/val_loss_logs.csv'
data = [[x] for x in val_loss]

with open(val_loss_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

print("Save val loss log in " + val_loss_file)