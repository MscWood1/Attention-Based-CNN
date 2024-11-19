import os

import torch
import torch.nn as nn
from CNN_num_4 import *
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader


'''cfg'''

mode = 'all'                                       # mode: best / all
test_path = './datasets/test/'
savemodel_path = './model/CNN_2023-07-18-09-12_P/'

'''cfg end'''

if mode == 'all':
    savemodel_path = savemodel_path + 'models/'

# data preprocess
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# load test data
test_dataset = ImageFolder(test_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# load model
best_acc = 0.0
best_model_name = []

for model_name in os.listdir(savemodel_path):

    if not model_name.endswith('.pth'):
        continue

    model_path = savemodel_path + model_name

    print("Load model from " + model_path)
    model_state_dict = torch.load(model_path)
    model = FourLayerCNN()
    model.load_state_dict(model_state_dict)

    # test
    device = torch.device('cpu')
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    correct = 0
    total = 0
    score = 0.0

    print("Start test for data in " + test_path)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            # print(outputs.data[:])
            soft_output = nn.functional.softmax(outputs, dim=1)
            max_output, _ = torch.max(soft_output, dim=1)
            batch_score = torch.sum(max_output)
            score += batch_score.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = correct / total * 100
    score = score / total * 100
    print(f'Test Accuracy: {test_acc:.2f}%\n')
    print(f'Test Score: {score:.4f}\n')

    if test_acc >= best_acc and mode == 'all':
        if test_acc == best_acc:
            best_model_name = best_model_name + ' / ' + model_name
        else:
            best_model_name = model_name
        best_acc = test_acc

if mode == 'all':
    print(f'Best test acc is {best_acc}%')
    print(f'Best models is: {best_model_name}')
