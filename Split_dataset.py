import os
import shutil

data_path = './datasets/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/'
train_path = './datasets/CT_Kidney_dataset/train/'
test_path = './datasets/CT_Kidney_dataset/test/'
val_path = './datasets/CT_Kidney_dataset/val/'

train_r = 7
val_r = 1
test_r = 2

round = train_r + val_r + test_r

os.mkdir(train_path)
os.mkdir(test_path)
os.mkdir(val_path)

for dir_name in os.listdir(data_path):
    dir_path = os.path.join(data_path, dir_name)

    if not os.path.isdir(dir_path):
        continue

    train_copy_path = os.path.join(train_path, dir_name)
    test_copy_path = os.path.join(test_path, dir_name)
    val_copy_path = os.path.join(val_path, dir_name)
    print(train_copy_path)
    print(test_copy_path)
    print(val_copy_path)

    if not os.path.exists(train_copy_path):
        os.mkdir(train_copy_path)
    if not os.path.exists(test_copy_path):
        os.mkdir(test_copy_path)
    if not os.path.exists(val_copy_path):
        os.mkdir(val_copy_path)

    count = 0

    for filename in os.listdir(dir_path):
        if filename.endswith('.jpg'):
            img_path = os.path.join(dir_path, filename)

            count = (count + 1) % round

            if count < train_r:
                copy_path = os.path.join(train_path, dir_name)
                copy_path = os.path.join(copy_path, filename)
                shutil.copy(img_path, copy_path)
            elif (train_r <= count) and (count < train_r + val_r):
                copy_path = os.path.join(val_path, dir_name)
                copy_path = os.path.join(copy_path, filename)
                shutil.copy(img_path, copy_path)
            else:
                copy_path = os.path.join(test_path, dir_name)
                copy_path = os.path.join(copy_path, filename)
                shutil.copy(img_path, copy_path)

