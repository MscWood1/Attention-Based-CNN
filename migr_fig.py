from PIL import Image
import numpy as np

# 创建一个16x8的空白大图
combined_image = Image.new('RGB', (16 * 900, 8 * 900))
num = 0
# 循环遍历每个小图，并将其放置到大图中对应的位置
for i in range(16):
    for j in range(8):
        # 读取小图
        path = f'./heatmap/'
        image_path = path + 'heatmap_att_' + str(num) + '.png'
        small_image = Image.open(image_path)

        # 将小图放置到大图中
        combined_image.paste(small_image, (i * 900, j * 900))
        num += 1

# 保存合并后的大图
combined_image.save('./heatmap/BIG_image_att.jpg')