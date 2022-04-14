import numpy as np
from PIL import Image

# 转换图像
def cvtColor(image):
    image = image.convert('RGB')
    return image

# resize
def resize_image(image, size):
    result = image.resize(size, Image.BICUBIC)
    return result

# 获得类
def get_classes(class_path):
    with open(class_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

# 获取图像输入大小
def get_new_img_size(height, width, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_height, resized_width
