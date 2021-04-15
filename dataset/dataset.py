import os
from PIL import Image
import cv2
import torch
from torch.utils import data
import numpy as np
import random


# 训练集图像类
class ImageDataTrain(data.Dataset):
    def __init__(self, data_root, data_list):
        # images和masks文件路径
        self.sal_root = data_root
        # image和mask匹配列表文件路径
        self.sal_source = data_list

        # 打开image和mask匹配列表文件, 将image和mask文件路径读取
        with open(self.sal_source, 'r') as f:
            self.sal_list = [x.strip() for x in f.readlines()]

        # image和mask对的数量
        self.sal_num = len(self.sal_list)

    def __getitem__(self, item):
        # image文件路径
        im_name = self.sal_list[item % self.sal_num].split()[0]
        # mask文件路径
        gt_name = self.sal_list[item % self.sal_num].split()[1]
        # 获取image数据
        sal_image = load_image(os.path.join(self.sal_root, im_name))
        # 获取mask数据
        sal_label = load_sal_label(os.path.join(self.sal_root, gt_name))
        # 概率50%随机翻转
        sal_image, sal_label = cv_random_flip(sal_image, sal_label)
        # 将image和mask转换成tensor
        sal_image = torch.Tensor(sal_image)
        sal_label = torch.Tensor(sal_label)

        # 将image和mask以字典形式储存
        sample = {'sal_image': sal_image, 'sal_label': sal_label}
        return sample

    def __len__(self):
        # 数据集长度
        return self.sal_num


# 测试集图像类
class ImageDataTest(data.Dataset):
    def __init__(self, data_root, data_list):
        # images和masks文件路径
        self.data_root = data_root
        self.data_list = data_list
        # 打开image和mask匹配列表文件, 将image和mask文件路径读取
        with open(self.data_list, 'r') as f:
            self.image_list = [x.strip() for x in f.readlines()]

        # image和mask对的数量
        self.image_num = len(self.image_list)

    def __getitem__(self, item):
        # 获取image数据和image尺寸
        image, im_size = load_image_test(os.path.join(self.data_root, self.image_list[item]))
        # 将image转换成tensor
        image = torch.Tensor(image)

        return {'image': image, 'name': self.image_list[item % self.image_num], 'size': im_size}

    def __len__(self):
        # 数据集长度
        return self.image_num


# 获取DataLoader
def get_loader(config, mode='train', pin=False):
    # 定义不同模式下的DataLoader
    if mode == 'train':
        shuffle = True
        dataset = ImageDataTrain(config.train_root, config.train_list)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_thread, pin_memory=pin)
    else:
        shuffle = False
        dataset = ImageDataTest(config.test_root, config.test_list)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_thread, pin_memory=pin)
    return data_loader


# 载入image
def load_image(path):
    # 判断image文件是否存在
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    # 返回一个[height, width, channel]的numpy.ndarray的image对象
    im = cv2.imread(path)
    # 数据类型转换
    in_ = np.array(im, dtype=np.float32)
    # 数据预处理, 减去image数据均值
    in_ -= np.array((104.00699, 116.66877, 122.67892))
    # [height, width, channel] --> [channel, width, height]
    in_ = in_.transpose((2, 0, 1))
    return in_


def load_image_test(path):
    # 判断image文件是否存在
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    # 返回一个[height, width, channel]的numpy.ndarray的image对象
    im = cv2.imread(path)
    # 数据类型转换
    in_ = np.array(im, dtype=np.float32)
    # image尺寸
    im_size = tuple(in_.shape[:2])
    # 数据预处理, 减去image数据均值
    in_ -= np.array((104.00699, 116.66877, 122.67892))
    # [height, width, channel] --> [channel, width, height]
    in_ = in_.transpose((2, 0, 1))
    return in_, im_size


def load_sal_label(path):
    # 判断mask文件是否存在
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    # 打开mask文件
    im = Image.open(path)
    # 数据类型转换
    label = np.array(im, dtype=np.float32)
    # ----------------------------------------------------------------
    # 只有RGB色彩空间中的R
    if len(label.shape) == 3:
        label = label[:, :, 0]
    # ----------------------------------------------------------------
    # 数据预处理, mask数据归一化
    label = label / 255.
    # ----------------------------------------------------------------
    label = label[np.newaxis, ...]
    # ----------------------------------------------------------------
    return label


# 随机翻转
def cv_random_flip(img, label):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img[:, :, ::-1].copy()
        label = label[:, :, ::-1].copy()
    return img, label
