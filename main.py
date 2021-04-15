import argparse
import os
from dataset.dataset import get_loader
from solver import Solver


# 根据不同模式选择数据集路径
def get_test_info(sal_mode='e'):
    if sal_mode == 'e':
        image_root = './data/ECSSD/Imgs/'
        image_source = './data/ECSSD/test.lst'
    elif sal_mode == 'p':
        image_root = './data/PASCALS/Imgs/'
        image_source = './data/PASCALS/test.lst'
    elif sal_mode == 'd':
        image_root = './data/DUTOMRON/Imgs/'
        image_source = './data/DUTOMRON/test.lst'
    elif sal_mode == 'h':
        image_root = './data/HKU-IS/Imgs/'
        image_source = './data/HKU-IS/test.lst'
    elif sal_mode == 's':
        image_root = './data/SOD/Imgs/'
        image_source = './data/SOD/test.lst'
    elif sal_mode == 't':
        image_root = './data/DUTS-TE/Imgs/'
        image_source = './data/DUTS-TE/test.lst'
    elif sal_mode == 'm_r':  # for speed test
        image_root = './data/MSRA/Imgs_resized/'
        image_source = './data/MSRA/test_resized.lst'
    elif sal_mode == 'mydataset':  # for speed test
        image_root = './data/mydataset/Image/'
        image_source = './data/mydataset/test.lst'

    return image_root, image_source


def main(config):
    # 选择使用模式
    if config.mode == 'train':
        # 获取DataLoader
        train_loader = get_loader(config)
        # 计数, 找到可以创建文件夹的名字
        run = 0
        while os.path.exists("%s/run-%d" % (config.save_folder, run)):
            run += 1
        # 创建保存模型文件夹
        os.mkdir("%s/run-%d" % (config.save_folder, run))
        os.mkdir("%s/run-%d/models" % (config.save_folder, run))
        # 在config中保存文件夹名称
        config.save_folder = "%s/run-%d" % (config.save_folder, run)
        # 开始进入training状态
        train = Solver(train_loader, None, config)
        train.train()
    elif config.mode == 'test':
        # 选择数据集路径
        config.test_root, config.test_list = get_test_info(config.sal_mode)
        # 获取DataLoader
        test_loader = get_loader(config, mode='test')
        # 如果文件路径不存在
        if not os.path.exists(config.test_fold):
            os.mkdir(config.test_fold)
        # 开始进入testing状态
        test = Solver(None, test_loader, config)
        test.test()
    else:
        # 非法输入
        raise IOError("illegal input!!!")


if __name__ == '__main__':
    # 权重路径
    vgg_path = './dataset/pretrained/vgg16_20M.pth'
    resnet_path = './dataset/pretrained/resnet50_caffe.pth'

    # 命令行参数初始化
    parser = argparse.ArgumentParser()

    # 超参数
    # 颜色数量
    parser.add_argument('--n_color', type=int, default=3)
    # 学习率 resnet 5e-5 or vgg 1e-4
    parser.add_argument('--lr', type=float, default=1e-4)
    # 权重衰退
    parser.add_argument('--wd', type=float, default=0.0005)
    # 是否使用cuda
    parser.add_argument('--no-cuda', dest='cuda', action='store_false')

    # 训练过程参数
    # 网络结构 resnet or vgg
    parser.add_argument('--arch', type=str, default='vgg')
    # 预训练权重模型
    parser.add_argument('--pretrained_model', type=str, default=vgg_path)
    # 训练次数
    parser.add_argument('--epoch', type=int, default=24)
    # batch数
    parser.add_argument('--batch_size', type=int, default=1)
    # 线程数量
    parser.add_argument('--num_thread', type=int, default=1)
    # 载入预训练模型或自行训练模型
    parser.add_argument('--load', type=str, default='')
    # 保存训练模型文件夹
    parser.add_argument('--save_folder', type=str, default='./results')
    # 每规定epoch保存一个训练模型文件
    parser.add_argument('--epoch_save', type=int, default=3)
    # 迭代次数
    parser.add_argument('--iter_size', type=int, default=10)
    # 展示信息epoch次数
    parser.add_argument('--show_every', type=int, default=50)

    # 训练数据集
    parser.add_argument('--train_root', type=str, default='')
    parser.add_argument('--train_list', type=str, default='')

    # 测试过程参数
    # 模型
    parser.add_argument('--model', type=str, default=None)
    # 测试结果保存文件夹
    parser.add_argument('--test_fold', type=str, default=None)
    # image测试数据集
    parser.add_argument('--sal_mode', type=str, default='t')

    # 运行模式
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    # 将参数保存至config
    config = parser.parse_args()

    # 训练模型文件夹路径是否存在
    if not os.path.exists(config.save_folder):
        os.mkdir(config.save_folder)

    # 获取测试集信息
    test_root, test_list = get_test_info(config.sal_mode)
    config.test_root = test_root
    config.test_list = test_list

    # 运行main函数
    main(config)
