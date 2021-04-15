import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.backends import cudnn
from networks.poolnet import build_model, weights_init
import numpy as np
import os
import cv2
import time


class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        # 训练集DataLoader
        self.train_loader = train_loader
        # 测试集DataLoader
        self.test_loader = test_loader
        # config配置
        self.config = config
        # 积累梯度迭代次数
        self.iter_size = config.iter_size
        # 展示信息epoch次数
        self.show_every = config.show_every
        # 学习率衰退epoch数
        self.lr_decay_epoch = [
            15,
        ]
        # 创建模型
        self.build_model()
        # 进入test模式
        if config.mode == 'test':
            print('Loading pre-trained model from %s...' % self.config.model)
            # 载入预训练模型并放入相应位置
            if self.config.cuda:
                self.net.load_state_dict(torch.load(self.config.model))
            else:
                self.net.load_state_dict(torch.load(self.config.model, map_location='cpu'))
            # 设置eval模式
            self.net.eval()

    # 打印网络信息和参数数量
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    # 建立模型
    def build_model(self):
        self.net = build_model(self.config.arch)
        # 是否将网络搬运至cuda
        if self.config.cuda:
            self.net = self.net.cuda()
        # self.net.train()
        # 设置eval状态
        self.net.eval()  # use_global_stats = True
        # 网络权重初始化
        self.net.apply(weights_init)
        # 载入预训练模型或自行训练模型
        if self.config.load == '':
            self.net.base.load_pretrained_model(torch.load(self.config.pretrained_model))
        else:
            self.net.load_state_dict(torch.load(self.config.load))

        # 学习率
        self.lr = self.config.lr
        # 权值衰减
        self.wd = self.config.wd

        # 设置优化器
        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr, weight_decay=self.wd)
        # 打印网络结构
        self.print_network(self.net, 'PoolNet Structure')

    # testing状态
    def test(self):
        # 训练模式
        mode_name = 'sal_fuse'
        # 开始时间
        time_s = time.time()
        # images数量
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader):
            # 获取image数据和name
            images, name = data_batch['image'], data_batch['name'][0]
            # testing状态
            with torch.no_grad():
                # 获取tensor数据并搬运指定设备
                images = torch.Tensor(images)
                if self.config.cuda:
                    images = images.cuda()
                # 预测值
                preds = self.net(images)
                pred = np.squeeze(torch.sigmoid(preds).cpu().data.numpy())
                # 创建image
                multi_fuse = 255 * pred
                cv2.imwrite(os.path.join(self.config.test_fold, name[:-4] + '_' + mode_name + '.png'), multi_fuse)
        # 结束时间
        time_e = time.time()
        print('Speed: %f FPS' % (img_num / (time_e - time_s)))
        print('Test Done!')

    # training状态
    def train(self):
        # 迭代次数
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        # 积累梯度次数
        aveGrad = 0
        for epoch in range(self.config.epoch):
            r_sal_loss = 0
            # 梯度归零
            self.net.zero_grad()
            for i, data_batch in enumerate(self.train_loader):
                # 获取image和mask数据
                sal_image, sal_label = data_batch['sal_image'], data_batch['sal_label']
                # image和mask尺寸不一致
                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue
                # 获取tensor数据并搬运指定设备
                sal_image, sal_label = torch.Tensor(sal_image), torch.Tensor(sal_label)
                if self.config.cuda:
                    cudnn.benchmark = True
                    sal_image, sal_label = sal_image.cuda(), sal_label.cuda()

                # 预测mask
                sal_pred = self.net(sal_image)
                # 误差
                sal_loss_fuse = F.binary_cross_entropy_with_logits(sal_pred, sal_label, reduction='sum')
                sal_loss = sal_loss_fuse / (self.iter_size * self.config.batch_size)
                r_sal_loss += sal_loss.data

                # 向后传播
                sal_loss.backward()

                # 积累梯度次数
                aveGrad += 1

                # accumulate gradients as done in DSS
                if aveGrad % self.iter_size == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    aveGrad = 0

                # 展示此时信息
                if i % (self.show_every // self.config.batch_size) == 0:
                    if i == 0:
                        x_showEvery = 1
                    print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Sal : %10.4f' % (epoch, self.config.epoch, i, iter_num, r_sal_loss / x_showEvery))
                    print('Learning rate: ' + str(self.lr))
                    r_sal_loss = 0

            # 保存训练模型
            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/models/epoch_%d.pth' % (self.config.save_folder, epoch + 1))

            # 学习率衰退
            if epoch in self.lr_decay_epoch:
                self.lr = self.lr * 0.1
                self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr, weight_decay=self.wd)

        # 保存训练模型
        torch.save(self.net.state_dict(), '%s/models/final.pth' % self.config.save_folder)
