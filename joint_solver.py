import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torch.backends import cudnn
from networks.joint_poolnet import build_model, weights_init
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
            8,
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
    def test(self, test_mode=1):
        # 训练模式
        mode_name = ['edge_fuse', 'sal_fuse']
        EPSILON = 1e-8
        # 开始时间
        time_s = time.time()
        # images数量
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader):
            # 获取image数据和name
            images, name, im_size = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size'])
            # edge_fuse模式
            if test_mode == 0:
                # [channel, width, height] --> [height, width, channel]
                images = images.numpy()[0].transpose((1, 2, 0))
                # 多维度不同scale测试
                scale = [0.5, 1, 1.5, 2]
                # scale = [1]
                # 保存预测edge数据
                multi_fuse = np.zeros(im_size, np.float32)
                for k in range(0, len(scale)):
                    # 图像缩放
                    im_ = cv2.resize(images, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
                    # [height, width, channel] --> [channel, width, height]
                    im_ = im_.transpose((2, 0, 1))
                    # ----------------------------------------------------------------
                    im_ = torch.Tensor(im_[np.newaxis, ...])
                    # ----------------------------------------------------------------

                    with torch.no_grad():
                        # 获取tensor数据并搬运指定设备
                        images = torch.Tensor(images)
                        if self.config.cuda:
                            im_ = im_.cuda()
                        # 预测值
                        preds = self.net(im_, mode=test_mode)
                        pred_0 = np.squeeze(torch.sigmoid(preds[1][0]).cpu().data.numpy())
                        pred_1 = np.squeeze(torch.sigmoid(preds[1][1]).cpu().data.numpy())
                        pred_2 = np.squeeze(torch.sigmoid(preds[1][2]).cpu().data.numpy())
                        pred_fuse = np.squeeze(torch.sigmoid(preds[0]).cpu().data.numpy())

                        # 均值
                        pred = (pred_0 + pred_1 + pred_2 + pred_fuse) / 4
                        # min-max归一化
                        pred = (pred - np.min(pred) + EPSILON) / (np.max(pred) - np.min(pred) + EPSILON)

                        # 还原图像
                        pred = cv2.resize(pred, (im_size[1], im_size[0]), interpolation=cv2.INTER_LINEAR)
                        # 不同scale下累加
                        multi_fuse += pred

                # 创建image
                multi_fuse /= len(scale)
                # ----------------------------------------------------------------
                multi_fuse = 255 * (1 - multi_fuse)
                # ----------------------------------------------------------------
                cv2.imwrite(os.path.join(self.config.test_fold, name[:-4] + '_' + mode_name[test_mode] + '.png'), multi_fuse)
            # sal_fuse模式
            elif test_mode == 1:
                with torch.no_grad():
                    # 获取tensor数据并搬运指定设备
                    images = torch.Tensor(images)
                    if self.config.cuda:
                        images = images.cuda()
                    # 预测值
                    preds = self.net(images, mode=test_mode)
                    pred = np.squeeze(torch.sigmoid(preds).cpu().data.numpy())
                    # 创建image
                    multi_fuse = 255 * pred
                    cv2.imwrite(os.path.join(self.config.test_fold, name[:-4] + '_' + mode_name[test_mode] + '.png'), multi_fuse)
        # 结束时间
        time_e = time.time()
        print('Speed: %f FPS' % (img_num / (time_e - time_s)))
        print('Test Done!')

    # training phase
    def train(self):
        iter_num = 30000  # each batch only train 30000 iters.(This number is just a random choice...)
        # 积累梯度次数
        aveGrad = 0
        for epoch in range(self.config.epoch):
            r_edge_loss, r_sal_loss, r_sum_loss = 0, 0, 0
            # 梯度归零
            self.net.zero_grad()
            for i, data_batch in enumerate(self.train_loader):
                if (i + 1) == iter_num:
                    break
                # 获取image、mask和edge时image、mask数据
                edge_image, edge_label, sal_image, sal_label = data_batch['edge_image'], data_batch['edge_label'], data_batch['sal_image'], data_batch['sal_label']
                # image和mask尺寸不一致
                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue
                # 获取tensor数据并搬运指定设备
                edge_image, edge_label, sal_image, sal_label = Variable(edge_image), Variable(edge_label), Variable(sal_image), Variable(sal_label)
                if self.config.cuda:
                    cudnn.benchmark = True
                    edge_image, edge_label, sal_image, sal_label = edge_image.cuda(), edge_label.cuda(), sal_image.cuda(), sal_label.cuda()

                # edge part
                # 预测mask
                edge_pred = self.net(edge_image, mode=0)
                # 误差
                edge_loss_fuse = bce2d(edge_pred[0], edge_label, reduction='sum')
                edge_loss_part = []
                for ix in edge_pred[1]:
                    edge_loss_part.append(bce2d(ix, edge_label, reduction='sum'))
                edge_loss = (edge_loss_fuse + sum(edge_loss_part)) / (self.iter_size * self.config.batch_size)
                r_edge_loss += edge_loss.data

                # sal part
                # 预测mask
                sal_pred = self.net(sal_image, mode=1)
                # 误差
                sal_loss_fuse = F.binary_cross_entropy_with_logits(sal_pred, sal_label, reduction='sum')
                sal_loss = sal_loss_fuse / (self.iter_size * self.config.batch_size)
                r_sal_loss += sal_loss.data

                # 总误差
                loss = sal_loss + edge_loss
                r_sum_loss += loss.data

                # 向后传播
                loss.backward()

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
                    print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Edge : %10.4f  ||  Sal : %10.4f  ||  Sum : %10.4f' %
                          (epoch, self.config.epoch, i, iter_num, r_edge_loss / x_showEvery, r_sal_loss / x_showEvery, r_sum_loss / x_showEvery))
                    print('Learning rate: ' + str(self.lr))
                    r_edge_loss, r_sal_loss, r_sum_loss = 0, 0, 0

            # 保存训练模型
            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/models/epoch_%d.pth' % (self.config.save_folder, epoch + 1))

            # 学习率衰退
            if epoch in self.lr_decay_epoch:
                self.lr = self.lr * 0.1
                self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr, weight_decay=self.wd)

        # 保存训练模型
        torch.save(self.net.state_dict(), '%s/models/final.pth' % self.config.save_folder)


# balance binary cross entropy损失函数
def bce2d(input, target, reduction=None):
    assert (input.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg / num_total
    beta = 1.1 * num_pos / num_total
    # target pixel = 1 -> weight beta
    # target pixel = 0 -> weight 1-beta
    weights = alpha * pos + beta * neg

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)
