import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data.sampler as sampler

from SENet import *
from PAFEM import *

from convolution import *
from KANConv import *
from KANLinear import *

import argparse
from create_dataset import *
from utils import *

# imtl 所需
from imtl import IMTL
import itertools


# 配置了 argpars 模块就可以在终端命令行中传入参数(按照顺序)
parser = argparse.ArgumentParser(description='Multi-task: Attention Network')
parser.add_argument('--weight', default='equal', type=str, help='multi-task weighting: equal, uncert, dwa')
parser.add_argument('--dataroot', default='nyuv2', type=str, help='dataset root')
parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')
parser.add_argument('--apply_augmentation', action='store_true', help='toggle to apply data augmentation on NYUv2')

parser.add_argument('--method', default='pcgrad', type=str, help='which optimization algorithm to use')

opt = parser.parse_args()


class KVMSANet(nn.Module):
    def __init__(self):
        super(KVMSANet, self).__init__()
        # initialise network parameters
        filter = [8, 16, 32,64]

        # 专门适应 decoder_SE_block 中通道数
        dechannel = [512, 256, 128, 64, 64]

        # 专门适应 encoder_PAFE_block 中通道数
        PAFE_en_channel = [64, 128, 256, 512, 512, 512]
        # 专门适应 decoder_PAFE_block 中通道数
        PAFE_de_channel = [512, 512, 256, 128, 64, 64]
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.class_nb = 7

        # define encoder decoder layers
        self.encoder_block = nn.ModuleList([self.conv_layer([3, filter[0]])])
        self.decoder_block = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(3):
            self.encoder_block.append(self.conv_layer([filter[i], filter[i + 1]]))
            self.decoder_block.append(self.conv_layer([filter[i + 1], filter[i]]))

        # SE模块
        self.encoder_SE_block = nn.ModuleList([SEModule(filter[0])])
        self.decoder_SE_block = nn.ModuleList([SEModule(dechannel[0])])
        for i in range(3):
            self.encoder_SE_block.append(SEModule(filter[i + 1]))
            self.decoder_SE_block.append(SEModule(dechannel[i + 1]))

        # PAFE模块
        self.encoder_PAFE_block = nn.ModuleList([PAFEM(PAFE_en_channel[0], PAFE_en_channel[1])])
        self.decoder_PAFE_block = nn.ModuleList([PAFEM(PAFE_de_channel[4], PAFE_de_channel[5])])
        for i in range(3):
            self.encoder_PAFE_block.append(PAFEM(PAFE_en_channel[i + 1], PAFE_en_channel[i + 2]))
            self.decoder_PAFE_block.append(PAFEM(PAFE_de_channel[-i - 3], PAFE_de_channel[-i - 2]))
        
        
        # define convolution layer
        self.conv_block_enc = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        self.conv_block_dec = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(3):
            self.conv_block_enc.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
            self.conv_block_dec.append(self.conv_layer([filter[i], filter[i]]))

        
        
        # self.kanconv = KAN_Convolutional_Layer(n_convs = 1,padding = (1,1),kernel_size= (3,3),device = device)
        
        # 用 Kancnv 替换全部的 conv_block_enc 会导致显存不够(故而注释掉)
        self.kconv_encoder_block_att = nn.ModuleList([KAN_Convolutional_Layer(n_convs = 2,padding = (1,1),kernel_size= (3,3),device = device)])
        self.kconv_encoder_block_att.append(KAN_Convolutional_Layer(n_convs = 2,padding = (1,1),kernel_size= (3,3),device = device))
        self.kconv_encoder_block_att.append(KAN_Convolutional_Layer(n_convs = 2,padding = (1,1),kernel_size= (3,3),device = device))
        self.kconv_encoder_block_att.append(KAN_Convolutional_Layer(n_convs = 1,padding = (1,1),kernel_size= (3,3),device = device))
        
        '''
        # define convolution layer (只有编码部分，对比上面注释代码，此举为了加入 KanConv) 还是显存不够
        self.conv_block_enc = nn.ModuleList([KAN_Convolutional_Layer(n_convs = 1,padding = (1,1),kernel_size= (3,3),device = device)])
        self.conv_block_dec = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            if i == 0:
                self.conv_block_dec.append(self.conv_layer([filter[i], filter[i]]))
                self.conv_block_dec.append(self.conv_layer([filter[i], filter[i]]))
            else:
                self.conv_block_dec.append(nn.Sequential(self.conv_layer([filter[i], filter[i]]),
                                                         self.conv_layer([filter[i], filter[i]])))
                self.conv_block_dec.append(nn.Sequential(self.conv_layer([filter[i], filter[i]]),
                                                         self.conv_layer([filter[i], filter[i]])))
        '''

        
        # define task attention layers
        self.encoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([filter[0], filter[0]])])])
        self.decoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([2 * filter[0], filter[0]])])])
        self.encoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[1]])])
        self.decoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])

        '''
        # PAFE模块
        self.encoder_PAFE_block = nn.ModuleList([PAFEM(filter[0],filter[1])])
        self.decoder_PAFE_block = nn.ModuleList([PAFEM(filter[0],filter[0])])
        '''

        for j in range(2):  # j 表示第几个任务
            if j < 1:
                self.encoder_att.append(nn.ModuleList([self.att_layer([filter[0], filter[0]])]))
                self.decoder_att.append(nn.ModuleList([self.att_layer([2 * filter[0], filter[0]])]))
            for i in range(3):  # i 表示第 j 个任务的，第几个注意力模块
                self.encoder_att[j].append(self.att_layer([2 * filter[i + 1], filter[i + 1]]))
                self.decoder_att[j].append(self.att_layer([filter[i + 1] + filter[i], filter[i]]))

        for i in range(3):
            if i < 1:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 2]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1], filter[i]]))
            elif i == 1:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 2]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1], filter[i]]))
            else:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1], filter[i]]))

        '''
        # PAFE模块
        for i in range(4):
            if i < 3:
                self.encoder_PAFE_block.append(PAFEM([filter[i + 1], filter[i + 2]]))
                self.decoder_PAFE_block.append(PAFEM([filter[i + 1], filter[i]]))
            else:
                self.encoder_PAFE_block.append(PAFEM([filter[i + 1], filter[i + 1]]))
                self.decoder_PAFE_block.append(PAFEM([filter[i + 1], filter[i + 1]]))
        '''

        self.pred_task1 = self.conv_layer([filter[0], self.class_nb], pred=True)
        self.pred_task2 = self.conv_layer([filter[0], 1], pred=True)
        # self.pred_task3 = self.conv_layer([filter[0], 3], pred=True)

        # define pooling and unpooling functions

        # self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True, ceil_mode=True)
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2,
                                          return_indices=True)  # 报错：warnings.warn("Note that order of the arguments: ceil_mode and return_indices will change"
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.logsigma = nn.Parameter(torch.FloatTensor([-0.5, -0.5, -0.5]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def conv_layer(self, channel, pred=False):
        if not pred:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
            )
        else:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[0], kernel_size=3, padding=1),
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            )
        return conv_block

    def att_layer(self, channel):
        att_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[1]),
            nn.Sigmoid(),
        )
        return att_block

    # 其他算法前提 File "/root/autodl-tmp/mtan/utils.py", line 533, in multi_task_trainer_qt
    def shared_modules(self):
        return [self.encoder_block, self.decoder_block,

                self.encoder_block_att, self.decoder_block_att,
                self.down_sampling, self.up_sampling]

    # pcgrad 必备
    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()

    def forward(self, x):
        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = ([0] * 4 for _ in range(5))        # range(5) 中的 5 取决于前面要赋值给 5 个变量
        for i in range(4):
            g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))

        # define attention list for tasks
        atten_encoder, atten_decoder = ([0] * 2 for _ in range(2))
        for i in range(2):
            atten_encoder[i], atten_decoder[i] = ([0] * 4 for _ in range(2))
        for i in range(2):
            for j in range(4):
                atten_encoder[i][j], atten_decoder[i][j] = ([0] * 3 for _ in range(2))

        
        # x = self.kanconv(x)
        
        # define global shared network
        for i in range(4):  # 编码器部分 (下采样)
            if i == 0:
                # print(f'共享编码器部分:i={i}时,输入x的尺寸为{x.shape}')
                g_encoder[i][0] = self.encoder_block[i](x)
                # print(f'共享编码器部分:i={i}时,g_encoder[{i}][0]的尺寸为{g_encoder[i][0].shape}')
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                # print(f'共享编码器部分:i={i}时,g_encoder[{i}][1]的尺寸为{g_encoder[i][1].shape}')

                # 插入SE残差
                # g_encoder[i][1] = self.encoder_SE_block[i](g_encoder[i][1])
                # print(f'共享编码器部分:i={i}时,g_encoder[{i}][1]的尺寸为{g_encoder[i][1].shape}')

                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
                # print(f'共享编码器部分:i={i}时,g_maxpool[{i}]和indices[{i}]的尺寸为{g_maxpool[i].shape},{indices[i].shape}')
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])

                # 插入SE残差
                # g_encoder[i][1] = self.encoder_SE_block[i](g_encoder[i][1])

                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])

        for i in range(4):  # 解码器部分 (上采样)
            if i == 0:
                g_upsampl[i] = self.up_sampling(g_maxpool[-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])

                # 插入SE残差
                # g_decoder[i][1] = self.decoder_SE_block[i](g_decoder[i][1])
            else:
                g_upsampl[i] = self.up_sampling(g_decoder[i - 1][-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])

                # 插入SE残差
                # g_decoder[i][1] = self.decoder_SE_block[i](g_decoder[i][1])

        # define task dependent attention module
        for i in range(2):  # i 表任务数量
            for j in range(4):  # j 表第几个模块
                if j == 0:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](g_encoder[j][0])  # self.encoder_att[i][j] (i、j为0) 对应图中 g h
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    # print(f'编码器中atten_encoder[{i}][{j}][1]的尺寸为{atten_encoder[i][j][1].shape}')
                    
                    # atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                    # print(f'编码器中atten_encoder[{i}][{j}][2]的尺寸为{atten_encoder[i][j][2].shape}')

                    # 用 PAFE模块代替编码器中注意力（上面注释的代码为原注意力代码）
                    # atten_encoder[i][j][2] = self.encoder_PAFE_block[j](atten_encoder[i][j][1])
                    
                    # 用 KConv 代替编码器中注意力 (encoder_block_att)
                    atten_encoder[i][j][2] = self.kconv_encoder_block_att[j](atten_encoder[i][j][1])

                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][j][2], kernel_size=2, stride=2)
                    # print(f'编码器中，经过MaxPool后，atten_encoder[{i}][{j}][2]的尺寸为{atten_encoder[i][j][2].shape}')
                else:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](torch.cat((g_encoder[j][0], atten_encoder[i][j - 1][2]), dim=1))
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    # print(f'编码器中atten_encoder[{i}][{j}][1]的尺寸为{atten_encoder[i][j][1].shape}')
                    
                    # atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                    # print(f'编码器中atten_encoder[{i}][{j}][2]的尺寸为{atten_encoder[i][j][2].shape}')

                    # 用 PAFE模块代替编码器中注意力（上面注释的代码为原注意力代码）
                    # atten_encoder[i][j][2] = self.encoder_PAFE_block[j](atten_encoder[i][j][1])
                    
                    # 用 KConv 代替编码器中注意力 (encoder_block_att)
                    atten_encoder[i][j][2] = self.kconv_encoder_block_att[j](atten_encoder[i][j][1])

                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][j][2], kernel_size=2, stride=2)
                    # print(f'编码器中，经过MaxPool后，atten_encoder[{i}][{j}][2]的尺寸为{atten_encoder[i][j][2].shape}')

            for j in range(4):
                if j == 0:
                    atten_decoder[i][j][0] = F.interpolate(atten_encoder[i][-1][-1], scale_factor=2, mode='bilinear',align_corners=True)
                    # print(f'解码器中atten_decoder[{i}][{j}][0]的尺寸为{atten_decoder[i][j][0].shape}')
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](atten_decoder[i][j][0])
                    # print(f'解码器中,经过decoder_block_att[{-j-1}]后，atten_decoder[{i}][{j}][0]的尺寸为{atten_decoder[i][j][0].shape}')

                    # 用 PAFE模块代替解码器中注意力（上面注释的代码为原注意力代码）
                    # atten_decoder[i][j][0] = self.decoder_PAFE_block[-j - 1](atten_decoder[i][j][0])

                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1))
                    atten_decoder[i][j][2] = (atten_decoder[i][j][1]) * g_decoder[j][-1]
                else:
                    atten_decoder[i][j][0] = F.interpolate(atten_decoder[i][j - 1][2], scale_factor=2, mode='bilinear',align_corners=True)
                    # print(f'解码器中atten_decoder[{i}][{j}][0]的尺寸为{atten_decoder[i][j][0].shape}')
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](atten_decoder[i][j][0])
                    # print(f'解码器中,经过decoder_block_att[{-j-1}]后，atten_decoder[{i}][{j}][0]的尺寸为{atten_decoder[i][j][0].shape}')

                    # 用 PAFE模块代替解码器中注意力（上面注释的代码为原注意力代码）
                    # atten_decoder[i][j][0] = self.decoder_PAFE_block[-j - 1](atten_decoder[i][j][0])

                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1))
                    atten_decoder[i][j][2] = (atten_decoder[i][j][1]) * g_decoder[j][-1]

        # define task prediction layers
        t1_pred = F.log_softmax(self.pred_task1(atten_decoder[0][-1][-1]), dim=1)
        t2_pred = self.pred_task2(atten_decoder[1][-1][-1])
        # t3_pred = self.pred_task3(atten_decoder[2][-1][-1])
        # t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)

        return [t1_pred, t2_pred], self.logsigma

    

# define model, optimiser and scheduler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
KVMSANet_MTAN = KVMSANet().to(device)


'''
# imtl 算法
imt = IMTL('gradient').to(device)
paraments = itertools.chain(KVMSANet_MTAN.parameters(), imt.parameters())
optimizer = torch.optim.Adam(paraments, lr=1e-3)        # 使用 imtl 算法时，需要将 VMSANet_MTAN.parameters() 参数调整为 paraments
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
'''

optimizer = optim.Adam(KVMSANet_MTAN.parameters(), lr=1e-4)  # .parameters() 将网络中的可训练参数以生成器(可以用列表查看)的形式返回
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)  # 调整优化器中的学习率

'''
optimizer = torch.optim.Adam(KVMSANet_MTAN.parameters(), lr=1e-3)        # multi_task_trainer_qt
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)        # multi_task_trainer_qt
'''
print('Parameter Space: ABS: {:.1f}, REL: {:.4f}'.format(count_parameters(KVMSANet_MTAN),
                                                         count_parameters(KVMSANet_MTAN) / 24981069))
print(
    'LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR | NORMAL_LOSS MEAN MED <11.25 <22.5 <30')

# define dataset
dataset_path = opt.dataroot  # '--dataroot', default='nyuv2' 默认为 'nyuv2'
if opt.apply_augmentation:  # '--apply_augmentation', action='store_true'，action='store_true’ 表示如果我们在命令行配置这个参数，则该参数为 True；不配置则默认为 False
    nyuv2_train_set = NYUv2(root=dataset_path, train=True, augmentation=True)
    print('Applying data augmentation on NYUv2.')
else:
    nyuv2_train_set = NYUv2(root=dataset_path, train=True)
    print('Standard training strategy without data augmentation.')  # 没有数据增强

nyuv2_test_set = NYUv2(root=dataset_path, train=False)

batch_size = 8
train_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_train_set,
    batch_size=batch_size,
    shuffle=True)

test_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_test_set,
    batch_size=batch_size,
    shuffle=False)

'''
if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()
'''


# 普通算法以及gradvac_(imtl)、imtl
# Train and evaluate multi-task network
multi_task_trainer(train_loader,
                   test_loader,
                   KVMSANet_MTAN,
                   device,
                   optimizer,
                   scheduler,
                   opt,
                   200)

'''
# 其他算法 pcgrad、cagrad、......
multi_task_trainer_qt(train_loader,
                      test_loader,
                      KVMSANet_MTAN,
                      device,
                      optimizer,
                      scheduler,
                      opt,
                      200)
'''