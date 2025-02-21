import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data.sampler as sampler

import argparse
from VMSANet import VMSANet
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

# define model, optimiser and scheduler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
VMSANet_MTAN = VMSANet().to(device)

'''
# imtl 算法
imt = IMTL('gradient').to(device)
paraments = itertools.chain(VMSANet_MTAN.parameters(), imt.parameters())
optimizer = torch.optim.Adam(paraments, lr=1e-3)        # 使用 imtl 算法时，需要将 VMSANet_MTAN.parameters() 参数调整为 paraments
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
'''

optimizer = optim.Adam(VMSANet_MTAN.parameters(), lr=1e-4)  # .parameters() 将网络中的可训练参数以生成器(可以用列表查看)的形式返回
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)  # 调整优化器中的学习率

'''
optimizer = torch.optim.Adam(VMSANet_MTAN.parameters(), lr=1e-3)        # multi_task_trainer_qt
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)        # multi_task_trainer_qt
'''
print('Parameter Space: ABS: {:.1f}, REL: {:.4f}'.format(count_parameters(VMSANet_MTAN),
                                                         count_parameters(VMSANet_MTAN) / 24981069))
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
                   VMSANet_MTAN,
                   device,
                   optimizer,
                   scheduler,
                   opt,
                   200)

'''
# 其他算法 pcgrad、cagrad、......
multi_task_trainer_qt(train_loader,
                      test_loader,
                      VMSANet_MTAN,
                      device,
                      optimizer,
                      scheduler,
                      opt,
                      200)
'''