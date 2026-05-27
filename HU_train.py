import os
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


from operator import truediv
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from utils.utils import *
from utils.get_cls_map import *
from MCFRNet import MCFRNet
from torch import nn

from thop import profile
from fvcore.nn import FlopCountAnalysis
import numpy as np
import torch
import torch.optim as optim
import time
import Flops


def loadData():
    # 读入数据
    data = sio.loadmat('data/Houston.mat')['Houston']
    labels = sio.loadmat('data/Houston_gt.mat')['Houston_gt']

    return data, labels


# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX


# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX


# 在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式
def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1

    return patchesData, patchesLabels


def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=testRatio,
                                                        random_state=randomState,
                                                        stratify=y)

    return X_train, X_test, y_train, y_test


BATCH_SIZE_TRAIN = 128


def create_data_loader():
    # 地物类别
    # class_num = 16
    # 读入数据
    X, y = loadData()
    # 用于测试样本的比
    test_ratio = 0.99
    # 每个像素周围提取 patch 的尺寸
    patch_size = 11 ### 11
    # 使用 PCA 降维，得到主成分的数量
    pca_components = 30

    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)

    print('\n... ... PCA tranformation ... ...')
    X_pca = applyPCA(X, numComponents=pca_components)
    print('Data shape after PCA: ', X_pca.shape)

    print('\n... ... create data cubes ... ...')
    X_pca, y_all = createImageCubes(X_pca, y, windowSize=patch_size)
    print('Data cube X shape: ', X_pca.shape)
    print('Data cube y shape: ', y.shape)

    print('\n... ... create train & test data ... ...')
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y_all, test_ratio)
    print('Xtrain shape: ', Xtrain.shape)
    print('Xtest  shape: ', Xtest.shape)

    # 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
    X = X_pca.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)
    print('before transpose: Xtrain shape: ', Xtrain.shape)
    print('before transpose: Xtest  shape: ', Xtest.shape)

    # 为了适应 pytorch 结构，数据要做 transpose
    X = X.transpose(0, 4, 3, 1, 2)
    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    Xtest = Xtest.transpose(0, 4, 3, 1, 2)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)

    # 创建train_loader和 test_loader
    X = TestDS(X, y_all)
    trainset = TrainDS(Xtrain, ytrain)
    testset = TestDS(Xtest, ytest)
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=True,
                                               num_workers=0,
                                               )
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                              batch_size=BATCH_SIZE_TRAIN,
                                              shuffle=False,
                                              num_workers=0,
                                              )
    all_data_loader = torch.utils.data.DataLoader(dataset=X,
                                                  batch_size=BATCH_SIZE_TRAIN,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  )

    return train_loader, test_loader, all_data_loader, y




""" Training dataset"""


class TestDS(torch.utils.data.Dataset):

    def __init__(self, Xtest, ytest):
        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_number(num):
    if num >= 1e9:
        return f'{num / 1e9:.4f} G'
    elif num >= 1e6:
        return f'{num / 1e6:.4f} M'
    elif num >= 1e3:
        return f'{num / 1e3:.4f} K'
    else:
        return str(num)


def calc_flops_params(model, device, train_loader):
    model.eval()

    # 直接从 train_loader 里取一个真实样本，避免手写维度出错
    sample_input, _ = next(iter(train_loader))
    dummy_input = sample_input[:1].to(device)

    with torch.no_grad():
        macs, _ = profile(model, inputs=(dummy_input,), verbose=False)

    # thop 返回 MACs，一般 FLOPs 约等于 2 * MACs
    flops = 2 * macs

    total_params, trainable_params = count_params(model)

    return total_params, trainable_params, macs, flops



def train(train_loader, epochs):
    # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 网络放到GPU上
    net = MCFRNet(num_classes=15).to(device)
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 初始化优化器
    optimizer = optim.Adam(net.parameters(), lr=0.0002)
    # 开始训练
    total_loss = 0
    for epoch in range(epochs):
        net.train()
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # 正向传播 +　反向传播 + 优化
            # 通过输入得到预测的输出
            outputs = net(data)
            # 计算损失函数
            loss = criterion(outputs, target)
            # 优化器梯度归零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' % (epoch + 1,
                                                                         total_loss / (epoch + 1),
                                                                         loss.item()))

    print('Finished Training')

    return net, device


def test(device, net, test_loader):
    net.eval()
    y_pred_test = []
    y_test = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = net(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            y_pred_test.extend(preds)
            y_test.extend(labels.numpy())

    return np.array(y_pred_test), np.array(y_test)


def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def acc_reports(y_test, y_pred_test):
    target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
        , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                    'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                    'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives']
    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa * 100, confusion, each_acc * 100, aa * 100, kappa * 100


if __name__ == '__main__':
    os.makedirs('HU', exist_ok=True)

    for count in range(3):
        print('第' + str(count) + '次')

        train_loader, test_loader, all_data_loader, y_all = create_data_loader()

        tic1 = time.perf_counter()
        net, device = train(train_loader, epochs=200)

        total_params, trainable_params, macs, flops = calc_flops_params(net, device, train_loader)

        print(f'Total Params: {format_number(total_params)}')
        print(f'Trainable Params: {format_number(trainable_params)}')
        print(f'MACs: {format_number(macs)}')
        print(f'FLOPs: {format_number(flops)}')

        toc1 = time.perf_counter()

        tic2 = time.perf_counter()
        y_pred_test, y_test = test(device, net, test_loader)
        toc2 = time.perf_counter()

        classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)

        Training_Time = toc1 - tic1
        Test_time = toc2 - tic2

        file_name = f'HU/hu{count}.txt'

        with open(file_name, 'w') as x_file:
            x_file.write('{} Training_Time (s)\n'.format(Training_Time))
            x_file.write('{} Test_time (s)\n'.format(Test_time))

            x_file.write('{} Total Params\n'.format(total_params))
            x_file.write('{} Trainable Params\n'.format(trainable_params))
            x_file.write('{} MACs\n'.format(macs))
            x_file.write('{} FLOPs\n'.format(flops))

            x_file.write('{} Total Params Formatted\n'.format(format_number(total_params)))
            x_file.write('{} Trainable Params Formatted\n'.format(format_number(trainable_params)))
            x_file.write('{} MACs Formatted\n'.format(format_number(macs)))
            x_file.write('{} FLOPs Formatted\n'.format(format_number(flops)))

            x_file.write('{} Overall accuracy (%)\n'.format(oa))
            x_file.write('{} Average accuracy (%)\n'.format(aa))
            x_file.write('{} Kappa accuracy (%)\n'.format(kappa))
            x_file.write('{} Each accuracy (%)\n'.format(each_acc))
            x_file.write('{}\n'.format(str(classification)))
            x_file.write('{}\n'.format(confusion))

    print('Finished Training')