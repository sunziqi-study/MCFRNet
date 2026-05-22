import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

from CA import CBAM
from MSEF import MSEF
from add import ConcatFusion
from CMUNeXtBloc import CMUNeXtBlock
from SAA import SCGA

class ChannelReducer(nn.Module):
    def __init__(self, input_dim, channel_num):
        super().__init__()
        """
            注意：根据降维的程度进行修改
        """
        self.feature_fuser = nn.Linear(input_dim, channel_num)  # 定义融合层，用于融合来自不同方向的信息

        self.conv_bn_activation = nn.Sequential(
            # 添加2D卷积层，输入通道数是原通道数的3倍，输出通道数为channel_num
            nn.Conv2d(input_dim, channel_num, kernel_size=1, stride=1),
            # 批量归一化层，对channel_num个特征图进行归一化
            nn.BatchNorm2d(channel_num),
            # ReLU激活函数，inplace=True表示直接在输入数据上进行修改以节省内存
            nn.ReLU(inplace=True),
        )

    def forward(self, x1):
        # 在通道维度上合并不同的特征
        merged_features = x1

        # 维度还原方式① 通过线性变换层进行维度变换，nn.Linear 默认对最后一个维度进行操
        # 调换位置 B C H W === B H W C
        # transposed_features = merged_features.permute(0, 2, 3, 1)
        # output_tensor = self.feature_fuser(transposed_features)
        # output_tensor = output_tensor.permute(0, 3, 1, 2)

        # 维度还原方式②
        output_tensor = self.conv_bn_activation(merged_features)
        return output_tensor





class MCFRNet(nn.Module):
    def __init__(self, channels=1, num_classes=16, drop=0.1):
        super(MCFRNet, self).__init__()
        self.stem_3D = nn.Sequential(
            nn.Conv3d(channels, out_channels=8, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        self.jiangwei = ChannelReducer(input_dim=224, channel_num=64)
        self.cmf = CMUNeXtBlock(64,64)



        self.msef =MSEF(64,64)

        self.saa = SCGA(64)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        #[64, 1, 30, 13, 13]
        x = self.stem_3D(x)#[64, 8, 28, 11, 11]

        x = rearrange(x, 'b c h w y -> b (c h) w y')

        x = self.jiangwei(x)

        x = self.cmf(x)#[64, 64, 13, 13]


        x = self.msef(x,x)


        x = self.saa(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = MCFRNet(channels=1, num_classes=9)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    input = torch.randn(64, 1, 30, 13, 13).to(device)
    y = model(input)
    print(y.size())
