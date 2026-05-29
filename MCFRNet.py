import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import math

from MSEF import MSEF
from cnn import SimpleConvBlock ,PMDSConvBlock
from add import ConcatFusion, AddFusion
from CMUNeXtBloc import CMUNeXtBlock
from SAA import SCGA
from CA import CBAM, ChannelAttention

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x
class CMUNeXtBlock(nn.Module):
    def __init__(self, ch_in, ch_out,kernel_size=3, stride=2, depth=1, k=3):
        super(CMUNeXtBlock, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    # deep wise
                    nn.Conv2d(ch_in, ch_in, kernel_size=(k, k), groups=ch_in, padding=(k // 2, k // 2)),
                    nn.ReLU6(),
                    nn.BatchNorm2d(ch_in)
                )),
                nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1)),
                nn.ReLU6(),
                nn.BatchNorm2d(ch_in * 4),
                nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1)),
                nn.ReLU6(),
                nn.BatchNorm2d(ch_in)
            ) for i in range(depth)]
        )
        self.up = conv_block(ch_in, ch_out,kernel_size,stride)
        # self.up = RotaryAttentionPositionEncoder(embed_dim=ch_in)
    def forward(self, x):
        x = self.block(x)
        # print(x.shape)
        # x = rearrange(x, 'b c w y -> b y w c')
        x = self.up(x)
        # x = rearrange(x, 'b y w c -> b c w y')

        return x
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out,kernel_size=3, stride=1):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class channel_att(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(channel_att, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1)
        y = y.transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
class local_att(nn.Module):
    def __init__(self, channel, reduction=16):
        super(local_att, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out
class MSEF(nn.Module): #多尺度有效融合模块
    def __init__(self, c1, c2):
        super().__init__()
        self.channel_att = channel_att(c2)
        self.local_att = local_att(c2)

        self.conv1 = nn.Conv2d(c1, c2, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(c2, c2, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(c2, c2, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(c2)
        self.sigomid = nn.Sigmoid()
        self.group_num = 16
        self.eps = 1e-10
        self.gamma = nn.Parameter(torch.randn(c2, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c2, 1, 1))
        self.gate_genator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(c2, c2, 1, 1),
            nn.ReLU(True),
            nn.Softmax(dim=1),
        )
        self.dwconv = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, groups=c2)
        self.conv3 = nn.Conv2d(c2, c2, kernel_size=1, stride=1)
        self.Apt = nn.AdaptiveAvgPool2d(1)
        self.one = c2
        self.two = c2
        self.conv4_gobal = nn.Conv2d(c2, 1, kernel_size=1, stride=1)
        for group_id in range(0, 4):
            self.interact = nn.Conv2d(c2 // 4, c2 // 4, 1, 1, )

    def forward(self, x1, x2):

        global_conv1 = self.conv1(x1)
        bn_x = self.bn(global_conv1)
        weight_1 = self.sigomid(bn_x)
        global_conv2 = self.conv2(x2)
        bn_x2 = self.bn(global_conv2)
        weight_2 = self.sigomid(bn_x2)
        X_GOBAL = global_conv1 + global_conv2

        temp = self.channel_att(X_GOBAL)

        x_conv4 = self.conv4_gobal(X_GOBAL)
        X_4_sigmoid = self.sigomid(x_conv4)
        X_ = X_4_sigmoid * X_GOBAL
        X_ = X_.chunk(4, dim=1)
        out = []
        for group_id in range(0, 4):
            out_1 = self.interact(X_[group_id])
            N, C, H, W = out_1.size()
            x_1_map = out_1.reshape(N, 1, -1)
            mean_1 = x_1_map.mean(dim=2, keepdim=True)
            x_1_av = x_1_map / mean_1
            x_2_2 = F.softmax(x_1_av, dim=1)
            x1 = x_2_2.reshape(N, C, H, W)
            x1 = X_[group_id] * x1
            out.append(x1)
        out = torch.cat([out[0], out[1], out[2], out[3]], dim=1)
        N, C, H, W = out.size()
        x_add_1 = out.reshape(N, self.group_num, -1)
        N, C, H, W = X_GOBAL.size()
        x_shape_1 = X_GOBAL.reshape(N, self.group_num, -1)
        mean_1 = x_shape_1.mean(dim=2, keepdim=True)
        std_1 = x_shape_1.std(dim=2, keepdim=True)
        x_guiyi = (x_add_1 - mean_1) / (std_1 + self.eps)
        x_guiyi_1 = x_guiyi.reshape(N, C, H, W)
        x_gui = (x_guiyi_1 * self.gamma + self.beta)

        weight_x3 = self.Apt(X_GOBAL)
        reweights = self.sigomid(weight_x3)
        x_up_1 = reweights >= weight_1
        x_low_1 = reweights < weight_1
        x_up_2 = reweights >= weight_2
        x_low_2 = reweights < weight_2
        x_up = x_up_1 * X_GOBAL + x_up_2 * X_GOBAL
        x_low = x_low_1 * X_GOBAL + x_low_2 * X_GOBAL
        x11_up_dwc = self.dwconv(x_low)
        x11_up_dwc = self.conv3(x11_up_dwc)
        x_so = self.gate_genator(x_low)
        x11_up_dwc = x11_up_dwc * x_so
        x22_low_pw = self.conv4(x_up)
        xL = x11_up_dwc + x22_low_pw

        xL = xL + x_gui + temp
        out = self.local_att(xL)
        return out

class PAM_Module(nn.Module):
    """空间注意力模块"""
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# 二次创新注意力模块 SCGA 自我感知协调注意力 冲SCI一区
'''
SCGA 自我感知协调注意力 内容介绍：

1.执行通道注意力机制。它对每个通道进行全局平均池化，
然后通过1D卷积来捕捉通道之间的交互信息。确保模型能够有效地聚焦在最相关的通道特征上。
2.全局空间注意力 Global Spatial Attention (GSA):
提取和整合全局空间信息，从而增强并优化特征表示。
GSA 通过对特征图进行卷积和重构，生成位置相关的注意力图，进而与输入特征结合，形成强化后的特征。
3.多头自注意力 Transformer Self Attention (TSA):
使用 Transformer 的多头自注意力机制，能够捕获全局上下文信息并建模长程依赖。
TSA 首先通过线性变换生成查询 (Q)、键 (K) 和值 (V) 的特征表示，然后通过点积操作计算注意力权重，聚合全局特征信息。
'''
class SCGA(nn.Module):
    def __init__(self, in_channels):
        super(SCGA, self).__init__()
        self.gsa = PAM_Module(in_dim=in_channels)
        # self.tsa = ScaledDotProductAttention()
        self.ca = ChannelAttention(in_channels)
    def forward(self, x):
        x1 = x * self.ca(x)
        x1 = x1 * self.gsa(x)

        x2 = self.gsa(x)

        out = x1 + x2
        return out

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
        # self.cmf = SimpleConvBlock(64,64)
        # self.cmf = PMDSConvBlock(64,64)


        self.msef =MSEF(64,64)
        # self.msef = ConcatFusion(64)
        # self.msef = AddFusion(64)


        self.saa = SCGA(64)
        # self.saa = CBAM(64)
        # self.saa = ChannelAttention(64)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        #[64, 1, 30, 13, 13]
        x = self.stem_3D(x)#[64, 8, 28, 11, 11]

        x = rearrange(x, 'b c h w y -> b (c h) w y')

        x = self.jiangwei(x)

        x = self.cmf(x)#[64, 64, 13, 13]
        # x = self.cmf(x)

        x = self.msef(x,x)
        # x = self.msef(x)

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
