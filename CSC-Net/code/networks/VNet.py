import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class DepthwiseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseConv3d, self).__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class CFF(nn.Module):
    def __init__(self, in_channel1, in_channel2, out_channel):
        super(CFF, self).__init__()

        act_fn = nn.ReLU(inplace=True)

        self.layer0 = BasicConv3d(in_channel1, out_channel // 2, 1)
        self.layer1 = BasicConv3d(in_channel2, out_channel // 2, 1)

        # Depthwise separable convolutions for efficiency
        self.layer3_1 = nn.Sequential(
            DepthwiseConv3d(out_channel, out_channel // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channel // 2), act_fn
        )
        self.layer3_2 = nn.Sequential(
            DepthwiseConv3d(out_channel, out_channel // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channel // 2), act_fn
        )
        self.layer5_1 = nn.Sequential(
            DepthwiseConv3d(out_channel, out_channel // 2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm3d(out_channel // 2), act_fn
        )
        self.layer5_2 = nn.Sequential(
            DepthwiseConv3d(out_channel, out_channel // 2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm3d(out_channel // 2), act_fn
        )
        # A final convolution layer to aggregate all features
        self.layer_out = nn.Sequential(
            nn.Conv3d(out_channel // 2, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channel), act_fn
        )

    def forward(self, x0, x1):

        x0_1 = self.layer0(x0)
        x1_1 = self.layer1(x1)

        x_3_1 = self.layer3_1(torch.cat((x0_1, x1_1), dim=1))
        x_5_1 = self.layer5_1(torch.cat((x1_1, x0_1), dim=1))

        x_3_2 = self.layer3_2(torch.cat((x_3_1, x_5_1), dim=1))
        x_5_2 = self.layer5_2(torch.cat((x_5_1, x_3_1), dim=1))

        out = self.layer_out(x0_1 + x1_1 + torch.mul(x_3_2, x_5_2))

        return out


def _upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='trilinear', align_corners=True)
    return src

class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages - 1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class DownsamplingConvBlock2(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock2, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, stride=stride, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, stride=stride, padding=1))

        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class Upsampling_function(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', mode_upsampling=1):
        super(Upsampling_function, self).__init__()

        ops = []
        if mode_upsampling == 0:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        if mode_upsampling == 1:
            ops.append(nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=True))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        elif mode_upsampling == 2:
            ops.append(nn.Upsample(scale_factor=stride, mode="nearest"))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))

        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class Encoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False, noneed=False):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        self.noneed= noneed
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.high_fusion1 = CFF(16, 32, 16)
        self.high_fusion2 = CFF(32, 64, 32)
        self.high_fusion3 = CFF(64, 128, 64)
        self.high_fusion4 = CFF(128, 256, 128)

    def forward(self, input, noneed=None):
        if noneed is None:
            noneed = self.noneed

        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        if self.has_dropout:
            x5 = self.dropout(x5)

        if noneed:
            res_down = [x1, x2, x3, x4]
            return res_down

        x1 = self.high_fusion1(x1,self.up(x2))
        x2 = self.high_fusion2(x2,self.up(x3))
        x3 = self.high_fusion3(x3,self.up(x4))
        x4 = self.high_fusion4(x4,self.up(x5))

        res = [x1, x2, x3, x4, x5]
        return res


class Decoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False, up_type=0):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization,
                                                 mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization,
                                                mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)

        return out_seg

import torch.nn.functional as F
class Decoder2(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False, up_type=0):
        super(Decoder2, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization,
                                                mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]

        x6_up = self.block_six_up(x4)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)
        out_seg_up = F.interpolate(out_seg, scale_factor=2, mode="trilinear", align_corners=True)

        return out_seg,out_seg_up

class SELayer3D(nn.Module):
    def __init__(self, in_channel=512, output_channel=2, reduction=16):
        super(SELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # 3D 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, output_channel, bias=False),
            nn.Softmax(dim=1)  # 指定 dim=1 以对通道维度进行 softmax
        )

        # 初始化线性层
        for module in self.fc:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        b, c, _, _, _ = x.size()  # (B, C, D, H, W)
        y = self.avg_pool(x).view(b, c)  # 变成 (B, C)
        y = self.fc(y).view(b, -1, 1, 1, 1)  # 变回 (B, 2, 1, 1, 1)
        return y

class Decoder3(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False, up_type=0):
        super(Decoder3, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization,
                                                 mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization,
                                                mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def extract_fuse_feat(self, hr_features, lr_features):

        x_hr = hr_features
        x_lr = lr_features

        d, h, w = x_hr.shape[-3:]
        x_lr = x_lr[:, :, :d, :h, :w]  # 截取低分辨率特征图，使其匹配高分辨率

        # 确保 x_lr 的通道数是 x_hr 的一半，通过卷积层扩展 x_lr 的通道数
        if x_hr.shape[1] == 2 * x_lr.shape[1]:
            conv = nn.Conv3d(x_lr.shape[1], x_hr.shape[1], kernel_size=1)
            conv = conv.to(x_hr.device)  # 确保卷积层与输入张量位于同一设备上
            x_lr = conv(x_lr)

        # 将 x_hr 和 x_lr 合并后传递给 SELayer3D
        se_layer = SELayer3D(in_channel=x_hr.shape[1] + x_lr.shape[1], output_channel=2).to(x_hr.device)
        score = se_layer(torch.cat([x_hr, x_lr], dim=1))  # 计算通道注意力

        # 使 score 可以广播计算
        score_0 = score[:, 0].unsqueeze(1)  # (B, 1, 1, 1, 1)
        score_1 = score[:, 1].unsqueeze(1)  # (B, 1, 1, 1, 1)

        # 加权融合
        x_fr = score_0 * x_hr + score_1 * x_lr

        return x_fr

    def forward(self, features1,features2):
        x1 = features1[0]
        x2 = features1[1]
        x3 = features1[2]
        x4 = features1[3]
        x5 = features1[4]

        y2 = features2[1]
        y3 = features2[2]
        y4 = features2[3]

        x3 = self.extract_fuse_feat(x3, y2)
        x4 = self.extract_fuse_feat(x4, y3)
        x5 = self.extract_fuse_feat(x5, y4)

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)

        return out_seg

class VNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        super(VNet, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)

    def forward(self, input):
        features = self.encoder(input)
        out_seg1 = self.decoder1(features)
        return out_seg1

class CCNet3d_V1(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False, noneed=False):
        super(CCNet3d_V1, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, noneed)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = Decoder2(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)
        self.decoder3 = Decoder3(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 2)

    def forward(self, input1,input2):
        features1 = self.encoder(input1)
        features2 = self.encoder(input2, noneed=True)
        out_seg1 = self.decoder1(features1)
        out_seg2, out_seg2_up = self.decoder2(features2)
        out_seg3 = self.decoder3(features1, features2)
        return out_seg1, out_seg2, out_seg2_up, out_seg3



if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    #     from ptflops import get_model_complexity_info

    #     model = CCNet3d(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=False)
    #     with torch.cuda.device(0):
    #         macs, params = get_model_complexity_info(model, (1, 112, 112, 80), as_strings=True,
    #                                                  print_per_layer_stat=True, verbose=True)
    #         print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #         print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    #     with torch.cuda.device(0):
    #         macs, params = get_model_complexity_info(model, (1, 96, 96, 96), as_strings=True,
    #                                                  print_per_layer_stat=True, verbose=True)
    #         print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #         print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    #     import ipdb;

    #     ipdb.set_trace()

    with torch.no_grad():
        import os

        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        x = torch.rand((4, 1, 112, 112, 80), device=cuda0)
        y = torch.rand((4, 1, 56, 56, 40), device=cuda0)
        model = CCNet3d_V1(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=False)
        model.cuda()
        output = model(x,y)
        print('output1:', output[0].shape)
        print('output2:', output[1].shape)
        print('output3:', output[2].shape)
        print('output4:', output[3].shape)
