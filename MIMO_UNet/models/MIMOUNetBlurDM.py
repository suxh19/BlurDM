import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import BasicConv, ResBlock
from thop import profile
import time

class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)
        self.kv = nn.Linear(256, 2 * channel, bias=False)

    def forward(self, x, prior):
        B, C, H, W = x.shape

        kv = self.kv(prior)
        kv = kv.view(B, C * 2, 1, 1)
        k, v = kv.chunk(2, dim=1)  

        x = x * k + v

        return self.layers(x)


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-3, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out


class MIMOUNetPlusPrior(nn.Module):
    def __init__(self, num_res = 20):
        super(MIMOUNetPlusPrior, self).__init__()
        base_channel = 32
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

        self.drop1 = nn.Dropout2d(0.1)
        self.drop2 = nn.Dropout2d(0.1)

    def forward(self, x, prior):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        res2 = self.drop2(res2)
        res1 = self.drop1(res1)

        z = self.Decoder[0](z, prior)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z, prior)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z, prior)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs


def build_MIMOUnet_net(model_name):
    class ModelError(Exception):
        def __init__(self, msg):
            self.msg = msg

        def __str__(self):
            return self.msg

    if model_name == "MIMOUNetBlurDM":
        return MIMOUNetPlusPrior()
    raise ModelError('Wrong Model!\nYou should choose MIMO-UNetPlus or MIMO-UNet.')

# if __name__ == '__main__':
#     # Debug
#     #logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#     net = build_MIMOUnet_net("MIMO-UNetPlusPrior")
#     net = net.cuda()
#     input = torch.randn(1, 3, 256, 256).cuda()
#     prior = torch.randn((1, 256)).cuda()
#     flops, params = profile(net, (input, prior))
#     print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
#     print('Params = ' + str(params / 1000 ** 2) + 'M')

if __name__ == '__main__':
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True  # 為不同大小的輸入加速

    # 建立模型
    net = build_MIMOUnet_net("MIMO-UNetPlusPrior").cuda()
    net.eval()

    # 建立 4K 測試影像輸入與 prior 向量
    input_tensor = torch.randn(1, 3, 2160, 3840).cuda()  # 4K 圖像大小
    prior = torch.randn(1, 256).cuda()  # 根據你的模型需求

    # 預熱 (warm-up)
    with torch.no_grad():
        for _ in range(5):
            _ = net(input_tensor, prior)

    # 監測記憶體與推理時間
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    with torch.no_grad():
        output = net(input_tensor, prior)
    end_time = time.time()
    inference_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

    # 計算 FLOPs 和參數量（注意：這裡是以 256x256 為估算基準）
    dummy_input = torch.randn(1, 3, 256, 256).cuda()
    dummy_prior = torch.randn(1, 256).cuda()
    flops, params = profile(net, (dummy_input, dummy_prior))

    # 輸出結果
    print(f"[4K Input] Inference Time: {inference_time:.3f} sec")
    print(f"[4K Input] Peak Memory: {peak_memory:.2f} MB")
    print(f"[256x256 Input] FLOPs = {flops / 1e9:.2f} GFLOPs")
    print(f"[256x256 Input] Params = {params / 1e6:.2f} M")