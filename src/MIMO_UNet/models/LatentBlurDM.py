# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from thop import profile
import time
from .LatentEncoder import LE_arch

class ResMLP(nn.Module):
    def __init__(self,n_feats = 512):
        super(ResMLP, self).__init__()
        self.resmlp = nn.Sequential(
            nn.Linear(n_feats , n_feats ),
            nn.LeakyReLU(0.1, True),
        )
    def forward(self, x):
        res=self.resmlp(x)
        return res
    
class denoise(nn.Module):
    def __init__(self,n_feats = 64, n_denoise_res = 5,timesteps=5):
        super(denoise, self).__init__()
        # self.max_period=timesteps*10
        self.max_period=timesteps
        n_featsx4=4*n_feats
        resmlp = [
            nn.Linear(n_featsx4*2+1, n_featsx4),
            nn.LeakyReLU(0.1, True),
        ]
        for _ in range(n_denoise_res):
            resmlp.append(ResMLP(n_featsx4))
        self.resmlp=nn.Sequential(*resmlp)

    def forward(self,x, t,c):
        t=t.float()
        t =t/self.max_period
        t=t.view(-1,1)
        c = torch.cat([c,t,x],dim=1)
        fea = self.resmlp(c)

        return fea

class LatentExposureDiffusion(nn.Module):
    def __init__(self, total_timestamps=5, spvised_mid_out=False):
        super().__init__()

        self.lcr_model = denoise(timesteps = total_timestamps)
        self.noise_model = denoise(timesteps = total_timestamps)
        self.condition_encoder = LE_arch()

        self.total_timestamps = total_timestamps
        self.spvised_mid_out = spvised_mid_out

        # self.r = 0.2
        beta_start = 0.0
        beta_end = 0.02 # default 0.02
        alpha_start = 1.0
        alpha_end = 2.0
        betas = torch.linspace(beta_start, beta_end, self.total_timestamps, dtype=torch.float32)
        alphas = torch.linspace(alpha_start, alpha_end, self.total_timestamps+1, dtype=torch.float32)
        betas_bar_list = self.get_beta_bar(alphas, betas)
        time_stamps_list = torch.tensor([torch.tensor(i) for i in range(self.total_timestamps, 0, -1)])
        
        self.register_buffer("alphas", alphas)
        self.register_buffer("betas_bar", betas_bar_list)
        self.register_buffer("time_stamps_list", time_stamps_list)

    def get_beta_bar(self, alphas, betas):
        betas_bar_list = []
        for t in range(1, self.total_timestamps+1):
            sub_betas = betas[:t]  # 取前 t 个 beta 值
            weights = torch.tensor([(alphas[i-1] / alphas[t])**2 for i in range(1, t+1)], dtype=torch.float32)
            result = torch.sum(weights * sub_betas)
            betas_bar_list.append(result.clone().detach().sqrt())
        return torch.tensor(betas_bar_list)
    
    def q_sample_d(self, img):
        noise = torch.randn_like(img)
        return img + self.betas_bar[self.total_timestamps-1] * noise, noise
    
    # def model_loading(self):
    #     print("Load Pretrained LCRDDM !")

    #     ce_checkpoint_path = 'experiments/GoPro_fftformer_stage1_1e-5/models/net_le_100000.pth'
    #     ce_checkpoint = torch.load(ce_checkpoint_path)
    #     ce_checkpoint = ce_checkpoint['params']
    #     self.condition_encoder.load_state_dict(ce_checkpoint)

    #     lcr_path = "experiments/FFTformer_stage2v2/lcr_final.pth"
    #     noise_path = "experiments/FFTformer_stage2v2/noise_final.pth"

    #     lcr_checkpoint_path = lcr_path
    #     lcr_checkpoint = torch.load(lcr_checkpoint_path)
    #     self.lcr_model.load_state_dict(lcr_checkpoint)

    #     noise_checkpoint_path = noise_path
    #     noise_checkpoint = torch.load(noise_checkpoint_path)
    #     self.noise_model.load_state_dict(noise_checkpoint)

    def forward(self, blur):
        pred_lcr_list = []
        pred_noise_list = []
        device = self.alphas.device
        b = blur.shape[0]
        T_z = self.condition_encoder(blur, blur)
        noise_img, noise = self.q_sample_d(T_z)
        for i in self.time_stamps_list:
            # t = i.unsqueeze(0)
            t = torch.full((b,), i,  device=device, dtype=torch.long)
            # print(t.shape, T_z.shape, noise_img.shape)
            pred_noise = self.noise_model(noise_img, t, T_z)
            pred_lcr = self.lcr_model(noise_img, t, T_z)

            if self.spvised_mid_out:
                pred_lcr_list.append(pred_lcr)
                pred_noise_list.append(pred_noise)

            if i == 1:
                noise_cof = self.betas_bar[i - 1]
            else: 
                beta_t_bar = self.betas_bar[i - 1]
                beta_t_minus1_bar = self.betas_bar[i - 2]
                noise_cof = (self.alphas[i]*beta_t_bar)/self.alphas[i-1] - beta_t_minus1_bar

            noise_img = ((self.alphas[i]*(noise_img) - pred_lcr)/self.alphas[i-1]) - noise_cof * pred_noise
        if self.spvised_mid_out:
            return noise_img, pred_lcr_list, pred_noise_list, noise
        else:
            return noise_img

# if __name__ == '__main__':
#     net = LatentExposureDiffusion(
#     ).cuda()
    
#     input = torch.randn((1, 3, 256, 256)).cuda()
#     flops, params = profile(net, (input,))
#     print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
#     print('Params = ' + str(params / 1000 ** 2) + 'M')

#     out = net(input)
#     print(out.shape)

if __name__ == '__main__':
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True  # 為不同大小的輸入加速

    # 建立模型
    net = LatentExposureDiffusion().cuda()
    net.eval()

    # 建立 4K 測試影像輸入與 prior 向量
    input_tensor = torch.randn(1, 3, 2160, 3840).cuda()  # 4K 圖像大小

    # 預熱 (warm-up)
    with torch.no_grad():
        for _ in range(5):
            _ = net(input_tensor)

    # 監測記憶體與推理時間
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    with torch.no_grad():
        output = net(input_tensor)
    end_time = time.time()
    inference_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

    # 計算 FLOPs 和參數量（注意：這裡是以 256x256 為估算基準）
    dummy_input = torch.randn(1, 3, 256, 256).cuda()
    flops, params = profile(net, (dummy_input, ))

    # 輸出結果
    print(f"[4K Input] Inference Time: {inference_time:.3f} sec")
    print(f"[4K Input] Peak Memory: {peak_memory:.2f} MB")
    print(f"[256x256 Input] FLOPs = {flops / 1e9:.2f} GFLOPs")
    print(f"[256x256 Input] Params = {params / 1e6:.2f} M")