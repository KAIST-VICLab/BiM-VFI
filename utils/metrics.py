import numpy as np
import math
import torch
import torch.nn.functional as F
from math import exp
from stlpips_pytorch import stlpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from basicsr.metrics.niqe import calculate_niqe
import lpips


def calculate_batch_psnr(gt_tensor, output_tensor, mode='avg'):
    # both parameters are in the form of tensor of size: BS, C, H, W

    if mode == 'avg':
        gt_np = gt_tensor.cpu().numpy().astype(np.float32)
        output_np = output_tensor.cpu().numpy().astype(np.float32)

        bs = gt_np.shape[0]
        psnr_list = []
        psnr = 0
        for i in range(bs):
            gt_im = gt_np[i, :, :, :]
            output_im = output_np[i, :, :, :]

            gt_im = gt_im.transpose((1, 2, 0))
            output_im = output_im.transpose((1, 2, 0))
            output_im = np.round(output_im * 255) / 255.
            # output_im = np.round(output_im * 255).astype('uint8') / 255
            psnr_it = -10 * np.log10(((gt_im - output_im) ** 2).mean())
            psnr_list.append(psnr_it)
            psnr += psnr_it

            # psnr_list.append(-10 * math.log10(((gt_im - output_im) * (gt_im - output_im)).mean()))
            # psnr += -10 * math.log10(((gt_im - output_im) * (gt_im - output_im)).mean())
            # psnr_list.append(peak_signal_noise_ratio(gt_im, output_im, data_range=1.))
            # psnr += peak_signal_noise_ratio(gt_im, output_im, data_range=1.)
        return float(psnr / bs), psnr_list
    else:
        raise NotImplementedError


def calculate_batch_ssim(gt_tensor, output_tensor, mode='avg'):
    if mode == 'avg':
        gt_np = gt_tensor.cpu().numpy().astype(np.float32)
        output_np = output_tensor.cpu().numpy().astype(np.float32)
        output_tensor = torch.round(output_tensor * 255) / 255.

        bs = gt_np.shape[0]
        ssim = 0
        # for i in range(bs):
        #     gt_im = gt_np[i, :, :, :]
        #     output_im = output_np[i, :, :, :]
        #     gt_im = gt_im.transpose((1, 2, 0))
        #     output_im = output_im.transpose((1, 2, 0))
        #     output_np = np.round(output_im * 255) / 255.
        #
        #     ssim += ssim_matlab(gt_im, output_im)
        for i in range(bs):
            ssim += ssim_matlab(gt_tensor[i:i+1], output_tensor[i:i+1])

        return float(ssim / bs), bs
    else:
        raise NotImplementedError


def calculate_batch_lpips(gt_tensor, output_tensor, lpips_metric, mode='avg'):
    if mode == 'avg':
        gt_tensor = gt_tensor * 2 - 1
        output_tensor = output_tensor * 2 - 1

        bs = gt_tensor.shape[0]
        lpips_sum = 0
        for i in range(bs):
            lpips_sum += lpips_metric(gt_tensor[i:i+1], output_tensor[i:i+1])

        return float(lpips_sum / bs), bs
    else:
        raise NotImplementedError


def calculate_batch_stlpips(gt_tensor, output_tensor, stlpips_metric, mode='avg'):
    if mode == 'avg':
        gt_tensor = gt_tensor * 2 - 1
        output_tensor = output_tensor * 2 - 1

        bs = gt_tensor.shape[0]
        stlpips_sum = 0
        for i in range(bs):
            stlpips_sum += stlpips_metric(gt_tensor[i:i+1], output_tensor[i:i+1])

        return float(stlpips_sum / bs), bs
    else:
        raise NotImplementedError


def calculate_batch_niqe(gt_tensor, output_tensor, niqe_metric, mode='avg'):
    if mode == 'avg':
        bs = gt_tensor.shape[0]
        niqe_sum = 0
        for i in range(bs):
            gt_np = (gt_tensor[i].cpu().numpy().transpose(1, 2, 0) * 255.).astype(np.uint8)
            output_np = (output_tensor[i].cpu().numpy().transpose(1, 2, 0) * 255.).astype(np.uint8)
            niqe_sum += calculate_niqe(output_np, crop_border=0)

        return float(niqe_sum), 1
    else:
        raise NotImplementedError


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0).cuda()
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def create_window_3d(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _2D_window.unsqueeze(2) @ (_1D_window.t())
    window = _3D_window.expand(1, channel, window_size, window_size, window_size).contiguous().cuda()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    # mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    # mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
    mu1 = F.conv2d(F.pad(img1, (5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=channel)
    mu2 = F.conv2d(F.pad(img2, (5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    # sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    # sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    sigma1_sq = F.conv2d(F.pad(img1 * img1, (5, 5, 5, 5), 'replicate'), window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(F.pad(img2 * img2, (5, 5, 5, 5), 'replicate'), window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(F.pad(img1 * img2, (5, 5, 5, 5), 'replicate'), window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def ssim_matlab(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=1):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, _, height, width) = img1.shape
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window_3d(real_size, channel=1).to(img1.device)
        # Channel is set to 1 since we consider color images as volumetric images

    img1 = img1.unsqueeze(1)
    img2 = img2.unsqueeze(1)

    mu1 = F.conv3d(F.pad(img1, (5, 5, 5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=1)
    mu2 = F.conv3d(F.pad(img2, (5, 5, 5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(F.pad(img1 * img1, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu1_sq
    sigma2_sq = F.conv3d(F.pad(img2 * img2, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu2_sq
    sigma12 = F.conv3d(F.pad(img1 * img2, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 3 channel for SSIM
        self.channel = 3
        self.window = create_window(window_size, channel=self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        _ssim = ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)
        dssim = (1 - _ssim) / 2
        return dssim


class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)
