import torch
import torch.nn as nn
import torch.nn.functional as F
from .backwarp import backwarp


class CAUN(nn.Module):
    def __init__(self, feat_channels):
        super(CAUN, self).__init__()
        self.enc0 = nn.Sequential(
            nn.Conv2d(feat_channels * 8, feat_channels * 4, 3, padding=1),
            nn.PReLU(feat_channels * 4),
        )
        self.enc1 = nn.Sequential(
            nn.Conv2d(feat_channels * 5, feat_channels * 4, 3, padding=1),
            nn.PReLU(feat_channels * 4),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(feat_channels * 3, feat_channels * 1, 3, padding=1),
            nn.PReLU(feat_channels * 1),
        )
        self.kernel_x2 = nn.Sequential(
            nn.Conv2d(feat_channels * 4, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2),
            nn.Conv2d(feat_channels * 2, 2 * 1 * 9, 3, padding=1)
        )
        self.kernel_x4 = nn.Sequential(
            nn.Conv2d(feat_channels * 1, feat_channels * 1, 3, padding=1),
            nn.PReLU(feat_channels * 1),
            nn.Conv2d(feat_channels * 1, 2 * 1 * 9, 3, padding=1)
        )

    def upsample_input(self, inp, mask, upsample_factor):
        N, c, H, W = inp.shape
        mask = mask.view(N, 1, 9, upsample_factor, upsample_factor, H, W)
        mask = torch.softmax(mask, dim=2)
        inp = F.pad(inp, [1, 1, 1, 1], mode='replicate')
        up_inp = F.unfold(inp, [3, 3])
        up_inp = up_inp.view(N, c, 9, 1, 1, H, W)

        up_inp = torch.sum(mask * up_inp, dim=2)
        up_inp = up_inp.permute(0, 1, 4, 2, 5, 3)
        return up_inp.reshape(N, c, upsample_factor*H, upsample_factor*W)

    def forward(self, flow, feat0, feat1, last_occ):
        """ Upsample flow field [H/4, W/4, 4] -> [H, W, 4] using convex combination """
        N, _, H, W = flow.shape
        feat0_warped_list, feat1_warped_list = [], []
        for i in range(3):
            flow_bi = F.interpolate(flow * 2 ** i, scale_factor=2 ** i, mode='bilinear')
            feat0_warped = backwarp(feat0[2-i], flow_bi[:, :2])
            feat1_warped = backwarp(feat1[2-i], flow_bi[:, 2:])
            feat0_warped_list.append(feat0_warped)
            feat1_warped_list.append(feat1_warped)
        feature = torch.cat([feat0_warped_list[0], feat1_warped_list[0]], dim=1)
        feature0 = self.enc0(feature)
        feature1 = self.enc1(torch.cat([F.pixel_shuffle(feature0, 2), feat0_warped_list[1], feat1_warped_list[1]], dim=1))
        feature2 = self.enc2(torch.cat([F.pixel_shuffle(feature1, 2), feat0_warped_list[2], feat1_warped_list[2]], dim=1))
        mask_x2 = self.kernel_x2(feature1)
        mask_x4 = self.kernel_x4(feature2)
        mask_x2 = mask_x2.view(N, 18, H, 2, W, 2).permute(0, 1, 3, 5, 2, 4).contiguous()
        mask_x2_0, mask_x2_1 = torch.chunk(mask_x2, 2, dim=1)
        mask_x4 = mask_x4.view(N, 18, H, 4, W, 4).permute(0, 1, 3, 5, 2, 4).contiguous()
        mask_x4_0, mask_x4_1 = torch.chunk(mask_x4, 2, dim=1)
        up_flow_x2_0 = self.upsample_input(flow[:, :2] * 2, mask_x2_0, 2)
        up_flow_x2_1 = self.upsample_input(flow[:, 2:] * 2, mask_x2_1, 2)
        up_flow_x4_0 = self.upsample_input(flow[:, :2] * 4, mask_x4_0, 4)
        up_flow_x4_1 = self.upsample_input(flow[:, 2:] * 4, mask_x4_1, 4)
        up_flow_x2 = torch.cat([up_flow_x2_0, up_flow_x2_1], dim=1)
        up_flow_x4 = torch.cat([up_flow_x4_0, up_flow_x4_1], dim=1)
        up_occ = F.interpolate(last_occ, scale_factor=4, mode='bilinear')
        return [up_flow_x4, up_flow_x2, flow], up_occ
