import torch
import torch.nn as nn

from .backwarp import backwarp
from .arch import LayerNorm, ResBlock
from .costvol import costvol_func


class BiMMConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_FV = nn.Sequential(
            LayerNorm(in_channels, data_format='channels_first'),
            ResBlock(out_channels, 1)
        )
        self.conv_r = nn.Sequential(
            LayerNorm(in_channels, data_format='channels_first'),
            ResBlock(out_channels, 1)
        )
        self.conv_phi = nn.Sequential(
            LayerNorm(in_channels, data_format='channels_first'),
            ResBlock(out_channels, 1)
        )

    def forward(self, FV, r, phi):
        FV_out1 = self.conv_FV(FV)
        r_out = self.conv_r(r)
        phi_out = self.conv_phi(phi)
        FV_out2 = FV_out1 + FV_out1 * r_out * phi_out
        return FV_out2, r_out, phi_out


class BiMFN(nn.Module):
    def __init__(self, feat_channels):
        super(BiMFN, self).__init__()
        self.conv_flow = nn.Sequential(
            nn.Conv2d(2, feat_channels * 2, 7, padding=3),
            nn.PReLU(feat_channels * 2),
            nn.Conv2d(feat_channels * 2, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2),
        )
        self.conv_occ = nn.Sequential(
            nn.Conv2d(1, feat_channels * 2, 7, padding=3),
            nn.PReLU(feat_channels * 2),
            nn.Conv2d(feat_channels * 2, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2),
        )
        self.conv_corr = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv2d(81, feat_channels * 2, 1, padding=0),
            nn.PReLU(feat_channels * 2),
            nn.Conv2d(feat_channels * 2, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2),
        )
        self.conv0 = nn.Sequential(
            nn.Conv2d(feat_channels * 18, feat_channels * 12, 1, padding=0),
            nn.PReLU(feat_channels * 12),
            nn.Conv2d(feat_channels * 12, feat_channels * 12, 3, padding=1),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(feat_channels * 12, feat_channels * 8, 1, padding=0),
            nn.PReLU(feat_channels * 8),
            nn.Conv2d(feat_channels * 8, feat_channels * 8, 3, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(feat_channels * 8, feat_channels * 6, 1, padding=0),
            nn.PReLU(feat_channels * 6),
            nn.Conv2d(feat_channels * 6, feat_channels * 6, 3, padding=1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(feat_channels * 6, feat_channels * 6, 1, padding=0),
            nn.PReLU(feat_channels * 6),
            nn.Conv2d(feat_channels * 6, feat_channels * 6, 3, padding=1),
        )
        self.dem = nn.Sequential(
            nn.Conv2d(1, feat_channels * 4, 1, padding=0),
            nn.PReLU(feat_channels * 4),
            nn.Conv2d(feat_channels * 4, feat_channels * 6, 1, padding=0),
        )
        self.aem = nn.Sequential(
            nn.Conv2d(2, feat_channels * 4, 1, padding=0),
            nn.PReLU(feat_channels * 4),
            nn.Conv2d(feat_channels * 4, feat_channels * 6, 1, padding=0),
        )

        self.bim_mconv = BiMMConv(feat_channels * 6, feat_channels * 6)

        self.conv_out = nn.Sequential(
            nn.Conv2d(feat_channels * 6, feat_channels * 4, 1, padding=0),
            nn.PReLU(feat_channels * 4),
            nn.Conv2d(feat_channels * 4, 4, 1, padding=0))

    def forward(self, feat0, feat1, r, phi, last_flow, last_occ):
        feat0_warp = backwarp(feat0, (last_flow[:, :2]))
        feat1_warp = backwarp(feat1, (last_flow[:, 2:]))
        volume0 = costvol_func.apply(feat0_warp, feat1_warp, 9)
        volume1 = costvol_func.apply(feat1_warp, feat0_warp, 9)
        corr0 = self.conv_corr(volume0)
        corr1 = self.conv_corr(volume1)
        flo0 = self.conv_flow(last_flow[:, :2])
        flo1 = self.conv_flow(last_flow[:, 2:])
        occ = self.conv_occ(last_occ)
        input_feat = torch.cat([corr0, corr1, feat0_warp, feat1_warp, flo0, flo1, occ], 1)
        FV = self.conv0(input_feat)
        FV = self.conv1(FV)
        FV = self.conv2(FV)
        FV0 = self.conv3(FV)
        r0 = self.dem(r)
        phi0 = self.aem(phi)
        bim_feat, _, _ = self.bim_mconv(FV0, r0, phi0)
        flow_res = self.conv_out(bim_feat)
        flow_low = flow_res + last_flow

        return flow_low, flow_res
