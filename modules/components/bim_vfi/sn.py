import torch
import torch.nn as nn

from .backwarp import backwarp


class SynthesisNetwork(nn.Module):
    def __init__(self, feat_channels):
        super(SynthesisNetwork, self).__init__()
        input_channels = 6 + 1
        self.conv_down1 = nn.Sequential(
            nn.Conv2d(input_channels, feat_channels, 7, padding=3),
            nn.PReLU(feat_channels),
            nn.Conv2d(feat_channels, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2))
        self.conv_down2 = nn.Sequential(
            nn.Conv2d(feat_channels * 4, feat_channels * 2, 2, stride=2, padding=0),
            nn.PReLU(feat_channels * 2),
            nn.Conv2d(feat_channels * 2, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2),
            nn.Conv2d(feat_channels * 2, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2))
        self.conv_down3 = nn.Sequential(
            nn.Conv2d(feat_channels * 6, feat_channels * 4, 2, stride=2, padding=0),
            nn.PReLU(feat_channels * 4),
            nn.Conv2d(feat_channels * 4, feat_channels * 4, 3, padding=1),
            nn.PReLU(feat_channels * 4),
            nn.Conv2d(feat_channels * 4, feat_channels * 4, 3, padding=1),
            nn.PReLU(feat_channels * 4))
        self.conv_up1 = nn.Sequential(
            torch.nn.Conv2d(feat_channels * 12, feat_channels * 8, 3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(feat_channels * 2),
            nn.Conv2d(feat_channels * 2, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2))
        self.conv_up2 = nn.Sequential(
            torch.nn.Conv2d(feat_channels * 4, feat_channels * 4, 3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(feat_channels * 1),
            nn.Conv2d(feat_channels * 1, feat_channels * 1, 3, padding=1),
            nn.PReLU(feat_channels * 1))
        self.conv_up3 = nn.Sequential(
            nn.Conv2d(feat_channels * 3, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2),
            nn.Conv2d(feat_channels * 2, feat_channels * 2, 3, padding=1),
            nn.PReLU(feat_channels * 2),
        )
        self.conv_out = nn.Conv2d(feat_channels * 2, 4, 3, padding=1)

    def get_warped_representations(self, bi_flow, c0, c1, i0=None, i1=None):
        flow_t0 = bi_flow[:, :2]
        flow_t1 = bi_flow[:, 2:4]
        warped_c0 = backwarp(c0, flow_t0)
        warped_c1 = backwarp(c1, flow_t1)
        if (i0 is None) and (i1 is None):
            return warped_c0, warped_c1
        else:
            warped_img0 = backwarp(i0, flow_t0)
            warped_img1 = backwarp(i1, flow_t1)
            return warped_img0, warped_img1, warped_c0, warped_c1

    def forward(self, i0, i1, c0_pyr, c1_pyr, bi_flow_pyr, occ):
        warped_img0, warped_img1, warped_c0, warped_c1 = \
            self.get_warped_representations(
                bi_flow_pyr[0], c0_pyr[0], c1_pyr[0], i0, i1)
        input_feat = torch.cat(
            (warped_img0, warped_img1, occ), 1)
        s0 = self.conv_down1(input_feat)
        s1 = self.conv_down2(torch.cat((s0, warped_c0, warped_c1), 1))
        warped_c0, warped_c1 = self.get_warped_representations(
            bi_flow_pyr[1], c0_pyr[1], c1_pyr[1], None, None)
        s2 = self.conv_down3(torch.cat((s1, warped_c0, warped_c1), 1))
        warped_c0, warped_c1 = self.get_warped_representations(
            bi_flow_pyr[2], c0_pyr[2], c1_pyr[2], None, None)

        x = self.conv_up1(torch.cat((s2, warped_c0, warped_c1), 1))
        x = self.conv_up2(torch.cat((x, s1), 1))
        x = self.conv_up3(torch.cat((x, s0), 1))

        refine = self.conv_out(x)
        refine_res = refine[:, :3]
        occ_res = refine[:, 3:]
        occ_out = occ + occ_res
        blending_mask = torch.sigmoid(occ_out)
        merged_img = (warped_img0 * blending_mask + warped_img1 * (1 - blending_mask)) + refine_res
        interp_img = merged_img

        extra_dict = {}
        extra_dict["refine_res"] = refine_res
        extra_dict["refine_mask"] = occ_out
        extra_dict["warped_img0"] = warped_img0
        extra_dict["warped_img1"] = warped_img1
        extra_dict["merged_img"] = merged_img

        return interp_img, occ_out, extra_dict
