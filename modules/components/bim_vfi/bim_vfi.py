import torch
import torch.nn.functional as F
import torch.nn as nn

from .backwarp import backwarp
from .resnet_encoder import ResNetPyramid
from .caun import CAUN
from .bimfn import BiMFN
from .sn import SynthesisNetwork

from ..components import register

from utils.padder import InputPadder


@register('bim_vfi')
class BiMVFI(nn.Module):
    def __init__(self, pyr_level=3, feat_channels=32, **kwargs):
        super(BiMVFI, self).__init__()
        self.pyr_level = pyr_level
        self.mfe = ResNetPyramid(feat_channels)
        self.cfe = ResNetPyramid(feat_channels)
        self.bimfn = BiMFN(feat_channels)
        self.sn = SynthesisNetwork(feat_channels)
        self.feat_channels = feat_channels
        self.caun = CAUN(feat_channels)

    def forward_one_lvl(self, img0, img1, last_flow, last_occ, teacher_input_dict=None, time_period=0.5):
        teacher_dict = dict()
        ### Extract motion features and context features of I0 and I1
        feat0_pyr = self.mfe(img0)
        feat1_pyr = self.mfe(img1)
        cfeat0_pyr = self.cfe(img0)
        cfeat1_pyr = self.cfe(img1)

        B, _, H, W = feat0_pyr[-1].shape
        if 'imgt_this_lvl' in teacher_input_dict:
            ### If it is training, do KDVCF
            featt_pyr = self.mfe(teacher_input_dict['imgt_this_lvl'])

            ### Prepare M_t->0t and M_t->t1
            r_0 = torch.zeros(B, 1, H, W, device=feat0_pyr[-1].device)
            r_1 = torch.ones(B, 1, H, W, device=feat0_pyr[-1].device)
            phi_tea0 = torch.rand(B, 1, H, W, device=feat0_pyr[-1].device) * torch.pi * 2
            phi_tea0 = torch.cat([torch.cos(phi_tea0), torch.sin(phi_tea0)], dim=1)
            phi_tea1 = torch.rand(B, 1, H, W, device=feat0_pyr[-1].device) * torch.pi * 2
            phi_tea1 = torch.cat([torch.cos(phi_tea1), torch.sin(phi_tea1)], dim=1)

            ### Prepare flows and occlusion masks for teacher process
            last_flow_for_tea = F.interpolate(
                input=last_flow.detach().clone(), scale_factor=0.5,
                mode="bilinear", align_corners=False) * 0.5
            last_occ_for_tea = F.interpolate(
                input=last_occ.detach().clone(), scale_factor=0.5,
                mode="bilinear", align_corners=False)
            last_flow_0t = torch.cat((last_flow_for_tea[:, :2], torch.zeros_like(last_flow_for_tea[:, :2])), dim=1)
            last_flow_t1 = torch.cat((torch.zeros_like(last_flow_for_tea[:, 2:]), last_flow_for_tea[:, 2:]), dim=1)

            ### Get teacher flows V_t->t|0t and V_t->0
            flow_0t_low, flow_0t_res = self.bimfn(
                feat0_pyr[-1], featt_pyr[-1], r_1, phi_tea0, last_flow_0t, last_occ_for_tea
            )
            ### Get teacher flows V_t->t|t1 and V_t->1
            flow_t1_low, flow_1t_res = self.bimfn(
                featt_pyr[-1], feat1_pyr[-1], r_0, phi_tea1, last_flow_t1, last_occ_for_tea
            )

            ### Calculate BiM of student process
            flow_t0_res_tea = (flow_0t_res[:, :2]).detach().clone()
            flow_t1_res_tea = (flow_1t_res[:, 2:]).detach().clone()
            flow_t0_r_tea = torch.norm(flow_t0_res_tea, dim=1, keepdim=True)
            flow_t1_r_tea = torch.norm(flow_t1_res_tea, dim=1, keepdim=True)
            flow_sin_tea = (flow_t0_res_tea[:, 0:1] * flow_t1_res_tea[:, 1:2] - flow_t0_res_tea[:,
                                                                                1:2] * flow_t1_res_tea[:, 0:1]) / (
                                   flow_t0_r_tea * flow_t1_r_tea)
            flow_cos_tea = (flow_t0_res_tea[:, 0:1] * flow_t1_res_tea[:, 0:1] + flow_t0_res_tea[:,
                                                                                1:2] * flow_t1_res_tea[:, 1:2]) / (
                                   flow_t0_r_tea * flow_t1_r_tea)
            flow_tea_low = torch.cat([flow_0t_low[:, :2], flow_t1_low[:, 2:]], dim=1)

            ### Upsample flows of teacher process
            bi_flow_tea_pyr, occ_tea = self.caun(flow_tea_low, cfeat0_pyr, cfeat1_pyr, last_occ_for_tea)
            flow_t0_tea = bi_flow_tea_pyr[0][:, :2]
            flow_t1_tea = bi_flow_tea_pyr[0][:, 2:]

            ### Interpolate image at current level for teacher process
            interp_img_tea, occ_tea, teacher_extra_dict = self.sn(
                img0, img1, cfeat0_pyr, cfeat1_pyr, bi_flow_tea_pyr, occ_tea
            )
            teacher_dict['flow_t0_tea'] = flow_t0_tea
            teacher_dict['flow_t0_res_tea'] = flow_t0_res_tea
            teacher_dict['flow_t1_tea'] = flow_t1_tea
            teacher_dict['flow_t1_res_tea'] = flow_t1_res_tea
            teacher_dict['interp_img_tea'] = interp_img_tea
            teacher_dict['flow_0t_res'] = flow_0t_res
            teacher_dict['flow_t1_res'] = flow_1t_res
            r = ((flow_t0_r_tea) / (flow_t1_r_tea + flow_t0_r_tea))
            phi = torch.cat([(flow_cos_tea), (flow_sin_tea)], dim=1)
            r, phi = r.detach().clone(), phi.detach().clone()
        else:
            # If it is not training, prepare uniform BiM
            r = torch.ones((B, 1, H, W), device=feat0_pyr[-1].device) * time_period
            phi = torch.ones((B, 1, H, W), device=feat0_pyr[-1].device) * torch.pi
            phi = torch.cat([torch.cos(phi), torch.sin(phi)], dim=1)

        last_flow = F.interpolate(
            input=last_flow.detach().clone(), scale_factor=0.5,
            mode="bilinear", align_corners=False) * 0.5
        last_occ = F.interpolate(
            input=last_occ.detach().clone(), scale_factor=0.5,
            mode="bilinear", align_corners=False)

        ### Get student flows V_t->0 and V_t->1
        flow_low, flow_res = self.bimfn(
            feat0_pyr[-1], feat1_pyr[-1], r, phi, last_flow, last_occ)

        ### Upsample student flows
        bi_flow_pyr, occ = self.caun(flow_low, cfeat0_pyr, cfeat1_pyr, last_occ)
        flow = bi_flow_pyr[0]

        ### Interpolate image at current level for student process
        interp_img, occ, extra_dict = self.sn(
            img0, img1, cfeat0_pyr, cfeat1_pyr, bi_flow_pyr, occ)
        extra_dict.update({'flow_res': flow_res})
        return flow, occ, interp_img, extra_dict, teacher_dict

    def forward(self, img0, img1, time_step,
                pyr_level=None, imgt=None, run_with_gt=False, **kwargs):
        if pyr_level is None: pyr_level = self.pyr_level
        N, _, H, W = img0.shape
        flowt0_pred_list = []
        flowt0_res_list = []
        flowt1_pred_list = []
        flowt1_res_list = []
        flow0t_tea_list = []
        flowt1_tea_list = []
        flowt0_pred_tea_list = []
        flowt0_res_tea_list = []
        flowt1_pred_tea_list = []
        flowt1_res_tea_list = []
        refine_mask_tea_list = []
        interp_imgs = []
        interp_imgs_tea = []

        padder = InputPadder(img0.shape, divisor=int(2 ** (pyr_level + 1)))

        ### Normalize input images
        with torch.set_grad_enabled(False):
            tenStats = [img0, img1]
            if self.training or run_with_gt:
                tenStats.append(imgt)
            tenMean_ = sum([tenIn.mean([1, 2, 3], True) for tenIn in tenStats]) / len(tenStats)
            tenStd_ = (sum([tenIn.std([1, 2, 3], False, True).square() + (
                    tenMean_ - tenIn.mean([1, 2, 3], True)).square() for tenIn in tenStats]) / len(tenStats)).sqrt()

            img0 = (img0 - tenMean_) / (tenStd_ + 0.0000001)
            img1 = (img1 - tenMean_) / (tenStd_ + 0.0000001)
            if self.training or run_with_gt:
                imgt = (imgt - tenMean_) / (tenStd_ + 0.0000001)

        ### Pad images for downsampling
        img0, img1 = padder.pad(img0, img1)
        if self.training or run_with_gt:
            imgt = padder.pad(imgt)

        N, _, H, W = img0.shape
        teacher_input_dict = dict()

        for level in list(range(pyr_level))[::-1]:
            ### Downsample images if needed
            if level != 0:
                scale_factor = 1 / 2 ** level
                img0_this_lvl = F.interpolate(
                    input=img0, scale_factor=scale_factor,
                    mode="bilinear", align_corners=False, antialias=True)
                img1_this_lvl = F.interpolate(
                    input=img1, scale_factor=scale_factor,
                    mode="bilinear", align_corners=False, antialias=True)
                if self.training or run_with_gt:
                    imgt_this_lvl = F.interpolate(
                        input=imgt, scale_factor=scale_factor,
                        mode="bilinear", align_corners=False, antialias=True)
                    teacher_input_dict['imgt_this_lvl'] = imgt_this_lvl
            else:
                img0_this_lvl = img0
                img1_this_lvl = img1
                if self.training or run_with_gt:
                    imgt_this_lvl = imgt
                    teacher_input_dict['imgt_this_lvl'] = imgt_this_lvl

            ### Initialize zero flows for lowest pyramid level
            if level == pyr_level - 1:
                last_flow = torch.zeros(
                    (N, 4, H // (2 ** (level + 1)), W // (2 ** (level + 1))), device=img0.device
                )
                last_occ = torch.zeros(N, 1, H // (2 ** (level + 1)), W // (2 ** (level + 1)), device=img0.device)
            else:
                last_flow = flow
                last_occ = occ

            ### Single pyramid level run
            flow, occ, interp_img, extra_dict, teacher_dict = self.forward_one_lvl(
                img0_this_lvl, img1_this_lvl, last_flow, last_occ, teacher_input_dict, time_step)

            flowt0_pred_list.append((flow[:, :2]))
            flowt1_pred_list.append((flow[:, 2:]))
            flowt0_res_list.append(extra_dict['flow_res'][:, :2])
            flowt1_res_list.append(extra_dict['flow_res'][:, 2:])
            interp_imgs.append((interp_img) * (tenStd_ + 0.0000001) + tenMean_)
            if self.training or run_with_gt:
                flowt0_pred_tea_list.append(
                    (teacher_dict['flow_t0_tea']))
                flowt1_pred_tea_list.append(
                    (teacher_dict['flow_t1_tea']))
                flowt0_res_tea_list.append(teacher_dict['flow_t0_res_tea'])
                flowt1_res_tea_list.append(teacher_dict['flow_t1_res_tea'])
                interp_imgs_tea.append((teacher_dict['interp_img_tea']) * (tenStd_ + 0.0000001) + tenMean_)
                flow0t_tea_list.append(teacher_dict['flow_0t_res'][:, 2:])
                flowt1_tea_list.append(teacher_dict['flow_t1_res'][:, :2])

        result_dict = {
            "imgt_preds": interp_imgs, "flowt0_pred_list": flowt0_pred_list[::-1],
            "flowt1_pred_list": flowt1_pred_list[::-1],
            'imgt_pred': padder.unpad(interp_imgs[-1].contiguous()),
            'flowt0_pred_tea_list': flowt0_pred_tea_list[::-1], 'flowt1_pred_tea_list': flowt1_pred_tea_list[::-1],
            'interp_imgs_tea': interp_imgs_tea, 'refine_mask_tea': refine_mask_tea_list,
            'flowt0_res_list': flowt0_res_list[::-1], 'flowt1_res_list': flowt1_res_list[::-1],
            'flowt0_res_tea_list': flowt0_res_tea_list[::-1], 'flowt1_res_tea_list': flowt1_res_tea_list[::-1],
            'flow0t_tea_list': flow0t_tea_list[::-1], 'flowt1_tea_list': flowt1_tea_list[::-1],
        }

        return result_dict
