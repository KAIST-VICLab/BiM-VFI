import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
losses = {}


def register(name):
    def decorator(cls):
        losses[name] = cls
        return cls

    return decorator


def make_loss_dict(loss_cfgs):
    loss_dict = dict()

    def make_loss(loss_spec):
        loss = losses[loss_spec['name']](**loss_spec['args'])
        return loss

    for loss_cfg in loss_cfgs:
        loss_dict[loss_cfg['name']] = make_loss(loss_cfg)

    return loss_dict


@register('l1')
class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()

    def forward(self, img0, img1):
        return F.l1_loss(input=img0, target=img1, reduction='mean')


@register('charbonnier')
class Charbonnier(nn.Module):
    def __init__(self, weight):
        super(Charbonnier, self).__init__()
        self.weight = weight

    def forward(self, imgt, imgt_pred, **kwargs):
        return (((imgt - imgt_pred) ** 2 + 1e-6) ** 0.5) * self.weight



@register('multiple_charbonnier')
class MultipleCharbonnier(nn.Module):
    def __init__(self, weight, gamma, **kwargs):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.charbonnier = Charbonnier(1)

    def forward(self, imgt_preds, imgt, **kwargs):
        loss_charbonnier = torch.Tensor([0]).cuda()
        for i in range(len(imgt_preds)):
            i_weight = self.gamma ** (len(imgt_preds) - i - 1)
            imgt_down = F.interpolate(imgt, scale_factor=1 / 2 ** (len(imgt_preds) - i - 1), mode='bilinear',
                                      align_corners=False, antialias=True)
            loss_charbonnier += self.charbonnier(imgt_preds[i], imgt_down).mean() * i_weight
        return loss_charbonnier * self.weight


@register('ternary')
class Ternary(nn.Module):
    def __init__(self, weight):
        super(Ternary, self).__init__()
        patch_size = 7
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape(
            (patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w).float().to(device)
        self.weight = weight

    # end

    def transform(self, img):
        patches = F.conv2d(img, self.w, padding=3, bias=None)
        transf = patches - img
        transf_norm = transf / torch.sqrt(0.81 + transf ** 2)
        return transf_norm

    # end

    def rgb2gray(self, rgb):
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    # end

    def hamming(self, t1, t2):
        dist = (t1 - t2) ** 2
        dist_norm = torch.mean(dist / (0.1 + dist), 1, True)
        return dist_norm

    # end

    def valid_mask(self, t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        mask = F.pad(inner, [padding] * 4)
        return mask

    # end

    def forward(self, imgt, imgt_pred, **kwargs):
        imgt = self.transform(self.rgb2gray(imgt))
        imgt_pred = self.transform(self.rgb2gray(imgt_pred))
        return (self.hamming(imgt, imgt_pred) * self.valid_mask(imgt, 1)) * self.weight
    # end


# end

@register('multiple_ternary')
class MultipleTernary(nn.Module):
    def __init__(self, weight, gamma, **kwargs):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.ternary = Ternary(1)

    def forward(self, imgt_preds, imgt, **kwargs):
        loss_ter = torch.Tensor([0]).cuda()
        for i in range(len(imgt_preds)):
            i_weight = self.gamma ** (len(imgt_preds) - i - 1)
            imgt_down = F.interpolate(imgt, scale_factor=1 / 2 ** (len(imgt_preds) - i - 1), mode='bilinear',
                                      align_corners=False, antialias=True)
            loss_ter += self.ternary(imgt_preds[i], imgt_down).mean() * i_weight
        return loss_ter * self.weight


@register('ada_charbonnier')
class AdaCharbonnierLoss(nn.Module):
    def __init__(self, weight) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, imgt_pred, imgt, weight, **kwargs):
        diff = imgt_pred - imgt
        loss = ((diff ** 2 + 1e-6) ** 0.5) * weight
        return loss.mean()


@register('multiple_flow_ada')
class MultipleFlowAdaLoss(nn.Module):
    def __init__(self, weight, gamma, beta=0.3) -> None:
        super().__init__()
        self.weight = weight
        self.beta = beta
        self.gamma = gamma
        self.ada_cb_loss = AdaCharbonnierLoss(1.0)

    def forward(self, flowt0_res_list, flowt1_res_list, flowt0_res_tea_list, flowt1_res_tea_list, flowt0_pred_list, flowt1_pred_list, flowt0, flowt1, **kwargs):
        robust_weight0 = self.get_mutli_flow_robust_weight(flowt0_pred_list[0], flowt0)
        robust_weight1 = self.get_mutli_flow_robust_weight(flowt1_pred_list[0], flowt1)
        loss = 0
        h, w = flowt0_res_list[0].shape[-2:]
        for lvl in range(0, len(flowt0_res_list)):
            h_lvl, w_lvl = flowt0_res_list[lvl].shape[-2:]
            scale_factor = h_lvl / h
            loss = loss + self.ada_cb_loss(**{
                'imgt_pred': flowt0_res_list[lvl],
                'imgt': flowt0_res_tea_list[lvl],
                'weight': F.interpolate(robust_weight0, scale_factor=scale_factor / 4, mode='bilinear')
            }) * self.gamma ** lvl
            loss = loss + self.ada_cb_loss(**{
                'imgt_pred': flowt1_res_list[lvl],
                'imgt': flowt1_res_tea_list[lvl],
                'weight': F.interpolate(robust_weight1, scale_factor=scale_factor / 4, mode='bilinear')
            }) * self.gamma ** lvl
        return loss * self.weight

    def resize(self, x, scale_factor):
        return scale_factor * F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)

    def get_mutli_flow_robust_weight(self, flow_pred, flow_gt):
        dims = flow_pred.shape
        if len(dims) == 5:
            b, num_flows, c, h, w = dims
        else:
            b, c, h, w = dims
            num_flows = 1
        flow_pred = flow_pred.view(b, num_flows, c, h, w)
        flow_gt = flow_gt.repeat(1, num_flows, 1, 1).view(b, num_flows, c, h, w)
        epe = ((flow_pred.detach() - flow_gt) ** 2).sum(dim=2, keepdim=True).max(1)[0] ** 0.5
        robust_weight = torch.exp(-self.beta * epe)
        return robust_weight


@register('multiple_flow')
class MultipleFlowLoss(nn.Module):
    def __init__(self, weight, gamma, beta=0.3) -> None:
        super().__init__()
        self.weight = weight
        self.beta = beta
        self.gamma = gamma
        self.ada_cb_loss = Charbonnier(1.0)

    def forward(self, flowt0_res_list, flowt1_res_list, flowt0_res_tea_list, flowt1_res_tea_list, **kwargs):
        loss = 0
        h, w = flowt0_res_list[0].shape[-2:]
        for lvl in range(0, len(flowt0_res_list)):
            h_lvl, w_lvl = flowt0_res_list[lvl].shape[-2:]
            loss = loss + self.ada_cb_loss(**{
                'imgt_pred': flowt0_res_list[lvl],
                'imgt': flowt0_res_tea_list[lvl],
            }).mean() * self.gamma ** lvl
            loss = loss + self.ada_cb_loss(**{
                'imgt_pred': flowt1_res_list[lvl],
                'imgt': flowt1_res_tea_list[lvl],
            }).mean() * self.gamma ** lvl
        return loss * self.weight

    def resize(self, x, scale_factor):
        return scale_factor * F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)

    def get_mutli_flow_robust_weight(self, flow_pred, flow_gt):
        dims = flow_pred.shape
        if len(dims) == 5:
            b, num_flows, c, h, w = dims
        else:
            b, c, h, w = dims
            num_flows = 1
        flow_pred = flow_pred.view(b, num_flows, c, h, w)
        flow_gt = flow_gt.repeat(1, num_flows, 1, 1).view(b, num_flows, c, h, w)
        epe = ((flow_pred.detach() - flow_gt) ** 2).sum(dim=2, keepdim=True).max(1)[0] ** 0.5
        # robust_weight = torch.exp(-self.beta * epe)
        robust_weight = torch.ones_like(epe)
        return robust_weight


@register('lap')
class LapLoss(torch.nn.Module):
    @staticmethod
    def gauss_kernel(size=5, channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    @staticmethod
    def laplacian_pyramid(img, kernel, max_levels=3):
        def downsample(x):
            return x[:, :, ::2, ::2]

        def upsample(x):
            cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).to(device)], dim=3)
            cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
            cc = cc.permute(0, 1, 3, 2)
            cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2).to(device)], dim=3)
            cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
            x_up = cc.permute(0, 1, 3, 2)
            return conv_gauss(x_up, 4 * LapLoss.gauss_kernel(channels=x.shape[1]))

        def conv_gauss(img, kernel):
            img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='replicate')
            out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
            return out

        current = img
        pyr = []
        for level in range(max_levels):
            filtered = conv_gauss(current, kernel)
            down = downsample(filtered)
            up = upsample(down)
            diff = current - up
            pyr.append(diff)
            current = down
        return pyr

    def __init__(self, max_levels=5, channels=3):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = LapLoss.gauss_kernel(channels=channels)

    def forward(self, imgt_pred, imgt):
        pyr_pred = LapLoss.laplacian_pyramid(
            img=imgt_pred, kernel=self.gauss_kernel, max_levels=self.max_levels)
        pyr_target = LapLoss.laplacian_pyramid(
            img=imgt, kernel=self.gauss_kernel, max_levels=self.max_levels)
        return sum(torch.nn.functional.l1_loss(a, b) for a, b in zip(pyr_pred, pyr_target))


@register('photo_teacher')
class PhotoTeacherLoss(nn.Module):
    def __init__(self, weight, gamma):
        super(PhotoTeacherLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.ternary = Ternary(weight)
        self.charbonnier = Charbonnier(weight)

    def forward(self, warped_img0_tea_list, warped_img1_tea_list, imgt, **kwargs):
        loss_ternary, loss_charbonnier = 0, 0
        warped_img0_tea_list = warped_img0_tea_list[::-1]
        warped_img1_tea_list = warped_img1_tea_list[::-1]
        for i in range(len(warped_img0_tea_list)):
            imgt_down = F.interpolate(imgt, scale_factor=1 / 2 ** i, mode='bilinear', antialias=True)
            loss_ternary += self.ternary(warped_img0_tea_list[i], imgt_down).mean() * (self.gamma ** i) + self.ternary(
                warped_img1_tea_list[i], imgt_down).mean() * (self.gamma ** i)
            loss_charbonnier += self.charbonnier(warped_img0_tea_list[i], imgt_down).mean() * (
                        self.gamma ** i) + self.charbonnier(warped_img1_tea_list[i], imgt_down).mean() * (
                                            self.gamma ** i)
        return (loss_ternary + loss_charbonnier) / 2 * self.weight


@register('photo_teacher')
class PhotoTeacherLoss(nn.Module):
    def __init__(self, weight, gamma):
        super(PhotoTeacherLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.ternary = Ternary(weight)
        self.charbonnier = Charbonnier(weight)

    def forward(self, interp_imgs_tea, imgt, **kwargs):
        loss_ternary, loss_charbonnier = 0, 0
        interp_imgs_tea = interp_imgs_tea[::-1]
        for i in range(len(interp_imgs_tea)):
            imgt_pred_tea = interp_imgs_tea[i]
            imgt_down = F.interpolate(imgt, scale_factor=1 / 2 ** i, mode='bilinear', antialias=True)
            loss_ternary = loss_ternary + self.ternary(imgt_pred_tea, imgt_down).mean() * (self.gamma ** i)
            loss_charbonnier = loss_charbonnier + self.charbonnier(imgt_pred_tea, imgt_down).mean() * (self.gamma ** i)
        return (loss_ternary + loss_charbonnier).mean() * self.weight



@register('flow_distill_res')
class FlowDistillResLoss(nn.Module):
    def __init__(self, weight, gamma, beta=0.3, zero_before=0) -> None:
        super().__init__()
        self.weight = weight
        self.beta = beta
        self.gamma = gamma
        self.zero_before = zero_before
        self.charbonnier = AdaCharbonnierLoss(1.0)

    def forward(self, flowt0_res_list, flowt1_res_list, flowt0_res_tea_list, flowt1_res_tea_list, **kwargs):
        loss = 0.
        for i in range(len(flowt0_res_list)):
            b, _, h, w = flowt0_res_list[i].shape
            loss_mask0 = ((flowt0_res_tea_list[i].detach().clone() ** 2).sum(1, True) < 32 ** 2).float()
            loss_mask1 = ((flowt1_res_tea_list[i].detach().clone() ** 2).sum(1, True) < 32 ** 2).float()
            loss = loss + ((self.charbonnier(flowt0_res_list[i] - flowt0_res_tea_list[
                i].detach().clone())) * loss_mask0).mean() * self.gamma ** i
            loss = loss + ((self.charbonnier(flowt1_res_list[i] - flowt1_res_tea_list[
                i].detach().clone())) * loss_mask1).mean() * self.gamma ** i
        return loss * self.weight


    def charbonnier(self, inp):
        return (inp ** 2 + (1e-6) ** 2) ** 0.5


@register('flow_teacher_reg')
class FlowTeacherRegLoss(nn.Module):
    def __init__(self, weight, gamma) -> None:
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.ada_cb_loss = Charbonnier(1)

    def forward(self, flow0t_tea_list, flowt1_tea_list, **kwargs):
        loss = 0
        for lvl in range(0, len(flow0t_tea_list)):
            loss = loss + ((flow0t_tea_list[lvl])).abs().mean() * self.gamma ** lvl
            loss = loss + ((flowt1_tea_list[lvl])).abs().mean() * self.gamma ** lvl
        return loss * self.weight


@register('flow_smooth_tea1')
class FlowSmoothnessTeacher1Loss(nn.Module):
    def __init__(self, weight, gamma, weight_type='exp', edge_constant=1):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        if weight_type == 'exp':
            self.power = 1
        else:
            self.power = 2
        self.edge_constant = edge_constant

    def gradient(self, inp):
        return (inp[:, :, :, 1:] - inp[:, :, :, :-1]), (inp[:, :, 1:] - inp[:, :, :-1])

    def forward(self, flowt0_pred_tea_list, flowt1_pred_tea_list, imgt, **kwargs):
        b, c, h, w = imgt.shape
        loss = torch.tensor([0.], device=imgt.device)
        for i in range(len(flowt0_pred_tea_list)):
            imgt_down = F.interpolate(imgt, scale_factor=1 / 2 ** (i), mode='bilinear', antialias=True)
            loss_mask0 = torch.ones(b, 1, h // 2**i, w // 2**i, device=imgt.device)
            loss_mask1 = torch.ones(b, 1, h // 2**i, w // 2**i, device=imgt.device)
            imgt_dx, imgt_dy = self.gradient(imgt_down)
            flowt0_dx, flowt0_dy = self.gradient(flowt0_pred_tea_list[i])
            flowt1_dx, flowt1_dy = self.gradient(flowt1_pred_tea_list[i])
            weights_x = torch.exp(-(self.edge_constant * torch.mean(torch.abs(imgt_dx), 1, keepdim=True)) ** self.power)
            weights_y = torch.exp(-(self.edge_constant * torch.mean(torch.abs(imgt_dy), 1, keepdim=True)) ** self.power)
            loss += (self.charbonnier(flowt0_dx) * weights_x * loss_mask0[:, :, :, 1:]).mean() * self.gamma ** i
            loss += (self.charbonnier(flowt1_dx) * weights_x * loss_mask1[:, :, :, 1:]).mean() * self.gamma ** i
            loss += (self.charbonnier(flowt0_dy) * weights_y * loss_mask0[:, :, 1:]).mean() * self.gamma ** i
            loss += (self.charbonnier(flowt1_dy) * weights_y * loss_mask1[:, :, 1:]).mean() * self.gamma ** i
        return loss * self.weight

    def charbonnier(self, inp):
        return (inp ** 2 + (1e-6) ** 2) ** 0.5


@register('flow_tea')
class FlowTeacherLoss(nn.Module):
    def __init__(self, weight, gamma):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def gradient(self, inp):
        return (inp[:, :, :, 1:] - inp[:, :, :, :-1]), (inp[:, :, 1:] - inp[:, :, :-1])

    def forward(self, flowt0_pred_tea_list, flowt1_pred_tea_list, flowt0, flowt1, **kwargs):
        loss = torch.tensor([0], dtype=torch.float32, device=flowt0.device)
        for i in range(len(flowt0_pred_tea_list)):
            flowt0_down = F.interpolate(flowt0, scale_factor=1 / 2 ** i, mode='bilinear') / 2 ** i
            flowt1_down = F.interpolate(flowt1, scale_factor=1 / 2 ** i, mode='bilinear') / 2 ** i
            loss += (self.charbonnier(flowt0_pred_tea_list[i] - flowt0_down)).mean() * self.gamma ** i
            loss += (self.charbonnier(flowt1_pred_tea_list[i] - flowt1_down)).mean() * self.gamma ** i
        return loss * self.weight

    def charbonnier(self, inp):
        return (inp ** 2 + (1e-6) ** 2) ** 0.5
