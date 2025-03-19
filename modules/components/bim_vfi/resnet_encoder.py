import torch.nn as nn
from typing import Optional, Callable
from torch import Tensor
from functools import partial


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=True,
        dilation=dilation,
    )


def conv2x2(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """2x2 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=2, stride=stride)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.InstanceNorm2d, data_format='channels_first')
            # norm_layer = nn.Identity
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bn1 = norm_layer(inplanes)
        if stride == 1:
            self.conv1 = conv3x3(inplanes, planes, stride)
        else:
            self.conv1 = conv2x2(inplanes, planes, stride)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.bn2 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.lrelu(out)

        out = self.bn2(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.lrelu(out)
        out = out + identity

        return out


class ResNetPyramid(nn.Module):
    """A 3-level feature pyramid, which by default is shared by the motion
    estimator and synthesis network.
    """

    def __init__(self, feat_channels: int):
        super(ResNetPyramid, self).__init__()
        self.conv = nn.Conv2d(3, feat_channels, kernel_size=3, stride=1, padding=1)
        self.layer0 = nn.Sequential(
            BasicBlock(feat_channels, feat_channels, norm_layer=nn.InstanceNorm2d),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels * 2, 2, 2),
            BasicBlock(feat_channels * 2, feat_channels * 2, norm_layer=nn.InstanceNorm2d),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(feat_channels * 2, feat_channels * 4, 2, 2),
            BasicBlock(feat_channels * 4, feat_channels * 4, norm_layer=nn.InstanceNorm2d),
        )
        self.conv_last = nn.Conv2d(feat_channels * 4, feat_channels * 4, 3, 1, 1)

    def forward(self, img):
        C0 = self.layer0(self.conv(img))
        C1 = self.layer1(C0)
        C2 = self.conv_last(self.layer2(C1))
        return [C0, C1, C2]