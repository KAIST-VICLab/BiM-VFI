import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            x = x.permute(0, 2, 3, 1)
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)

class ResBlock(nn.Module):
    def __init__(self, feat_channels, kernel_size=3, padding_mode='zeros'):
        super().__init__()
        self.conv1 = nn.Conv2d(feat_channels, feat_channels, kernel_size, padding=(kernel_size - 1) // 2,
                               padding_mode=padding_mode)
        self.act = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(feat_channels, feat_channels, kernel_size, padding=(kernel_size - 1) // 2,
                               padding_mode=padding_mode)

    def forward(self, x):
        inp = x
        x = self.conv2(self.act(self.conv1(x)))
        return inp + x