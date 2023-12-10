import math
from thop import profile
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class KFNet(nn.Module):
    def __init__(self, num_classes = 1000):
        super(KFNet, self).__init__()

        self.stem = nn.Sequential(nn.Conv2d(3, 96, kernel_size=(4,4), stride=(4,4)),
                             LayerNorm(96, eps=1e-6, data_format="channels_first"))
        # self.features_1 = nn.Sequential(
        #     GhostBottleneck(96, 192, has_se=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        self.features_2 = self.make_layer(drop_path_rate=0.2)
        self.norm = LayerNorm(768, eps=1e-6, data_format="channels_last")  # final norm layer
        self.head = nn.Linear(768, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_out",  nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        # x = self.features_2(x)
        # x = self.norm(x.mean([-2, -1]))
        # x = self.head(x)
        return x

    def make_layer(self,drop_path_rate: float = 0.):

        stages = []
        cfgs = [[96,96,0],[96,96,0],[96,192,1],[192,192,0],[192,192,0],[192,384,1],[384,384,0],[384,384,0],[384,384,0],[384,384,0],[384,384,0],[384,384,0],[384,384,0],[384,768,1],[768,768,0],[768,768,0],[768,768,0]]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, len(cfgs) * 2)]

        for num, cfg in enumerate(cfgs):

            stages += nn.Sequential(
                KFNet_block(cfg[0], cfg[1],drop_rate=dp_rates[num * 2], has_se=True, if_dw=True),
                KFNet_block(cfg[1], cfg[1],drop_rate=dp_rates[num * 2 + 1]),
            )
            if cfg[2] == 1:
                downsample_layer = nn.Sequential(LayerNorm(cfg[1], eps=1e-6, data_format="channels_first"),
                                                 nn.Conv2d(cfg[1], cfg[1], kernel_size=2, stride=2))
                stages += [downsample_layer]

        return nn.Sequential(*stages)



class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        # self.primary_conv = nn.Sequential(
        #     nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
        #     nn.BatchNorm2d(init_channels),
        # )
        #
        # self.cheap_operation = nn.Sequential(
        #     nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
        #     nn.BatchNorm2d(new_channels),
        # )
        self.conv = nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn_dw1 = nn.BatchNorm2d(init_channels)
        self.conv_dw = nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False)
        self.bn_dw2 = nn.BatchNorm2d(new_channels)

    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.bn_dw1(x1)
        x2 = self.conv_dw(x1)
        x2 = self.bn_dw2(x2)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class KFNet_block(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, out_chs, dw_kernel_size=3,
                 stride=1, has_se=False, squeeze_factor: int = 4, drop_rate: float = 0., if_dw=False):
        super(KFNet_block, self).__init__()
        self.stride = stride
        self.if_dw = if_dw

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, out_chs)


        # Depth-wise convolution
        if self.if_dw:
            self.conv_dw = nn.Conv2d(out_chs, out_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2,
                                     groups=out_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(out_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(out_chs)
        else:
            self.se = nn.GELU()

        # Point-wise linear projection
        self.ghost2 = GhostModule(out_chs, out_chs)

        # shortcut
        if (in_chs == out_chs):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.if_dw:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x = self.drop_path(x)
        x += self.shortcut(residual)

        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf

    This function is taken from the rwightman.
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = nn.GELU()
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

if __name__ == '__main__':
    model = KFNet()
    model.eval()
    # print(model)
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, (input,))
    print('flops: ', flops/1e8, 'params: ', params)
    y = model(input)
    print(y.size())
