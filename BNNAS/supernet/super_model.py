import torch.nn as nn
import math
from torch_blocks import *
import copy
import pdb 

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class Select_one_OP(nn.Module):
  def __init__(self, inp, oup, stride):
    super(Select_one_OP, self).__init__()
    self._ops = nn.ModuleList()
    self.input_channel = inp
    self.output_channel = oup
    self.stride = stride
    for idx, key in enumerate(config.blocks_keys):
      op = blocks_dict[key](inp, oup, stride)
      op.idx = idx
      self._ops.append(op)

  def forward(self, x, id):
    if id < 0:  #Identity
        return x
    return self._ops[id](x)


class SuperNetwork(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(SuperNetwork, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [  #for GPU search
            # t, c, n, s
            [6, 32,  4, 2],
            [6, 56,  4, 2],
            [6, 112, 4, 2],
            [6, 128, 4, 1],
            [6, 256, 4, 2],
            [6, 432, 1, 1],
        ]
        # building first layer
        input_channel = int(40 * width_mult)
        self.last_channel = int(1728 * width_mult) if width_mult > 1.0 else 1728
        self.conv_bn = conv_bn(3, input_channel, 2)
        self.MBConv_ratio_1 = InvertedResidual(input_channel, int(24*width_mult), 3, 1, 1, 1)
        input_channel = int(24*width_mult)
        self.features = nn.ModuleList()
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            t = None
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(Select_one_OP(input_channel, output_channel, s))
                else:
                    self.features.append(Select_one_OP(input_channel, output_channel, 1))
                input_channel = output_channel
        self.unshare_weights = nn.ModuleList()
        # building last several layers
        self.conv_1x1_bn = conv_1x1_bn(input_channel, self.last_channel)
        self.avgpool = nn.AvgPool2d(input_size//32)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.last_channel, n_class),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                m.affine = True

    def forward(self, x, rngs):
        x = self.conv_bn(x)
        x = self.MBConv_ratio_1(x)
        for i, select_op in enumerate(self.features):
            x = select_op(x, rngs[i])
        x = self.conv_1x1_bn(x)
        x = self.avgpool(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

    def architecture(self):
        arch = []
        for feat in self.features:
            if feat.stride == 2:
                arch.append('reduce')
            else:
                arch.append('normal')
        return arch