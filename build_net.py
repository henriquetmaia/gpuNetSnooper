import torch
import torch.nn as nn

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

## Modified Torch's base ResNet model to support arbitrary networks
class genNet(nn.Module):

    def __init__(self, layerSpecs, inputDims):
        super(genNet, self).__init__()

        self.addOuput = [layerSpecs[i+1] for i, x in enumerate(layerSpecs) if x == 'add']
        self.specs = layerSpecs
        self.inputDims = inputDims
        self.origDims = inputDims
        self.layers = nn.ModuleList()
        self.deadBranch = nn.Sequential()
        self._create_net2()

    def _create_net(self):

        imgSize = 32 # for print/debugging image size only
        ldx = 0
        sdx = 0
        branchDims = self.inputDims
        while sdx < len(self.specs):

            layer = self.specs[sdx]
            if ldx in self.addOuput:
                branchDims = self.inputDims

            if layer == 'add':
                self.layers.append( nn.Sequential() )
                sdx += 2
            elif layer == 'C2':
                filters = int(self.specs[sdx + 1])
                kernel = int(self.specs[sdx + 2])
                stride = int(self.specs[sdx + 3])
                # pad = int(self.specs[sdx + 4])
                pad = int((kernel-1)/2) # C2S
                dilation = 1

                if sdx > 0 and self.specs[sdx - 1] == 'b' and self.specs[sdx + 6] == 'add':
                    self.layers.append( nn.Conv2d(branchDims, filters, kernel_size=kernel, padding=pad, dilation=1, stride=stride, bias=False))
                    self.layers.append( nn.BatchNorm2d(filters) )
                else:
                    self.layers.append( nn.Conv2d(self.inputDims, filters, kernel_size=kernel, padding=pad, dilation=1, stride=stride, bias=False))
                    self.layers.append( nn.BatchNorm2d(filters) )
                    imgSize = int((imgSize + 2 * pad - dilation * (kernel - 1) - 1)/stride + 1)
                self.inputDims = filters
                sdx += 5

            elif layer == 'N':
                self.layers.append(Flatten())
                filters = int(self.specs[sdx + 1])
                self.layers.append( nn.Linear( self.inputDims, filters ))
                self.inputDims = filters
                sdx += 2

                imgSize = 1
            elif layer == 'F':
                self.layers.append(Flatten())
                sdx += 1

            elif layer == 'P2':
                if self.specs[sdx + 1] == 'Pa':
                    # self.layers.append( nn.AvgPool2d(4))
                    self.layers.append( nn.AvgPool2d(2))
                    # self.layers.append( nn.AdaptiveAvgPool2d((1, 1)))
                else:
                    self.layers.append( nn.MaxPool2d( kernel_size=2))

                imgSize = imgSize / 2                    
                sdx += 2
            elif layer == 'PS':
                self.layers.append( nn.AdaptiveAvgPool2d((1, 1)))
                imgSize = 1
                sdx += 2
            elif layer == 'b':
                self.layers.append( nn.BatchNorm2d(self.inputDims) )
                sdx += 1
            elif layer == 'Ar':
                self.layers.append( nn.ReLU(inplace=False) )
                sdx += 1
            elif layer == 'At':
                self.layers.append( nn.Tanh() )
                sdx += 1                
            elif layer == 'As':
                self.layers.append( nn.Sigmoid() )
                sdx += 1                
            elif layer == 'Al':
                self.layers.append( nn.LeakyReLU() )
                sdx += 1                
            elif layer == 'Asm':
                self.layers.append( nn.Softmax )
                sdx += 1                
            else:
                print('unknown layer, exiting', layer)
                return
            ldx += 1

        return 


    def _create_net2(self):

        # self.deadBranch = nn.ConvTranspose2d(self.inputDims, 64, kernel_size=7, padding=3, dilation=1, stride=2, bias=False)
        # self.deadBranch = nn.ConvTranspose2d(3, 4, 7, stride=2)
        self.deadBranch = nn.Conv2d(self.inputDims, 8, kernel_size=3, bias=True)

        imgSize = 224
        ldx = 0
        sdx = 0
        branchDims = self.inputDims
        while sdx < len(self.specs):

            layer = self.specs[sdx]
            if ldx in self.addOuput:
            # if sdx in self.addOuput:
                branchDims = self.inputDims

            if layer == 'add':
                self.layers.append( nn.Sequential() )
                sdx += 2
            elif layer == 'C2':
                filters = int(self.specs[sdx + 1])
                kernel = int(self.specs[sdx + 2])
                stride = int(self.specs[sdx + 3])
                # pad = int(self.specs[sdx + 4])
                pad = int((kernel-1)/2) # C2S                
                dilation = 1

                if sdx > 0 and self.specs[sdx - 1] == 'b' and self.specs[sdx + 6] == 'add':
                    self.layers.append( nn.Conv2d(branchDims, filters, kernel_size=kernel, padding=pad, dilation=1, stride=stride, bias=False))
                    self.layers.append( nn.BatchNorm2d(filters) )
                else:
                    self.layers.append( nn.Conv2d(self.inputDims, filters, kernel_size=kernel, padding=pad, dilation=1, stride=stride, bias=False))
                    # self.layers.append( nn.Conv2d(self.inputDims, filters, kernel_size=kernel, padding=pad, dilation=1, stride=stride, bias=True))
                    self.layers.append( nn.BatchNorm2d(filters) )
                    imgSize = int((imgSize + 2 * pad - dilation * (kernel - 1) - 1)/stride + 1)
                self.inputDims = filters
                sdx += 5
                ldx += 1
            elif layer == 'N':
                # self.layers.append(Flatten())
                filters = int(self.specs[sdx + 1])
                self.layers.append( nn.Linear( self.inputDims, filters ))
                self.inputDims = filters
                sdx += 2

                imgSize = 1
            elif layer == 'F':
                self.layers.append(Flatten())
                sdx += 1

            elif layer == 'P2':
                if self.specs[sdx + 1] == 'Pa':
                    # self.layers.append( nn.AvgPool2d(4))
                    self.layers.append( nn.AdaptiveAvgPool2d((1, 1)))
                else:
                    self.layers.append( nn.MaxPool2d( kernel_size=3, stride=2, padding=1 ))
                    # self.layers.append( nn.MaxPool2d( kernel_size=2))

                imgSize = imgSize / 2                    
                sdx += 2
            elif layer == 'PS':
                self.layers.append( nn.AdaptiveAvgPool2d((1, 1)))
                imgSize = 1
                sdx += 2
            elif layer == 'b':
                # self.layers.append( nn.BatchNorm2d(self.inputDims) )
                sdx += 1
            elif layer == 'Ar':
                self.layers.append( nn.ReLU(inplace=False) )
                sdx += 1
            elif layer == 'At':
                self.layers.append( nn.Tanh() )
                sdx += 1                
            elif layer == 'As':
                self.layers.append( nn.Sigmoid() )
                sdx += 1                
            elif layer == 'Al':
                self.layers.append( nn.LeakyReLU() )
                sdx += 1                
            elif layer == 'Asm':
                self.layers.append( nn.Softmax )
                sdx += 1                
            else:
                print('unknown layer, exiting', layer)
                return
            ldx += 1

        return 


    def forward(self, x):
        ldx = 0
        sdx = 0
        branchX = x
        w = x
        while sdx < len(self.specs):
            layer = self.specs[sdx]

            if ldx in self.addOuput:

                branchX = x
            if layer == 'add':
                x = x + branchX
                sdx += 2
                # continue
            elif layer == 'C2':
                stride = int(self.specs[sdx + 3])
                if sdx > 0 and self.specs[sdx - 1] == 'b' and self.specs[sdx + 6] == 'add':
                    branchX = self.layers[ldx](branchX)
                    branchX = self.layers[ldx+1](branchX)
                    ldx += 2
                    sdx += 6
                    continue
                sdx += 4
            elif layer == 'F':
                sdx += 1
            elif layer == 'N':
                # x = torch.flatten(x, 1)
                sdx += 2
            elif layer == 'P2':
                sdx += 2
            elif layer == 'PS':
                sdx += 2
            elif layer == 'b':
                sdx += 1
            elif layer == 'Ar':
                sdx += 1
            elif layer == 'At':
                sdx += 1
            elif layer == 'As':
                sdx += 1
            elif layer == 'Al':
                sdx += 1
            elif layer == 'Asm':
                sdx += 1
            else:
                print('unknown layer, exiting', layer)

            x = self.layers[ldx](x)
            ldx += 1

            if layer == 'b':
                z = nn.Conv2d(self.origDims, 64, kernel_size=7, padding=3, dilation=1, stride=2, bias=True)(w)

        return x


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, act=nn.ReLU(inplace=True)):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = act
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, act=nn.ReLU(inplace=True)):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = act
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, actStr, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        if actStr == 'Ar':
            activation = nn.ReLU(inplace=False)
        elif actStr == 'At':
            activation = nn.Tanh()
        elif actStr == 'As':
            activation = nn.Sigmoid()
        elif actStr == 'Ae':
            activation = nn.ELU()
        else:
            activation = nn.LeakyReLU(inplace=False)
        self.relu = activation
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, act=self.relu))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, act, pretrained, progress, **kwargs):
    model = ResNet(block, layers, act, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18( act='Ar', pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], act, pretrained, progress,
                   **kwargs)


def resnet34(act='Ar', pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], act, pretrained, progress,
                   **kwargs)


def resnet50(act='Ar', pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], act, pretrained, progress,
                   **kwargs)


def resnet101(act='Ar', pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], act, pretrained, progress,
                   **kwargs)


def resnet152(act='Ar', pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], act, pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)




class cBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, act=nn.ReLU()):
        super(cBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.relu = act

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class cBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, act=nn.ReLU()):
        super(cBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.relu = act

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class cResNet(nn.Module):
    def __init__(self, block, num_blocks, actStr, num_classes=10):
        super(cResNet, self).__init__()
        self.in_planes = 64
        if actStr == 'Ar':
            activation = nn.ReLU()
        elif actStr == 'At':
            activation = nn.Tanh()
        elif actStr == 'As':
            activation = nn.Sigmoid()
        elif actStr == 'Ae':
            activation = nn.ELU()
        else:
            activation = nn.LeakyReLU()
        self.relu = activation
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.relu))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def cResNet18( act='Ar' ):
    return cResNet(cBasicBlock, [2, 2, 2, 2], act)


def cResNet34( act='Ar' ):
    return cResNet(cBasicBlock, [3, 4, 6, 3], act)


def cResNet50( act='Ar' ):
    return cResNet(cBottleneck, [3, 4, 6, 3], act)


def cResNet101( act='Ar' ):
    return cResNet(cBottleneck, [3, 4, 23, 3], act)


def cResNet152( act='Ar' ):
    return cResNet(cBottleneck, [3, 8, 36, 3], act)


_cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _make_layers(cfg):
    layers = []
    in_channels = 3
    for layer_cfg in cfg:
        if layer_cfg == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels=in_channels,
                                    out_channels=layer_cfg,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=True))
            layers.append(nn.BatchNorm2d(num_features=layer_cfg))
            layers.append(nn.ReLU(inplace=True))
            in_channels = layer_cfg
    return nn.Sequential(*layers)


class _VGG(nn.Module):
    """
    VGG module for 3x32x32 input, 10 classes
    """

    def __init__(self, name):
        super(_VGG, self).__init__()
        cfg = _cfg[name]
        self.layers = _make_layers(cfg)
        flatten_features = 512
        self.fc1 = nn.Linear(flatten_features, 10)
        # self.fc2 = nn.Linear(4096, 4096)
        # self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        y = self.layers(x)
        y = y.view(y.size(0), -1)
        y = self.fc1(y)
        # y = self.fc2(y)
        # y = self.fc3(y)
        return y


def VGG11():
    return _VGG('VGG11')


def VGG13():
    return _VGG('VGG13')


def VGG16():
    return _VGG('VGG16')


def VGG19():
    return _VGG('VGG19')


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x
