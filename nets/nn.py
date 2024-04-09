import torch


def copy_weights(model1, model2):
    model1.eval()
    model2.eval()
    with torch.no_grad():
        m1_std = model1.state_dict().values()
        m2_std = model2.state_dict().values()
        for m1, m2 in zip(m1_std, m2_std):
            m1.copy_(m2)

    state = {'model': model1.half()}
    torch.save(state, f'./weights/resnet_152.pt')


class Conv(torch.nn.Module):
    def __init__(self, inp, oup, act=True, k=1, s=1, p=0, d=1, g=1):
        super(Conv, self).__init__()
        self.conv = torch.nn.Conv2d(inp, oup, k, s, p, d, g, bias=False)
        self.bn = torch.nn.BatchNorm2d(oup)
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act:
            x = torch.nn.ReLU()(x)
        return x


class BasicBlock(torch.nn.Module):
    exp = 1

    def __init__(self, inp, oup, s=1):
        super(BasicBlock, self).__init__()
        self.add_conv = s != 1 or inp != oup * self.exp
        self.relu = torch.nn.ReLU()
        self.conv1 = Conv(inp, oup, True, 3, s, 1)
        self.conv2 = Conv(oup, oup, False, 3, 1, 1)

        if self.add_conv:
            self.conv3 = Conv(inp, oup * self.exp, False, 1, s)

    def zero_init(self):
        return torch.nn.init.constant_(self.conv2.bn.weight, 0)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.add_conv:
            identity = self.conv3(x)
        out += identity
        return self.relu(out)


class Bottleneck(torch.nn.Module):
    exp = 4

    def __init__(self, inp, oup, s=1):
        super(Bottleneck, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.add_conv = s != 1 or inp != oup * self.exp
        self.conv1 = Conv(inp, oup, True)
        self.conv2 = Conv(oup, oup, True, 3, s, 1)
        self.conv3 = Conv(oup, oup * self.exp, False)
        if self.add_conv:
            self.conv4 = Conv(inp, oup * self.exp, False, 1, s)

    def zero_init(self):
        return torch.nn.init.constant_(self.conv3.bn.weight, 0)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.add_conv:
            identity = self.conv4(x)
        out += identity
        return self.relu(out)


class ResNet(torch.nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inp = 64
        self.conv1 = Conv(3, self.inp, True, 7, 2, 3)
        self.maxpool = torch.nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], s=2)
        self.layer3 = self._make_layer(block, 256, layers[2], s=2)
        self.layer4 = self._make_layer(block, 512, layers[3], s=2)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512 * block.exp, num_classes)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if hasattr(m, 'zero_init'):
                m.zero_init()

    def _make_layer(self, block, oup, blocks, s=1):
        layers = []
        layers.append(block(self.inp, oup, s))
        self.inp = oup * block.exp
        for i in range(1, blocks):
            layers.append(block(self.inp, oup))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.layer2(self.layer1(x))
        x = self.layer4(self.layer3(x))
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


def resnet(model_name, num_cls):
    configs = {
        "18": (BasicBlock, [2, 2, 2, 2]),
        "34": (BasicBlock, [3, 4, 6, 3]),
        "50": (Bottleneck, [3, 4, 6, 3]),
        "101": (Bottleneck, [3, 4, 23, 3]),
        "152": (Bottleneck, [3, 8, 36, 3]),
    }
    block, layers = configs[model_name]
    return ResNet(block, layers, num_cls)
