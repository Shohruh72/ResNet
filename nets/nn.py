import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, inp, oup, act, k=1, s=1, p=0, d=1, g=1):
        super().__init__()
        self.conv = nn.Conv2d(inp, oup, k, s, p, d, g, False)
        self.bn = nn.BatchNorm2d(oup)
        self.act = act

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class BasicBlock(nn.Module):
    exp = 1

    def __init__(self, inp, oup, s=1, shortcut=None):
        super().__init__()
        self.act = nn.ReLU()
        self.shortcut = shortcut

        self.conv1 = Conv(inp, oup, nn.ReLU(), 3, s, 1)
        self.conv2 = Conv(oup, oup, nn.Identity(), 3, 1, 1)

    def zero_init(self):
        torch.nn.init.zeros_(self.conv2.bn.weight)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.shortcut:
            x = self.shortcut(x)
        out += x
        return self.act(out)


class Bottleneck(nn.Module):
    exp = 4

    def __init__(self, inp, oup, s=1, shortcut=None):
        super().__init__()
        self.act = nn.ReLU()
        self.shortcut = shortcut

        self.conv1 = Conv(inp, oup, nn.ReLU())
        self.conv2 = Conv(oup, oup, nn.ReLU(), 3, s, 1)
        self.conv3 = Conv(oup, oup * self.exp, nn.Identity())

    def zero_init(self):
        torch.nn.init.zeros_(self.conv3.bn.weight)

    def forward(self, x):
        # identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.shortcut:
            x = self.shortcut(x)
        out += x
        return self.act(out)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_cls):
        super(ResNet, self).__init__()
        self.inp = 64
        self.conv1 = Conv(3, self.inp, nn.ReLU(), 7, 2, 3)
        self.max_pool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.exp, num_cls)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, 0, 'fan_out', 'relu')
            if isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
        for m in self.modules():
            if hasattr(m, 'zero_init'):
                m.zero_init()

    def _make_layer(self, block, oup, depth, s=1):
        shortcut = None
        if s != 1 or self.inp != oup * block.exp:
            shortcut = Conv(self.inp, oup * block.exp, nn.Identity(), s=s)

        layers = []
        layers.append(block(self.inp, oup, s, shortcut))
        self.inp = oup * block.exp
        for _ in range(1, depth):
            layers.append(block(self.inp, oup))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.max_pool(self.conv1(x))
        x = self.layer2(self.layer1(x))
        x = self.layer4(self.layer3(x))
        x = torch.flatten(self.avg_pool(x), 1)
        return self.fc(x)


def model_summary(nets, vis=False, out_path='../outputs/'):
    from torchinfo import summary
    from torchviz import make_dot
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, net in enumerate(nets, start=1):
        net.to(device)
        print(f"\nModel #{i} Summary:")
        summary(net, input_size=(1, 3, 224, 224))

        if vis:
            input_tensor = torch.rand(1, 3, 224, 224).to(device)
            output = net(input_tensor)
            dot = make_dot(output, params=dict(list(net.named_parameters()) + [('input', input_tensor)]))
            dot.render(f"{out_path}_{i}")
            print(f"Model #{i} computation graph saved to {out_path}{i}")


def resnet(model_name, num_cls):
    configs = {
        "18": (BasicBlock, [2, 2, 2, 2]),
        "50": (Bottleneck, [3, 4, 6, 3]),
        "101": (Bottleneck, [3, 4, 23, 3]),
        "152": (Bottleneck, [3, 8, 36, 3]),
    }
    block, layers = configs[model_name]
    return ResNet(block, layers, num_cls)
