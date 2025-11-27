import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    A simple fully-connected classifier used primarily for loss landscape
    experiments on CIFAR-like datasets. The implementation emphasizes
    modularity and readability compared to typical compact versions.
    """

    def __init__(self, input_dim=32 * 32 * 3, hidden_layers=(512, 256), num_classes=10):
        super().__init__()

        layers = []
        in_dim = input_dim

        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h

        layers.append(nn.Linear(in_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten all dimensions except batch
        x = x.reshape(x.size(0), -1)
        return self.network(x)



class BasicBlock(nn.Module):
    """
    A standard 3x3 → 3x3 residual block used in CIFAR-type ResNets.
    The implementation is intentionally explicit and avoids compact,
    harder-to-read shortcuts.
    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut mapping to match size when needed
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + self.shortcut(identity)
        out = F.relu(out, inplace=True)
        return out



class ResNet(nn.Module):
    """
    ResNet architecture adapted for CIFAR-10/100.
    The network builds three stages with progressively increasing
    feature map sizes and decreasing spatial resolution.
    """

    def __init__(self, block, layers_per_stage, num_classes=10):
        super().__init__()

        self.initial_channels = 16

        # Initial 3x3 convolution
        self.conv1 = nn.Conv2d(
            3, self.initial_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.initial_channels)

        # Residual stages
        self.stage1 = self._make_stage(block, 16, layers_per_stage[0], stride=1)
        self.stage2 = self._make_stage(block, 32, layers_per_stage[1], stride=2)
        self.stage3 = self._make_stage(block, 64, layers_per_stage[2], stride=2)

        # Final classification head
        self.classifier = nn.Linear(64 * block.expansion, num_classes)

    def _make_stage(self, block, out_channels, num_blocks, stride):
        """
        Build a stack of residual blocks. The first block may downsample
        using the provided stride; later blocks always use stride=1.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for s in strides:
            layers.append(block(self.initial_channels, out_channels, s))
            self.initial_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial convolution
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)

        # Residual stages
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)

        # Global average pooling
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)

        return self.classifier(out)


def ResNet20(num_classes=10):
    """
    Classic CIFAR-10 ResNet-20:
        layers_per_stage = [3, 3, 3]
    """
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)
