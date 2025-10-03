from torch import nn
import torch.nn.functional as F


class ModelV1(nn.Module):
    def __init__(self):
        super(ModelV1, self).__init__()

        self.DROPOUT_VALUE = 0.10

        # Block 01 | Entry
        self.block01 = nn.Sequential(
            # Layer 01
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=True,
            ),
            nn.ReLU(),
            # Layer 02
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=True,
            ),
            nn.ReLU(),
            # Layer 03
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(self.DROPOUT_VALUE),
        )
        # Block 01 | Depthwise Separable
        self.block02 = nn.Sequential(
            # Layer 01
            nn.Conv2d(64, 128, 3, padding=0),
            nn.ReLU(),
            # Layer 02 | Depthwise
            nn.Conv2d(128, 128, 3, padding=1, groups=128),
            nn.ReLU(),
            # Layer 03 | Pointwise
            nn.Conv2d(128, 128, 1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(self.DROPOUT_VALUE),
        )
        # Block 03
        self.block03 = nn.Sequential(
            # Layer 01
            nn.Conv2d(128, 32, 3, padding=1),
            nn.ReLU(),
            # Layer 02 | Dilated Convolution
            nn.Conv2d(32, 32, 3, padding=0, dilation=2),
            nn.ReLU(),
            # Layer 03
            nn.Conv2d(32, 64, 1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(self.DROPOUT_VALUE),
        )
        # Block 04
        self.block04 = nn.Sequential(
            # Layer 01
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            # Layer 02
            nn.Conv2d(32, 10, 3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(self.DROPOUT_VALUE),
        )
        # Block 05
        self.block05 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))

    def forward(self, x):
        x = self.block01(x)
        x = self.block02(x)
        x = self.block03(x)
        x = self.block04(x)
        x = self.block05(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)

    def name(self) -> str:
        return self.__class__.__name__
