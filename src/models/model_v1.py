from torch import nn
import torch.nn.functional as F


class ModelV1(nn.Module):
    def __init__(self):
        super(ModelV1, self).__init__()

        self.DROPOUT_VALUE = 0.10

        # Block 01 | Entry
        self.block01 = nn.Sequential(
            # Layer 01
            nn.Conv2d(3, 32, 1, padding=0),
            nn.ReLU(),
            # Layer 02
            nn.Conv2d(32, 32, 3, padding=0),
            nn.ReLU(),
            # Layer 03
            nn.Conv2d(32, 64, 1, padding=0),
            nn.ReLU(),
        )
        # Block 01 | Depthwise Separable
        self.block02 = nn.Sequential(
            # Layer 01
            nn.Conv2d(64, 256, 1, padding=0),
            nn.ReLU(),
            # Layer 02 | Depthwise
            nn.Conv2d(256, 256, 3, padding=1, groups=256),
            nn.ReLU(),
            # Layer 03 | Pointwise
            nn.Conv2d(256, 128, 1, padding=0),
            nn.ReLU(),
        )
        # Block 03
        self.block03 = nn.Sequential(
            # Layer 01
            nn.Conv2d(128, 64, 1, padding=0),
            nn.ReLU(),
            # Layer 02 | Dilated Convolution
            nn.Conv2d(64, 64, 3, padding=0, dilation=2),
            nn.ReLU(),
            # Layer 03
            nn.Conv2d(64, 128, 1, padding=0),
            nn.ReLU(),
        )
        # Block 04
        self.block04 = nn.Sequential(
            # Layer 01
            nn.Conv2d(128, 64, 1, padding=0),
            nn.ReLU(),
            # Layer 02
            nn.Conv2d(64, 64, 3, padding=0),
            nn.ReLU(),
            # Layer 03
            nn.Conv2d(64, 128, 1, padding=0),
            nn.ReLU(),
        )
        # Block 05
        self.block05 = nn.Sequential(nn.AvgPool2d(1))

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
