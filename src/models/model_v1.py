from torch import nn
import torch.nn.functional as F


class ModelV1(nn.Module):
    def __init__(self):
        super(ModelV1, self).__init__()

        self.name = "ModelV1"
        # Layer 01
        self.layer01 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU())
        # Layer 02
        self.layer02 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
        )
        # Layer 03
        self.layer03 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.25),
        )
        # Layer 04
        self.layer04 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
        )
        # Layer 05
        self.layer05 = nn.Sequential(
            nn.Conv2d(32, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )
        # Layer 06
        self.layer06 = nn.Sequential(
            nn.Conv2d(16, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )
        # Layer 07
        self.layer07 = nn.Sequential(nn.Conv2d(16, 10, 3), nn.ReLU())
        # Layer 08
        self.layer08 = nn.Sequential(nn.AvgPool2d(1))

    def forward(self, x):
        x = self.layer01(x)
        x = self.layer02(x)
        x = self.layer03(x)
        x = self.layer04(x)
        x = self.layer05(x)
        x = self.layer06(x)
        x = self.layer07(x)
        x = self.layer08(x)

        x = x.view(-1, 10)
        return F.log_softmax(x)
