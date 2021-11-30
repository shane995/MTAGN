import torch.nn as nn
from MTAGN_utils.attention_block import Flatten


def conv_block(channel, kernel):
    return nn.Sequential(
        nn.Conv1d(channel[0], channel[1], kernel_size=kernel, stride=1, padding=kernel // 2),
        nn.BatchNorm1d(channel[1]),
        nn.ReLU(),
    )


class TrunkNet(nn.Module):
    def __init__(self):
        super(TrunkNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, 12, 2, 5),  # [b, 16, 1024]
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 16, 12, 2, 5),  # [b, 16, 512]
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(16, 24, 9, 2, 4),  # [b, 24, 256]
            nn.BatchNorm1d(24),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(24, 24, 9, 2, 4),  # [b, 24, 128]
            nn.BatchNorm1d(24),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(24, 32, 6, 2, 2),  # [b, 32, 64]
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.conv5(y)
        return y  # [b, 32, 64]


class MT1DCNN(nn.Module):
    def __init__(self):
        super(MT1DCNN, self).__init__()
        self.class1 = 3
        self.class2 = 4
        self.trunk = TrunkNet()
        self.FTI = nn.Sequential(
            nn.Sequential(
                nn.Conv1d(32, 32, 6, 2, 2),  # [b, 32, 32]
                nn.BatchNorm1d(32),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv1d(32, 64, 3, 2, 1),  # [b, 64, 16]
                nn.BatchNorm1d(64),
                nn.ReLU(),
            ),
            nn.AdaptiveAvgPool1d(1),
            Flatten(),
            nn.Linear(64, self.class1)
        )
        self.FSI = nn.Sequential(
            nn.Sequential(
                nn.Conv1d(32, 32, 6, 2, 2),  # [b, 32, 32]
                nn.BatchNorm1d(32),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv1d(32, 64, 3, 2, 1),  # [b, 64, 16]
                nn.BatchNorm1d(64),
                nn.ReLU(),
            ),
            nn.AdaptiveAvgPool1d(1),
            Flatten(),
            nn.Linear(64, self.class2)
        )

    def forward(self, x):
        x = self.trunk(x)
        x1 = self.FTI(x)
        x2 = self.FSI(x)
        return x1, x2


if __name__ == '__main__':
    model = MT1DCNN()
    print(model)
