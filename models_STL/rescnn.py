import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.channels = channels
        self.conv1 = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(channels)
        )

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        z = F.relu(x + y)
        return z


class ResCNN(nn.Module):
    def __init__(self, out_class):
        super(ResCNN, self).__init__()  # [b, 1, 2048]
        self.out_class = out_class
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 10, 10, 4, 3),  # [b, 10, 512]
            nn.BatchNorm1d(10),
            nn.ReLU(),
        )
        self.block1 = ResBlock(10)  # [b, 10, 512]
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.block2 = ResBlock(10)  # [b, 10, 512]

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 10, 100),
            # nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(100, out_class)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.maxpool(x)
        x = self.block2(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = ResCNN(out_class=4)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2f" % total)
    print(sum(param.numel() for param in model.parameters()))