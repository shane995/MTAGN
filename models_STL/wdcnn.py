import torch
import torch.nn as nn


class WDCNN(nn.Module):
    def __init__(self, out_class=4):
        super(WDCNN, self).__init__()
        self.out_class = out_class
        self.conv1 = nn.Sequential(  # (1, 2048)
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=64, stride=16, padding=24),  # (16, 128)
            nn.MaxPool1d(kernel_size=2, stride=2),  # (16, 64)
            nn.ReLU(),
            nn.BatchNorm1d(16),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, 3, 1, 1),  # (32, 64)
            nn.MaxPool1d(2, 2),  # (32, 32)
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, 3, 1, 1),  # (64, 32)
            nn.MaxPool1d(2, 2),  # (64, 16)
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 64, 3, 1, 1),  # (64, 16)
            nn.MaxPool1d(2, 2),  # (64, 8)
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(64, 64, 3, 1, 0),  # (64, 6)
            nn.MaxPool1d(2, 2),  # (64, 3)
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3, 100),
            nn.Linear(100, out_class)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = WDCNN()
    # print(model)
    # t = torch.randn(32, 1, 2048)
    # t1 = model(t)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2f" % total)
    print(sum(param.numel() for param in model.parameters()))
