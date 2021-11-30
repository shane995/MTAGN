import torch.nn as nn
from MTAGN_utils.attention_block import Flatten


def conv_block(channel):
    """
        w_out = [w_in - k_size + 2*padding ]/stride + 1
        :param channel:
        :return:
        """
    return nn.Sequential(
        nn.Conv1d(channel[0], channel[1], kernel_size=3, stride=1, padding=1),
        nn.BatchNorm1d(channel[1]),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2)
    )


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = conv_block([1, 8])
        self.conv2 = conv_block([8, 16])
        self.conv3 = conv_block([16, 16])
        self.conv4 = conv_block([16, 16])
        self.conv5 = conv_block([16, 16])
        self.conv6 = conv_block([16, 16])

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.conv5(y)
        y = self.conv6(y)
        return y  # [b, 32, 16]


class JLCNN(nn.Module):
    def __init__(self):
        super(JLCNN, self).__init__()
        self.class1 = 4
        self.class2 = 4
        self.encoder = Encoder()
        self.FTI_fc = nn.Sequential(
            Flatten(),
            nn.Linear(32 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, self.class1)
        )
        self.FSI_fc = nn.Sequential(
            Flatten(),
            nn.Linear(32 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, self.class2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x1 = self.FTI_fc(x)
        x2 = self.FSI_fc(x)
        return x1, x2


if __name__ == '__main__':
    model = JLCNN()
    print(model)
