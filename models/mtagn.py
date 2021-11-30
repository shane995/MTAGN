import torch
import torch.nn as nn
import torch.nn.functional as F
from MTAGN_utils.attention_block import Flatten, ECA_Layer


def conv_layer(channel, kernel):
    conv_block = nn.Sequential(
        nn.Conv1d(in_channels=channel[0], out_channels=channel[1], kernel_size=kernel, padding=kernel // 2),
        nn.BatchNorm1d(channel[1]),
        nn.ReLU()
    )
    return conv_block


# ECA layer
class att_layer(nn.Module):
    def __init__(self, k=3):
        super(att_layer, self).__init__()
        self.att = ECA_Layer(kernel_size=k)

    def forward(self, x):
        out = self.att(x)
        return out + x


class MTAGN(nn.Module):
    def __init__(self):
        super(MTAGN, self).__init__()
        # initialise network parameters
        filters = [16, 32, 64]
        self.class_1 = 4
        self.class_2 = 4

        # define first layers of every encoder block
        self.encoder_block_1 = nn.ModuleList([conv_layer([1, 16], 7)])
        self.encoder_block_1.append(conv_layer([16, 32], 5))
        self.encoder_block_1.append(conv_layer([32, 64], 3))

        # define second layers of every encoder block
        self.encoder_block_2 = nn.ModuleList([conv_layer([16, 16], 7)])
        self.encoder_block_2.append(conv_layer([32, 32], 5))
        self.encoder_block_2.append(conv_layer([64, 64], 3))

        # define task attention layers
        self.encoder_att = nn.ModuleList([nn.ModuleList([att_layer(3)])])
        for j in range(2):
            if j < 1:
                self.encoder_att.append(nn.ModuleList([att_layer(3)]))
            for i in range(2):
                self.encoder_att[j].append(att_layer(3))

        # define task conv layers
        self.encoder_att_conv = nn.ModuleList([conv_layer([16, 32], 1)])
        for i in range(2):
            if i == 0:
                self.encoder_att_conv.append(conv_layer([filters[i + 1], filters[i + 2]], 1))
            else:
                self.encoder_att_conv.append(conv_layer([filters[i + 1], 2 * filters[i + 1]], 1))

        # define pooling function
        self.maxpool = nn.MaxPool1d(kernel_size=4, stride=4)

        # define fc layers
        self.task1_fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Flatten(),
            # nn.Dropout(0.2),
            # nn.Linear(128, 32),
            # nn.ReLU(),
            nn.Linear(128, self.class_1)
        )
        self.task2_fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Flatten(),
            # nn.Dropout(0.2),
            # nn.Linear(128, 32),
            # nn.ReLU(),
            nn.Linear(128, self.class_2)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        g_encoder, g_maxpool = ([0] * 3 for _ in range(2))
        for i in range(3):
            g_encoder[i] = [0] * 2

        # def attention list for tasks
        atten_encoder = [0, 0]
        for j in range(2):
            atten_encoder[j] = [0] * 3
        for i in range(2):
            for k in range(3):
                atten_encoder[i][k] = [0] * 3

        # define global shared network
        for i in range(3):
            if i == 0:
                g_encoder[i][0] = self.encoder_block_1[i](x)
                g_encoder[i][1] = self.encoder_block_2[i](g_encoder[i][0])
                g_maxpool[i] = self.maxpool(g_encoder[i][1])
            elif i == 1:
                g_encoder[i][0] = self.encoder_block_1[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.encoder_block_2[i](g_encoder[i][0])
                g_maxpool[i] = self.maxpool(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block_1[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.encoder_block_2[i](g_encoder[i][0])

        # define task dependent module
        for i in range(2):
            for j in range(3):
                if j == 0:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](g_encoder[j][1])
                    atten_encoder[i][j][1] = self.encoder_att_conv[j](atten_encoder[i][j][0])
                    atten_encoder[i][j][2] = F.max_pool1d(atten_encoder[i][j][1], kernel_size=4, stride=4)
                elif j == 1:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](g_encoder[j][1] + atten_encoder[i][j - 1][2])
                    atten_encoder[i][j][1] = self.encoder_att_conv[j](atten_encoder[i][j][0])
                    atten_encoder[i][j][2] = F.max_pool1d(atten_encoder[i][j][1], kernel_size=4, stride=4)
                else:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](g_encoder[j][1] + atten_encoder[i][j - 1][2])
                    atten_encoder[i][j][1] = self.encoder_att_conv[j](atten_encoder[i][j][0])
                    atten_encoder[i][j][2] = F.max_pool1d(atten_encoder[i][j][1], kernel_size=2, stride=2)

        # define task prediction layers
        t1_pred = self.task1_fc(atten_encoder[0][-1][-1])
        t2_pred = self.task2_fc(atten_encoder[1][-1][-1])
        return t1_pred, t2_pred


if __name__ == '__main__':
    t = torch.randn(32, 1, 2048)
    Net = MTAGN()
    t = Net(t)
    print(Net)
