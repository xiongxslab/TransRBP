import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_factor):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=6, dilation=dilation_factor, padding='same')
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out += residual
        return out

class rbpmodel(nn.Module):
    def __init__(self):
        super(rbpmodel, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=4, out_channels=128, kernel_size=12, padding='same')
        self.res_blocks = nn.ModuleList([ResidualBlock(128, 128, 2**i) for i in range(9)])
        self.conv_transpose = nn.Conv1d(in_channels=128, out_channels=1, kernel_size=20,padding='same')

    def forward(self, x):
        x = x.float()
        x = self.conv1d(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.conv_transpose(x)
        x = x.permute(0, 2, 1)
        return x
