import torch
import torch.nn as nn


class ResidualEncoder(nn.Module):
    def __init__(self):
        super(ResidualEncoder, self).__init__()
        
        self.res_blocks = self.get_blocks(num_blocks = 2)


    def forward(self, x):
        x = self.res_blocks(x)
        return x
        
    def get_blocks(self,num_blocks):
        # hidden_in = [16, 32, 32, 64, 64, 128, 128, 256]
        # hidden_out =    [32, 32, 64, 64, 128, 128, 256, 256]
        hidden_in = [256] * num_blocks
        hidden_out = [256] * num_blocks
            
        block = []
        for i, hi, ho in zip(range(num_blocks), hidden_in, hidden_out):
            block.append(ResBlock(in_channel = hi, out_channel = ho))
        blocks = nn.Sequential(*block)
        return blocks
    
    
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size = 3, stride = 1, downsample = True):
        super(ResBlock, self).__init__()
        padding = int(kernel_size/2)
        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channel, out_channel, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.downsample = nn.Sequential(nn.Conv1d(in_channel, out_channel, 1, 1),
                                        nn.BatchNorm1d(out_channel))                               
            
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.conv1.in_channels != self.conv1.out_channels:
            identity = self.downsample(identity)
        
        output = self.relu(x+identity)
        
        return output


class ResidualDecoder(nn.Module):
    def __init__(self, hidden_channel = 256, num_blocks = 4):
        super(ResidualDecoder, self).__init__()
        self.conv_start = nn.Sequential(
                                    nn.Conv1d(hidden_channel, hidden_channel, 1),
                                    nn.BatchNorm1d(hidden_channel),
                                    nn.ReLU()
                                    )
        self.dilated_res_blocks = self.get_res_blocks(num_blocks, hidden_channel)
        # self.res_blocks = self.get_blocks(num_blocks = 8)

    def forward(self, x):
        x = self.conv_start(x)
        x = self.dilated_res_blocks(x)
        # x = self.res_blocks(x)
        return x

    def get_res_blocks(self, n, hidden):
        blocks = []
        for i in range(n):
            dilation = 2 ** (i + 1)
            blocks.append(ResBlockDilated(hidden = hidden, dil = dilation))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks
    
    
class ResBlockDilated(nn.Module):
    def __init__(self, hidden, dil):
        super(ResBlockDilated, self).__init__()
        self.res = nn.Sequential(
                        nn.Conv1d(hidden, hidden, kernel_size=3, stride=1, padding=dil,
                            dilation = dil),
                        nn.BatchNorm1d(hidden),
                        nn.ReLU(),
                        nn.Conv1d(hidden, hidden, kernel_size=3, stride=1, padding=dil,
                            dilation = dil),
                        nn.BatchNorm1d(hidden)
                        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x 
        out = self.res(x)
        out = self.relu(out + identity)
        return out
        