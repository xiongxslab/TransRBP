import torch
import torch.nn as nn

import sys
sys.path.append("TransRBP/model")
import TransformerBlocks
import resblocks


class RBPModel(nn.Module):
    def __init__(self, features = 4, record_attn = False):
        super(RBPModel, self).__init__()
        
        print('\n')
        print('------------------')
        print('Initializing Model')
        print('------------------')
        print('\n')
        
        self.features = features
        self.record_attn = record_attn
        in_channel = 256
        out_channel = 1
        self.pre_conv = nn.Sequential(nn.Conv1d(self.features, 256, kernel_size=5, padding=2),
                              nn.BatchNorm1d(256),)
        self.encoder_res = resblocks.ResidualEncoder()
        self.attn = TransformerBlocks.AttnModule(hidden = in_channel)
        self.decoder_res = resblocks.ResidualDecoder()
        self.end_conv = nn.Sequential(nn.Conv1d(in_channel, out_channel, kernel_size=1),
                                     nn.BatchNorm1d(out_channel),
                                     nn.ReLU())
        # self.end_pool = nn.AdaptiveAvgPool1d(1)
        # self.dense = nn.Linear(256,1)
        # self.act = nn.ReLU()
    
    
    def forward(self, x):
        x = x.float()
        x = self.pre_conv(x)
        x = torch.exp(x)
        x = self.encoder_res(x)
            
        x = x.permute(0, 2, 1).contiguous()
        
        x = self.attn(x)
        
        x = x.permute(0, 2, 1).contiguous()
        x = self.decoder_res(x)

        output = self.end_conv(x)

        return output