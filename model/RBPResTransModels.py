import torch
import torch.nn as nn

from . import TransformerBlocks
from . import resblocks


class RBPModel(nn.Module):
    def __init__(self, record_attn ,features = 4, variant = False):
        super(RBPModel, self).__init__()
        
        self.features = features
        self.variant = variant
        self.record_attn = record_attn
        in_channel = 256
        out_channel = 1
        if not self.variant:
            self.pre_conv = nn.Sequential(nn.Conv1d(self.features, 256, kernel_size=5, padding=2),
                                        nn.BatchNorm1d(256),)
            self.encoder_res = resblocks.ResidualEncoder()
        else:
            self.pre_conv_p = nn.Sequential(nn.Conv1d(self.features, 128, kernel_size=5, padding=2),
                                        nn.BatchNorm1d(128),)
            self.pre_conv_m = nn.Sequential(nn.Conv1d(self.features, 128, kernel_size=5, padding=2),
                                        nn.BatchNorm1d(128),)

            self.encoder_res_p = resblocks.ResidualEncoder(variant = True)
            self.encoder_res_m = resblocks.ResidualEncoder(variant = True)
            
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
        if not self.variant:
            x = self.pre_conv(x)
            x = torch.exp(x)
            x = self.encoder_res(x)
        else:
            x_p = x[:,:4,:]
            x_m = x[:,4:,:]
            x_p = self.pre_conv_p(x_p)
            x_m = self.pre_conv_m(x_m)
            x_p = torch.exp(x_p)
            x_m = torch.exp(x_m)
            x_p = self.encoder_res_p(x_p)
            x_m = self.encoder_res_m(x_m)
            x = torch.cat((x_p,x_m),1)
            
        x = x.permute(0, 2, 1).contiguous()
        
        if self.record_attn:
            x, attn_weights = self.attn(x)
        else:
            x, _ = self.attn(x)
        
  
        x = x.permute(0, 2, 1).contiguous()
        x = self.decoder_res(x)

        output = self.end_conv(x)

        return (output, attn_weights) if self.record_attn else output
