import torch
import torch.nn as nn

from model import TransformerBlocks
from model import resblocks


class RBPMultiTaskModel(nn.Module):
    """
    Multi-task pretraining model with m6A as a 5th input channel.

    Backbone is identical to RBPModel in RBPResTransModels.py (features=5, variant=False).
    The single end_conv (256->1) is replaced with a 2-layer position-wise MLP head
    that outputs num_tasks tracks simultaneously.

    Input:  (batch, features, 800)   default features=5: 4-ch RNA one-hot + 1-ch m6A
    Output: (batch, num_tasks, 800)  default num_tasks=32

    Fine-tuning recipe for a single RBP:
        1. Load pretrained weights.
        2. Replace task_head with nn.Sequential(
               nn.Conv1d(128, 1, kernel_size=1), nn.BatchNorm1d(1), nn.ReLU())
        3. Freeze everything except the new head (optionally unfreeze decoder_res too).
        4. Train with lr ~1e-4 using the single-RBP RBPDataset from dataset_windowneg.py.

    Layer freeze guide:
        Freeze  : pre_conv, encoder_res, attn, decoder_res, task_head[0:3]  (256->128 layer)
        Replace : task_head[3:]  (Conv1d 128->32 becomes Conv1d 128->1)
    """

    def __init__(self, num_tasks=32, features=5):
        super(RBPMultiTaskModel, self).__init__()

        self.num_tasks = num_tasks
        in_channel = 256

        # --- Backbone (identical to RBPModel features=5, variant=False) ---
        self.pre_conv = nn.Sequential(
            nn.Conv1d(features, in_channel, kernel_size=5, padding=2),
            nn.BatchNorm1d(in_channel),
        )
        self.encoder_res = resblocks.ResidualEncoder()
        self.attn        = TransformerBlocks.AttnModule(hidden=in_channel)
        self.decoder_res = resblocks.ResidualDecoder()

        # --- 2-layer position-wise MLP head ---
        # Conv1d(k=1) == linear transform applied independently at each of the 800 positions
        self.task_head = nn.Sequential(
            nn.Conv1d(in_channel, 128, kernel_size=1),   # [0] compress: 256 -> 128
            nn.BatchNorm1d(128),                          # [1]
            nn.ReLU(),                                    # [2]
            nn.Conv1d(128, num_tasks, kernel_size=1),    # [3] task outputs: 128 -> 32
            nn.BatchNorm1d(num_tasks),                    # [4]
            nn.ReLU(),                                    # [5]
        )

    def forward(self, x):
        x = x.float()

        # Encode
        x = self.pre_conv(x)                            # (B, 256, L)
        x = torch.exp(x)
        x = self.encoder_res(x)                         # (B, 256, L)

        # Attention
        x = x.permute(0, 2, 1).contiguous()            # (B, L, 256)
        x, _ = self.attn(x)                             # discard attn weights
        x = x.permute(0, 2, 1).contiguous()            # (B, 256, L)

        # Decode
        x = self.decoder_res(x)                         # (B, 256, L)

        # Multi-task head
        return self.task_head(x)                        # (B, num_tasks, L)


if __name__ == '__main__':
    model = RBPMultiTaskModel(num_tasks=32, features=5)
    dummy = torch.randn(2, 5, 800)
    out   = model(dummy)
    print('output shape:', out.shape)   # expect (2, 32, 800)
    assert out.shape == (2, 32, 800), f'Unexpected shape: {out.shape}'
    print('shape check passed.')
