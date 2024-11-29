import torch
import torch.nn as nn
import copy
from mamba_ssm import Mamba


class MultiStageModel(nn.Module):
    def __init__(self, num_block, com_factor, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(com_factor, int(dim // (2**s)), int(dim // (2 ** (s + 1))))) for s in range(num_block)])
        self.classify = nn.Conv1d(int(dim / (2 ** (num_block))), num_classes, 1)

    def forward(self, x):
        for s in self.stages:
            x = s(x)
        x = self.classify(x)
        return x


class SingleStageModel(nn.Module):
    def __init__(self, com_factor, dim, out_f_maps):
        super(SingleStageModel, self).__init__()
        base_channels = com_factor // 8

        # Bottleneck
        self.conv_1x1 = nn.Conv1d(dim, com_factor, 1)
        self.conv_out = nn.Conv1d(com_factor, out_f_maps, 1)

        # FCTF
        self.conv1 = nn.Conv1d(com_factor, base_channels, kernel_size=3, padding=2, dilation=2)
        self.conv2 = nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=4, dilation=4)
        self.conv3 = nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=8, dilation=8)
        self.conv_1x1_output = nn.Conv1d(com_factor + base_channels * 3, com_factor, 1)

        self.layers = nn.ModuleList([Mamba(d_model=com_factor, d_state=16, d_conv=4, expand=2) for i in range(1)])

    def forward(self, x):
        # Bottleneck
        out = self.conv_1x1(x)

        # FCTF
        x1 = self.conv1(out)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        out = torch.cat([out, x1, x2, x3], dim=1)
        out = self.conv_1x1_output(out)

        # Mamba
        out = out.permute(0, 2, 1)  # (B, C, L) -> (B, L, C)
        for layer in self.layers:
            out = layer(out)

        # Bottleneck
        out = self.conv_out(out.permute(0, 2, 1))
        return out


if __name__ == "__main__":
    from utils import parameter_count_table

    model = MultiStageModel(3, 1, 64, 1000, 1)
    print(parameter_count_table(model))
