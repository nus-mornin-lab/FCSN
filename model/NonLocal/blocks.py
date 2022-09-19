import torch
from torch import nn
from torch.nn import functional as F

class GlobalAggregationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ck, cv, query_transform, dropout=0.5):
        """
        :param in_channels: int
            input channels
        :param out_channels: int
            number of channels to output
        :param ck: int
            channels of keys
        :param cv: int
            channels of values
        :param query_transform: string in ['up', 'down', 'same']
        or torch.nn.Module
            if string utils.get_conv_transform is used.
            Parameter might be a torch module that handles tenzors
            with shape of (Batch, Channels, Depth, Height, Width)
            and outputs tensor (Batch, ck, NewDepth, NewHeight, NewWidth .
        """
        super(GlobalAggregationBlock, self).__init__()
        self.ck = ck
        self.cv = cv
        self.softmax = nn.Softmax(-1)
        self.conv_1_ck = nn.Conv2d(in_channels, ck, 1)
        self.conv_1_cv = nn.Conv2d(in_channels, cv, 1)
        if type(query_transform) is str:
            self.query_transform = get_conv_transform(in_channels, ck, query_transform)
        else:
            self.query_transform = query_transform
        self.conv_1_co = nn.Conv2d(cv, out_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: torch tensor (Batch, Channels, Depth, Height, Width)
        :return: 5d torch tensor
        """
        queries = self.query_transform(x)
        batch, cq, hq, wq = queries.shape

        queries = queries.flatten(start_dim=2, end_dim=-1)
        keys = self.conv_1_ck(x).flatten(start_dim=2, end_dim=-1)
        values = self.conv_1_cv(x).flatten(start_dim=2, end_dim=-1)

        queries = queries.transpose(2, 1)

        attention = torch.matmul(queries, keys) / (self.ck ** 0.5)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        values = values.transpose(2, 1)
        output = torch.matmul(attention, values)
        output = self.conv_1_co(output.view(batch, self.cv, hq, wq))
        return output


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.batch_norm1 = nn.BatchNorm2d(in_channels)
        self.batch_norm2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU6()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def forward(self, x):
        out = self.batch_norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        x = x + out
        return x


class DownSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSamplingBlock, self).__init__()
        self.batch_norm1 = nn.BatchNorm2d(in_channels)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6()
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1, 2)
        self.conv1 = get_conv_transform(in_channels, out_channels, 'down')
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        residual = self.residual_conv(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = x + residual
        return x


class BottomAggBlock(nn.Module):
    def __init__(self, in_channels, ck, cv, dropout=0.5):
        super(BottomAggBlock, self).__init__()
        self.agg_block = GlobalAggregationBlock(in_channels, in_channels, ck, cv, 'same', dropout=dropout)

    def forward(self, x):
        x = self.agg_block(x)
        return x


class BottomBlock(nn.Module):
    def __init__(self, in_channels):
        super(BottomBlock, self).__init__()
        self.relu = nn.ReLU6()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


class UpSamplingAggBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ck, cv, dropout=0.5):
        super(UpSamplingAggBlock, self).__init__()
        self.agg_block = GlobalAggregationBlock(in_channels, out_channels, ck, cv, 'up', dropout=dropout)
        self.residual_deconv = get_conv_transform(in_channels, out_channels, 'up')

    def forward(self, x):
        residual = self.residual_deconv(x)
        x = self.agg_block(x)
        x = x + residual
        return x


class UpSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSamplingBlock, self).__init__()
        self.residual_deconv = get_conv_transform(in_channels, out_channels, 'up')

    def forward(self, x):
        x = self.residual_deconv(x)
        return x
    
def get_conv_transform(in_channels, out_channels, mode):
    """
    :param in_channels: int
    :param out_channels: int
    :param mode: string in ['up, 'down', 'same']
        'up' - ConvTranspose3d with output 2*D, 2*W, 2*H
        'down' - Conv3d with output W/2, H/2
        'same' - Conv3d with output W, H
        where W, H - shape of the input
    :return:
    """
    if mode == 'up':
        return nn.ConvTranspose2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=2,
                                  stride=2,
                                  padding=0)
    elif mode == 'down':
        return nn.Conv2d(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=3,
                         stride=2,
                         padding=1)
    elif mode == 'same':
        return nn.Conv2d(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=1,
                         stride=1)