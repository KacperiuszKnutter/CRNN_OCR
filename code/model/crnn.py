import torch
import torch.nn as nn
import torch.nn.functional as F

def activation_layer(activation: str="relu", alpha: float=0.1, inplace: bool=False):
    # activation layer wrapper, alpha - parametr dla LeakyReLU
    if activation == "relu":
        return nn.ReLU(inplace=inplace)
    if activation == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=alpha, inplace=inplace)
    raise ValueError(f"Unsupported activation: {activation}")

class ConvBlock(nn.Module):
    # 2d Convolutional block with batch normalization
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int | tuple = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,  # Bias is redundant before BatchNorm2d
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,  stride : int | tuple = 1, dropout = 0.2, activation = "leaky_relu"):
        super().__init__()
        self.convb1 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.act1 = activation_layer(activation)

        self.convb2 = ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout2d(p=dropout)

        # Match tensor dimensions on the skip connection if channels or spatial size changed
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvBlock(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

        self.act2 = activation_layer(activation)

    def forward(self, x):
        skip = self.shortcut(x) if self.shortcut is not None else x

        out = self.act1(self.convb1(x))
        out = self.convb2(out)

        # Residual addition before final activation
        out = self.act2(out + skip)
        out = self.dropout(out)
        return out

class CRNN(nn.Module):
    def __init__(self, num_chars: int, in_channels: int = 1, activation: str = "leaky_relu", dropout: float = 0.2, base: int = 16):
        #
        # Convolutional Recurrent Neural Network (CRNN) for line-level OCR with CTC loss.
        # Expected input shape: [Batch, in_channels, 64, Width] normalized to [0.0, 1.0].to
        # Output shape: [T, Batch, num_chars] where T = Width // 4.
        # base = number of channels in the first blocks (16 -> 16/32/64 channels, 32 -> 32/64/128).
        #
        super().__init__()
        c1, c2, c3 = base, 2 * base, 4 * base
        self.rb1 = ResidualBlock(in_channels, c1, stride=1, activation=activation,dropout=dropout)  # -> [B, 16, 64, W]
        self.rb2 = ResidualBlock(c1, c1, stride=2, activation=activation, dropout=dropout)  # -> [B, 16, 32, W/2]
        self.rb3 = ResidualBlock(c1, c1, stride=1, activation=activation, dropout=dropout)  # -> [B, 16, 32, W/2]

        self.rb4 = ResidualBlock(c1, c2, stride=2, activation=activation, dropout=dropout)  # -> [B, 32, 16, W/4]
        self.rb5 = ResidualBlock(c2, c2, stride=1, activation=activation, dropout=dropout)  # -> [B, 32, 16, W/4]

        # Asymmetric stride=(2, 1): halves height (16 -> 8) while preserving width (W/4)
        # so narrow characters have enough horizontal time steps for CTC alignment.
        self.rb6 = ResidualBlock(c2, c3, stride=(2, 1), activation=activation, dropout=dropout)      # -> [B, 64, 8, W/4]

        self.lstm = nn.LSTM(c3 * 8, 256, bidirectional=True, num_layers=2, dropout=dropout)
        self.fc = nn.Linear(256 * 2, num_chars)

    def forward(self, images: torch.Tensor):
        # extract visual feature maps
        x = self.rb1(images)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.rb5(x)
        x = self.rb6(x)

        # transform from [B,C,H, T] --> [T, B, C, H]
        x = x.permute(3, 0, 1, 2)
        # flatten channels and height: c3 x 8 (512 for base=16) -> [T, B, c3 * 8]
        x = x.reshape(x.size(0), x.size(1),-1)

        x, _ = self.lstm(x)

        x = self.fc(x)
        x = F.log_softmax(x, dim=2)

        return x


# if __name__ == "__main__":
#     # Quick shape sanity check before training
#     dummy_batch = torch.randn(4, 1, 64, 320)  # Batch=4, Channels=1, Height=64, Width=320
#     model = CRNN(num_chars=94)
#     output = model(dummy_batch)
#     print(f"Input shape:  {dummy_batch.shape}")   # [4, 1, 64, 320]
#     print(f"Output shape: {output.shape}")        # [80, 4, 94] -> T=80, Batch=4, Classes=94