import torch.nn as nn

def init_layer(layer, nonlinearity='relu'):
    classname = layer.__class__.__name__
    if 'Conv' in classname or 'Linear' in classname:
        nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
    elif 'BatchNorm' in classname:
        nn.init.normal_(layer.weight, 1.0, 0.02)
        nn.init.constant_(layer.bias, 0.0)    

class DoubleConv2D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, inplace=True, bias=True,
                 ):
        super.__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=inplace),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=inplace),
        )
        for layer in self.double_conv:
            init_layer(layer)

    def forward(self, x):
        return self.double_conv(x)
    



















