import torch
import torch.nn as nn
from seld.utils.model.blocks import DoubleConv2D, init_layer

class EINV2(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
    
        # if "GCCPHAT" in cfg['features']['type']:
        #     in_channels = 10
        # else:
        #     in_channels = 7

        out_channels_list = [64, 128, 256, 512]

        # Convolution blocks
        self.sed_conv1 = nn.Sequential(
            DoubleConv2D(in_channels=in_channels, out_channels=out_channels_list[0]),
            nn.AvgPool2d(kernel_size=2),
        )
        self.sed_conv2 = nn.Sequential(
            DoubleConv2D(in_channels=out_channels_list[0], out_channels=out_channels_list[1]),
            nn.AvgPool2d(kernel_size=2),
        )
        self.sed_conv3 = nn.Sequential(
            DoubleConv2D(in_channels=out_channels_list[1], out_channels=out_channels_list[2]),
            nn.AvgPool2d(kernel_size=2),
        )
        self.sed_conv4 = nn.Sequential(
            DoubleConv2D(in_channels=out_channels_list[2], out_channels=out_channels_list[3]),
            nn.AvgPool2d(kernel_size=2),
        )

        self.doa_conv1 = nn.Sequential(
            DoubleConv2D(in_channels=in_channels, out_channels=out_channels_list[0]),
            nn.AvgPool2d(kernel_size=2),
        )
        self.doa_conv2 = nn.Sequential(
            DoubleConv2D(in_channels=out_channels_list[0], out_channels=out_channels_list[1]),
            nn.AvgPool2d(kernel_size=2),
        )
        self.doa_conv3 = nn.Sequential(
            DoubleConv2D(in_channels=out_channels_list[1], out_channels=out_channels_list[2]),
            nn.AvgPool2d(kernel_size=2),
        )
        self.doa_conv4 = nn.Sequential(
            DoubleConv2D(in_channels=out_channels_list[2], out_channels=out_channels_list[3]),
            nn.AvgPool2d(kernel_size=2),
        )

        # Parameter sharing blocks
        self.cross_stitch = nn.ParameterList([
            nn.Parameter(torch.Tensor(out_channels_list[0], 2, 2).uniform_(0.1, 0.9)),
            nn.Parameter(torch.Tensor(out_channels_list[1], 2, 2).uniform_(0.1, 0.9)),
            nn.Parameter(torch.Tensor(out_channels_list[2], 2, 2).uniform_(0.1, 0.9)),
        ])

        # MHSA blocks
        self.sed_mhsa1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=out_channels_list[3], nhead=8, dim_feedforward=1024, dropout=0.2, batch_first=True), num_layers=2)
        self.sed_mhsa2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=out_channels_list[3], nhead=8, dim_feedforward=1024, dropout=0.2, batch_first=True), num_layers=2)
        self.doa_mhsa1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=out_channels_list[3], nhead=8, dim_feedforward=1024, dropout=0.2, batch_first=True), num_layers=2)
        self.doa_mhsa2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=out_channels_list[3], nhead=8, dim_feedforward=1024, dropout=0.2, batch_first=True), num_layers=2)

        # FC blocks
        self.sed_fc1 = nn.Linear(in_features=out_channels_list[3], out_features=3)
        self.sed_fc2 = nn.Linear(in_features=out_channels_list[3], out_features=14)
        self.doa_fc1 = nn.Linear(in_features=out_channels_list[3], out_features=3)
        self.doa_fc2 = nn.Linear(in_features=out_channels_list[3], out_features=14)

        # Binarize
        self.sed_bin = nn.Sigmoid()
        self.doa_bin = nn.Tanh()

        # init layers
        init_layer(self.sed_fc1)
        init_layer(self.sed_fc2)
        init_layer(self.doa_fc1)
        init_layer(self.doa_fc2)

    def forward(self, x):
        x_sed = x[:,:4,:,:]
        x_doa = x

        # Encoder
        x_sed = self.sed_conv1(x_sed)
        x_doa = self.doa_conv1(x_doa)
        x_sed = torch.einsum('c,ncft -> ncft', self.cross_stitch[0][:, 0, 0], x_sed) + \
                torch.einsum('c,ncft -> ncft', self.cross_stitch[0][:, 0, 1], x_doa) 
        x_doa = torch.einsum('c,ncft -> ncft', self.cross_stitch[0][:, 1, 0], x_doa) + \
                torch.einsum('c,ncft -> ncft', self.cross_stitch[0][:, 1, 1], x_sed) 

        x_sed = self.sed_conv2(x_sed)
        x_doa = self.doa_conv2(x_doa)
        x_sed = torch.einsum('c,ncft -> ncft', self.cross_stitch[1][:, 0, 0], x_sed) + \
                torch.einsum('c,ncft -> ncft', self.cross_stitch[1][:, 0, 1], x_doa) 
        x_doa = torch.einsum('c,ncft -> ncft', self.cross_stitch[1][:, 1, 0], x_doa) + \
                torch.einsum('c,ncft -> ncft', self.cross_stitch[1][:, 1, 1], x_sed) 

        x_sed = self.sed_conv3(x_sed)
        x_doa = self.doa_conv3(x_doa)
        x_sed = torch.einsum('c,ncft -> ncft', self.cross_stitch[2][:, 0, 0], x_sed) + \
                torch.einsum('c,ncft -> ncft', self.cross_stitch[2][:, 0, 1], x_doa) 
        x_doa = torch.einsum('c,ncft -> ncft', self.cross_stitch[2][:, 1, 0], x_doa) + \
                torch.einsum('c,ncft -> ncft', self.cross_stitch[2][:, 1, 1], x_sed) 

        x_sed = self.sed_conv4(x_sed)
        x_doa = self.doa_conv4(x_doa)

        x_sed = x_sed.mean(dim=2)
        x_doa = x_doa.mean(dim=2)

        # MHSA
        x_sed_track1 = self.sed_mhsa1(x_sed.transpose(1,2))
        x_sed_track2 = self.sed_mhsa2(x_sed.transpose(1,2))
        
        x_doa_track1 = self.doa_mhsa1(x_doa.transpose(1,2))
        x_doa_track2 = self.doa_mhsa2(x_doa.transpose(1,2))

        # FC
        x_sed_track1 = self.sed_bin(self.sed_fc1(x_sed_track1))
        x_sed_track2 = self.sed_bin(self.sed_fc2(x_sed_track2))

        x_doa_track1 = self.doa_bin(self.doa_fc1(x_doa_track1))
        x_doa_track2 = self.doa_bin(self.doa_fc2(x_doa_track2))

        x_sed = torch.stack((x_sed_track1, x_sed_track2), dim=2)
        x_doa = torch.stack((x_doa_track1, x_doa_track2), dim=2)
        return {
            "sed": x_sed,
            "doa": x_doa
        }










