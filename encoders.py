###########################
# Neural Laplace: Learning diverse classes of differential equations in the Laplace domain
# Author: Samuel Holt
###########################
import logging

import torch
from torch import nn
from NeuralLaplace.torchlaplace import laplace_reconstruct
import NeuralLaplace.torchlaplace.inverse_laplace

logger = logging.getLogger()


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels,
                      out_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1), nn.BatchNorm1d(out_channels), nn.LeakyReLU())
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels,
                      out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1), nn.BatchNorm1d(out_channels))
        self.downsample = downsample
        self.relu = nn.LeakyReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, input_dim, output_dim):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.ReLU())
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool1d(4, stride=1)
        self.fc = nn.Linear(512, output_dim)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:

            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm1d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.mean(axis=-1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def convBNReLU(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(),
    )


# # TODO: encodes the observed and static future trajectory into latent vector
# # One is reverse and the other one is normal
class CNNEncoder(nn.Module):
    # Encodes observed trajectory into latent vector
    def __init__(self,
                 dimension_in,
                 latent_dim,
                 hidden_units,
                 encode_obs_time=False,
                 **kwargs):
        super(CNNEncoder, self).__init__()
        self.encode_obs_time = encode_obs_time
        timesteps = kwargs.get("timesteps")

        if self.encode_obs_time:
            dimension_in += 1
        self.blk1 = convBNReLU(dimension_in, hidden_units)
        timesteps = timesteps // 2
        self.blk2 = convBNReLU(hidden_units, hidden_units * 2)
        timesteps = timesteps // 2
        self.pool = nn.AvgPool1d(4, stride=2, padding=1)
        timesteps = timesteps // 2
        # self.blk3 = convBNReLU(hidden_units * 2, hidden_units * 4)
        # timesteps = timesteps // 2
        # self.pool2 = nn.AvgPool1d(4, stride=2, padding=1)
        # timesteps = timesteps // 2
        self.linear_out = nn.Linear(hidden_units * 2 * timesteps, latent_dim)

        # self.resblk = ResNet(ResidualBlock, [1,1,1,1],input_dim=dimension_in, output_dim=latent_dim)
        # self.pool = nn.AvgPool1d(4, stride=2, padding=1)
        # self.linear_out = nn.Linear(hidden_units * (timesteps // 2),
        #                             latent_dim)
        # self.resblk = ResidualBlock(dimension_in, out_channels=hidden_units)
        # self.pool = nn.AvgPool1d(4, stride=2, padding=1)
        # self.linear_out = nn.Linear(hidden_units * (timesteps // 2),
        #                             latent_dim)

    def forward(self, observed_data, observed_tp):
        trajs_to_encode = observed_data  # (batch_size, t_observed_dim, observed_dim)
        if self.encode_obs_time:
            if len(observed_tp) > 1:
                trajs_to_encode = torch.cat(
                    (observed_data, observed_tp.view(len(observed_tp), -1, 1)),
                    dim=2)
            else:
                trajs_to_encode = torch.cat(
                    (observed_data, observed_tp.view(1, -1, 1).repeat(
                        observed_data.shape[0], 1, 1)),
                    dim=2)
        out = trajs_to_encode.transpose(1, 2)
        out = self.blk1(out)
        out = self.blk2(out)
        # out = self.blk3(out)
        # out = self.resblk(out)
        out = self.pool(out)
        out = nn.Flatten()(out)
        out = self.linear_out(out)
        return out


# # TODO: encodes the observed and static future trajectory into latent vector
# # One is reverse and the other one is normal
class DNNEncoder(nn.Module):
    # Encodes observed trajectory into latent vector
    def __init__(self,
                 dimension_in,
                 latent_dim,
                 hidden_units,
                 encode_obs_time=False,
                 **kwargs):
        super(DNNEncoder, self).__init__()
        self.encode_obs_time = encode_obs_time
        print("encode_obs_time:", encode_obs_time)
        if self.encode_obs_time:
            dimension_in += 1
        timesteps = kwargs.get("timesteps")
        self.blk = nn.Sequential(
            nn.Flatten(), nn.Linear(dimension_in * timesteps, hidden_units),
            # nn.BatchNorm1d(hidden_units),
              nn.ReLU(),
            nn.Linear(hidden_units, latent_dim))

    def forward(self, observed_data, observed_tp):
        trajs_to_encode = observed_data  # (batch_size, t_observed_dim, observed_dim)
        if self.encode_obs_time:
            if len(observed_tp) > 1:
                trajs_to_encode = torch.cat(
                    (observed_data, observed_tp.view(len(observed_tp), -1, 1)),
                    dim=2)
            else:
                trajs_to_encode = torch.cat(
                    (observed_data, observed_tp.view(1, -1, 1).repeat(
                        observed_data.shape[0], 1, 1)),
                    dim=2)
        return self.blk(trajs_to_encode)


# # TODO: encodes the observed and static future trajectory into latent vector
# # One is reverse and the other one is normal
class GRUEncoder(nn.Module):
    # Encodes observed trajectory into latent vector
    def __init__(self,
                 dimension_in,
                 latent_dim,
                 hidden_units,
                 encode_obs_time=False,
                 **kwargs):
        super(GRUEncoder, self).__init__()
        self.encode_obs_time = encode_obs_time
        if self.encode_obs_time:
            dimension_in += 1
        self.gru = nn.GRU(
            dimension_in,
            hidden_units,
            num_layers=2,
            #   nonlinearity="relu",
            batch_first=True)
        self.linear_out = nn.Linear(hidden_units, latent_dim)
        # nn.init.xavier_uniform_(self.linear_out.weight)

    def forward(self, observed_data, observed_tp):
        trajs_to_encode = observed_data  # (batch_size, t_observed_dim, observed_dim)
        if self.encode_obs_time:
            if len(observed_tp) > 1:
                trajs_to_encode = torch.cat(
                    (observed_data, observed_tp.view(len(observed_tp), -1, 1)),
                    dim=2)
            else:
                trajs_to_encode = torch.cat(
                    (observed_data, observed_tp.view(1, -1, 1).repeat(
                        observed_data.shape[0], 1, 1)),
                    dim=2)
        # reversed_trajs_to_encode = torch.flip(trajs_to_encode, (1, ))
        reversed_trajs_to_encode = trajs_to_encode
        out, _ = self.gru(reversed_trajs_to_encode)
        return self.linear_out(out[:, -1, :])


# TODO: encodes the observed and static future trajectory into latent vector
# One is reverse and the other one is normal
class BiEncoder(nn.Module):
    # Encodes observed trajectory into latent vector
    def __init__(self,
                 dimension_in,
                 latent_dim,
                 hidden_units,
                 encode_obs_time=False,
                 **kwargs):
        super(BiEncoder, self).__init__()
        self.encode_obs_time = encode_obs_time
        encoder = kwargs.get("encoder")
        hist_tsteps, fcst_tsteps = kwargs.get("timesteps")
        hist_dim_in, fcst_dim_in = dimension_in
        # Encoder 1: for historical data
        if encoder == "cnn":
            self.hist_encoder = CNNEncoder(hist_dim_in,
                                           latent_dim,
                                           hidden_units // 2,
                                           encode_obs_time,
                                           timesteps=hist_tsteps)
        elif encoder == "dnn":
            self.hist_encoder = DNNEncoder(hist_dim_in,
                                           latent_dim,
                                           hidden_units // 2,
                                           encode_obs_time,
                                           timesteps=hist_tsteps)
        elif encoder == "rnn":
            self.hist_encoder = GRUEncoder(hist_dim_in, latent_dim,
                                           hidden_units // 2, encode_obs_time)
        else:
            raise ValueError("encoders only include dnn, cnn and rnn")

        # Encoder 2: for available forecasts if needed
        self.fcst_encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(fcst_dim_in * fcst_tsteps, hidden_units),
            nn.ReLU(), nn.Linear(
                hidden_units, latent_dim)) if fcst_dim_in is not None else None
        # self.fcst_encoder = nn.GRU(
        #     fcst_dim_in,
        #     hidden_units // 2,
        #     num_layers=1,
        #     batch_first=True) if fcst_dim_in is not None else None
        # self.fcst_linear = nn.Linear(
        #     hidden_units // 2, latent_dim) if fcst_dim_in is not None else None

        # Concat togather and linear out
        self.linear_out = nn.Linear(
            latent_dim * 2, latent_dim) if fcst_dim_in is not None else None
        # nn.init.xavier_uniform_(self.linear_out.weight)

    def forward(self, observed_data, available_forecasts, observed_tp):
        hist_latent = self.hist_encoder(observed_data, observed_tp)
        if self.fcst_encoder is not None:
    
            # GRU fcst encoder
            # reverse_fcst = torch.flip(available_forecasts, (1, ))
            # fcst_latent, _ = self.fcst_encoder(reverse_fcst)
            # fcst_latent = self.fcst_linear(fcst_latent[:, -1, :])
            # out = self.linear_out(
            #     torch.concat((hist_latent, fcst_latent), axis=-1))

            # DNN fcst encoder
            fcst_latent = self.fcst_encoder(available_forecasts)
            out = self.linear_out(
                torch.concat((hist_latent, fcst_latent), axis=-1))
        else:
            out = hist_latent
        return out


class Interpolate(nn.Module):

    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode)
        return x


class Transpose(nn.Module):

    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.transpose = torch.transpose
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = self.transpose(x, self.dim0, self.dim1)
        return x
