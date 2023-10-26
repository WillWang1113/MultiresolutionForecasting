import logging

import torch
from torch import nn

logger = logging.getLogger()


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


class CNNEncoder(nn.Module):
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

        self.linear_out = nn.Linear(hidden_units * 2 * timesteps, latent_dim)


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
        out = self.pool(out)
        out = nn.Flatten()(out)
        out = self.linear_out(out)
        return out



class DNNEncoder(nn.Module):
    def __init__(self,
                 dimension_in,
                 latent_dim,
                 hidden_units,
                 encode_obs_time=False,
                 **kwargs):
        super(DNNEncoder, self).__init__()
        self.encode_obs_time = encode_obs_time
        if self.encode_obs_time:
            dimension_in += 1
        timesteps = kwargs.get("timesteps")
        self.blk = nn.Sequential(
            nn.Flatten(), nn.Linear(dimension_in * timesteps, hidden_units),
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


class GRUEncoder(nn.Module):
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
            batch_first=True)
        self.linear_out = nn.Linear(hidden_units, latent_dim)

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
        reversed_trajs_to_encode = trajs_to_encode
        out, _ = self.gru(reversed_trajs_to_encode)
        return self.linear_out(out[:, -1, :])



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
        # it can also be other NN
        self.fcst_encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(fcst_dim_in * fcst_tsteps, hidden_units),
            nn.ReLU(), nn.Linear(
                hidden_units, latent_dim)) if fcst_dim_in is not None else None

        # Concat togather and linear out
        self.linear_out = nn.Linear(
            latent_dim * 2, latent_dim) if fcst_dim_in is not None else None

    def forward(self, observed_data, available_forecasts, observed_tp):
        hist_latent = self.hist_encoder(observed_data, observed_tp)
        if self.fcst_encoder is not None:
    
            fcst_latent = self.fcst_encoder(available_forecasts)
            out = self.linear_out(
                torch.concat((hist_latent, fcst_latent), axis=-1))
        else:
            out = hist_latent
        return out

