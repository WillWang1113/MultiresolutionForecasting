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


def convBNReLU(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.InstanceNorm1d(out_channels),
        nn.LeakyReLU(0.2, True),
    )

# ? Should be independent with s
class SphereSurfaceModel(nn.Module):
    # C^{b+k} -> C^{bxd} - In Riemann Sphere Co ords : b dim s reconstruction terms, k is latent encoding dimension, d is output dimension
    def __init__(self, s_dim, output_dim, latent_dim, hidden_units=64):
        super(SphereSurfaceModel, self).__init__()
        self.s_dim = s_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(s_dim * 2 + latent_dim, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, (s_dim) * 2 * output_dim),
        )

        # for m in self.linear_tanh_stack.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight)

        # TODO: manual set the phi min max
        # self.phi_max = 2 * (torch.atan(torch.tensor(3.0)) - torch.pi / 4.0)
        self.phi_max = torch.pi / 2.0
        # self.phi_min = 2 * (torch.atan(torch.tensor(20.0)) - torch.pi / 4.0)
        self.phi_min = -torch.pi / 2.0
        self.phi_scale = self.phi_max - self.phi_min

        self.theta_max = torch.pi
        # self.theta_min = 2 * (torch.atan(torch.tensor(20.0)) - torch.pi / 4.0)
        self.theta_min = -torch.pi
        self.theta_scale = self.theta_max - self.theta_min
        self.nfe = 0

    def forward(self, i):
        # Take in initial conditon p and the Rieman representation
        self.nfe += 1
        out = self.linear_tanh_stack(
            i.view(-1, self.s_dim * 2 + self.latent_dim)).view(
                -1, 2 * self.output_dim, self.s_dim)
        # theta = nn.Tanh()(
        #     out[:, :self.output_dim, :]) * self.theta_scale / 2.0 + self.theta_min + self.theta_scale / 2.0
        theta = nn.Tanh()(
            out[:, :self.output_dim, :]) * torch.pi  # From - pi to + pi
        # phi = (nn.Tanh()(out[:, self.output_dim:, :]) * self.phi_scale / 2.0 +
        #        self.phi_min + self.phi_scale / 2.0)  # Form -pi / 2 to + pi / 2
        phi = (nn.Tanh()(out[:, self.output_dim:, :]) * self.phi_scale / 2.0 -
               torch.pi / 2.0 + self.phi_scale / 2.0
               )  # Form -pi / 2 to + pi / 2
        return theta, phi


class SurfaceModel(nn.Module):
    # C^{b+k} -> C^{bxd} - In Riemann Sphere Co ords : b dim s reconstruction terms, k is latent encoding dimension, d is output dimension
    def __init__(self, s_dim, output_dim, latent_dim, hidden_units=64):
        super(SurfaceModel, self).__init__()
        self.s_dim = s_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(s_dim * 2 + latent_dim, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, (s_dim) * 2 * output_dim),
        )

        # for m in self.linear_tanh_stack.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight)
        self.nfe = 0

    def forward(self, i):
        self.nfe += 1
        out = self.linear_tanh_stack(
            i.view(-1, self.s_dim * 2 + self.latent_dim)).view(
                -1, 2 * self.output_dim, self.s_dim)
        real = out[:, :self.output_dim, :]
        imag = out[:, self.output_dim:, :]
        return real, imag


# # TODO: encodes the observed and static future trajectory into latent vector
# # One is reverse and the other one is normal
class CNNEncoder(nn.Module):
    # Encodes observed trajectory into latent vector
    def __init__(self,
                 dimension_in,
                 latent_dim,
                 hidden_units,
                 timesteps,
                 encode_obs_time=False):
        super(CNNEncoder, self).__init__()
        self.encode_obs_time = encode_obs_time
        if self.encode_obs_time:
            dimension_in += 1
        self.blk1 = convBNReLU(dimension_in, hidden_units)
        timesteps = timesteps // 2
        self.blk2 = convBNReLU(hidden_units, hidden_units * 2)
        timesteps = timesteps // 2
        self.blk3 = convBNReLU(hidden_units * 2, hidden_units * 4)
        timesteps = timesteps // 2
        self.linear_out = nn.Linear(hidden_units * 4 * timesteps, latent_dim)
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
        out = trajs_to_encode.transpose(1, 2)
        out = self.blk1(out)
        out = self.blk2(out)
        out = self.blk3(out)
        out = nn.Flatten()(out)
        out = self.linear_out(out)
        return out


# # TODO: encodes the observed and static future trajectory into latent vector
# # One is reverse and the other one is normal
class ReverseGRUEncoder(nn.Module):
    # Encodes observed trajectory into latent vector
    def __init__(self,
                 dimension_in,
                 latent_dim,
                 hidden_units,
                 encode_obs_time=False):
        super(ReverseGRUEncoder, self).__init__()
        self.encode_obs_time = encode_obs_time
        if self.encode_obs_time:
            dimension_in += 1
        self.gru = nn.GRU(dimension_in, hidden_units, 2, batch_first=True)
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
class BiGRUEncoder(nn.Module):
    # Encodes observed trajectory into latent vector
    def __init__(self,
                 obs_dim_in,
                 fcst_dim_in,
                 latent_dim,
                 hidden_units,
                 encode_obs_time=False,
                 encode_fcst_time=False):
        super(BiGRUEncoder, self).__init__()
        self.encode_obs_time = encode_obs_time
        self.encode_fcst_time = encode_fcst_time
        if self.encode_obs_time:
            obs_dim_in += 1
        if self.encode_fcst_time:
            fcst_dim_in += 1
        self.gru_obs = nn.GRU(obs_dim_in, hidden_units, 2, batch_first=True)
        self.gru_fcst = nn.GRU(fcst_dim_in, hidden_units, 2, batch_first=True)
        self.linear_out = nn.Linear(hidden_units * 2, latent_dim)
        # nn.init.xavier_uniform_(self.linear_out.weight)

    def forward(self, observed_data, forecast_data, observed_tp, forecast_tp):
        obs_trajs = observed_data  # (batch_size, t_observed_dim, observed_dim)
        if self.encode_obs_time:
            if len(observed_tp) > 1:
                obs_trajs = torch.cat(
                    (observed_data, observed_tp.view(len(observed_tp), -1, 1)),
                    dim=2)
            else:
                obs_trajs = torch.cat((observed_data, observed_tp.view(
                    1, -1, 1).repeat(observed_data.shape[0], 1, 1)),
                                      dim=2)

        if self.encode_fcst_time:
            if len(observed_tp) > 1:
                fcst_trajs = torch.cat(
                    (forecast_data, forecast_tp.view(len(observed_tp), -1, 1)),
                    dim=2)
            else:
                fcst_trajs = torch.cat(
                    (forecast_data, forecast_tp.view(1, -1, 1).repeat(
                        forecast_data.shape[0], 1, 1)),
                    dim=2)
        # reversed_trajs_to_encode = torch.flip(trajs_to_encode, (1, ))
        obs_trajs_to_encode = obs_trajs
        fcst_trajs_to_encode = torch.flip(fcst_trajs, (1, ))
        obs_out, _ = self.gru_obs(obs_trajs_to_encode)
        fcst_out, _ = self.gru_fcst(fcst_trajs_to_encode)
        linear_in = torch.concat([obs_out, fcst_out], axis=-1)
        return self.linear_out(linear_in[:, -1, :])


class MyNeuralLaplace(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 latent_dim=2,
                 hidden_units=64,
                 s_recon_terms=33,
                 use_sphere_projection=True,
                 encode_obs_time=True,
                 ilt_algorithm="fourier",
                 device="cpu",
                 timesteps=None):
        super(MyNeuralLaplace, self).__init__()
        # self.encoder = CNNEncoder(input_dim,
        #                           latent_dim,
        #                           hidden_units // 2,
        #                           timesteps,
        #                           encode_obs_time=encode_obs_time)
        self.encoder = ReverseGRUEncoder(input_dim,
                                         latent_dim,
                                         hidden_units // 2,
                                         encode_obs_time=encode_obs_time)
        # self.output_dense = nn.Sequential(
        #     nn.Linear(output_dim, hidden_units), nn.ReLU(),
        #     nn.Linear(hidden_units, hidden_units), nn.ReLU(),
        #     nn.Linear(hidden_units, output_dim))
        self.use_sphere_projection = use_sphere_projection
        self.output_dim = output_dim
        self.ilt_algorithm = ilt_algorithm
        self.s_recon_terms = s_recon_terms
        if use_sphere_projection:
            self.laplace_rep_func = SphereSurfaceModel(s_recon_terms,
                                                       output_dim, latent_dim)
        else:
            self.laplace_rep_func = SurfaceModel(s_recon_terms, output_dim,
                                                 latent_dim)
        NeuralLaplace.torchlaplace.inverse_laplace.device = device

    def forward(self, observed_data, observed_tp, tp_to_predict):
        # trajs_to_encode : (N, T, D) tensor containing the observed values.
        # tp_to_predict: Is the time to predict the values at.
        p = self.encoder(observed_data, observed_tp)
        out = laplace_reconstruct(
                self.laplace_rep_func,
                p,
                tp_to_predict,
                ilt_reconstruction_terms=self.s_recon_terms,
                recon_dim=self.output_dim,
                use_sphere_projection=self.use_sphere_projection,
                ilt_algorithm=self.ilt_algorithm)
        # out = self.output_dense(out)
        # out = out.reshape(observed_data.shape[0], -1, self.output_dim)
        return out


class GeneralNeuralLaplace(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 latent_dim=2,
                 hidden_units=64,
                 s_recon_terms=33,
                 use_sphere_projection=True,
                 encode_obs_time=True,
                 ilt_algorithm="fourier",
                 device="cpu",
                 **kwargs):
        super(GeneralNeuralLaplace, self).__init__()

        self.model = MyNeuralLaplace(
            input_dim,
            output_dim,
            latent_dim=latent_dim,
            hidden_units=hidden_units,
            s_recon_terms=s_recon_terms,
            use_sphere_projection=use_sphere_projection,
            encode_obs_time=encode_obs_time,
            ilt_algorithm=ilt_algorithm,
            device=device,
            timesteps=kwargs.get("timesteps"))
        self.loss_fn = torch.nn.MSELoss()

    def _get_loss(self, dl):
        cum_loss = 0
        cum_samples = 0
        for batch in dl:
            preds = self.model(batch["observed_data"], batch["observed_tp"],
                               batch["tp_to_predict"])
            cum_loss += self.loss_fn(torch.flatten(preds),
                                     torch.flatten(batch["data_to_predict"])
                                     ) * batch["observed_data"].shape[0]
            cum_samples += batch["observed_data"].shape[0]
        mse = cum_loss / cum_samples
        return mse

    def training_step(self, batch):
        preds = self.model(batch["observed_data"], batch["observed_tp"],
                           batch["tp_to_predict"])
        return self.loss_fn(torch.flatten(preds),
                            torch.flatten(batch["data_to_predict"]))

    def validation_step(self, dlval):
        self.model.eval()
        mse = self._get_loss(dlval)
        return mse, mse

    def test_step(self, dltest):
        self.model.eval()
        mse = self._get_loss(dltest)
        return mse, mse

    def predict(self, dl):
        self.model.eval()
        predictions, trajs = [], []
        for batch in dl:
            predictions.append(
                self.model(batch["observed_data"], batch["observed_tp"],
                           batch["tp_to_predict"]))
            if batch["mode"] == "extrap":
                # trajs.append(batch["data_to_predict"])
                trajs.append(
                    torch.cat(
                        (batch["observed_data"], batch["data_to_predict"]),
                        axis=1))
            else:
                trajs.append(batch["data_to_predict"])
        return torch.cat(predictions, 0), torch.cat(trajs, 0)

    def encode(self, dl):
        encodings = []
        for batch in dl:
            encodings.append(
                self.model.encode(batch["observed_data"],
                                  batch["observed_tp"]))
        return torch.cat(encodings, 0)

    def _get_and_reset_nfes(self):
        """Returns and resets the number of function evaluations for model."""
        iteration_nfes = self.model.laplace_rep_func.nfe
        self.model.laplace_rep_func.nfe = 0
        return iteration_nfes

