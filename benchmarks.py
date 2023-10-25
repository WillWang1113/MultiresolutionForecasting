import logging

import torch
from torch import nn

logger = logging.getLogger()

device = "cuda" if torch.cuda.is_available() else "cpu"


class GRUEncoder(nn.Module):

    def __init__(self, dimension, hidden_units):
        super(GRUEncoder, self).__init__()
        self.gru = nn.GRU(dimension, hidden_units, 2, batch_first=True)
        self.linear_out = nn.Linear(hidden_units, dimension)

    def forward(self, i):
        out, _ = self.gru(i)
        return self.linear_out(out[:, -1, :])


class LSTMNetwork(nn.Module):

    def __init__(self, in_dim, in_timesteps, hidden_units, out_dim,
                 out_timesteps):
        super(LSTMNetwork, self).__init__()
        hist_dim_in, avail_fcst_dim_in = in_dim
        _, avail_fcst_tsteps = in_timesteps
        self.avail_fcst = False if avail_fcst_dim_in is None else True

        self.lstm = nn.LSTM(hist_dim_in, hidden_units, 2, batch_first=True)

        concat_dim = hidden_units
        if avail_fcst_dim_in is not None:
            concat_dim += avail_fcst_dim_in * avail_fcst_tsteps
        print(self.avail_fcst)

        self.linear_out = nn.Linear(concat_dim, out_dim * out_timesteps)
        self.out_timesteps = out_timesteps
        self.out_dim = out_dim

    def forward(self, observed_data, available_forecasts):
        out, _ = self.lstm(observed_data)

        if self.avail_fcst:
            out = torch.concat((out[:, -1, :], available_forecasts.flatten(1)),
                               axis=-1)
        else:
            out = out[:, -1, :]
        out = self.linear_out(out).reshape(-1, self.out_timesteps,
                                           self.out_dim)

        return out


class MLPNetwork(nn.Module):

    def __init__(self, in_dim, in_timesteps, out_timesteps, hidden_units,
                 out_dim):
        super(MLPNetwork, self).__init__()
        hist_dim_in, avail_fcst_dim_in = in_dim
        hist_tsteps, avail_fcst_tsteps = in_timesteps
        self.avail_fcst = False if avail_fcst_dim_in is None else True
        inputs_dim = hist_dim_in * hist_tsteps
        if avail_fcst_dim_in is not None:
            inputs_dim += avail_fcst_dim_in * avail_fcst_tsteps
        self.nn = nn.Sequential(nn.Linear(inputs_dim, hidden_units),
                                nn.Sigmoid())
        self.linear_out = nn.Linear(hidden_units, out_dim * out_timesteps)
        self.out_timesteps = out_timesteps
        self.out_dim = out_dim
        print(self.avail_fcst)

    def forward(self, observed_data, available_forecasts):
        if self.avail_fcst:
            out = torch.concat(
                (observed_data.flatten(1), available_forecasts.flatten(1)),
                axis=-1)
        else:
            out = observed_data.flatten(1)
        out = self.nn(out)
        out = self.linear_out(out).reshape(-1, self.out_timesteps,
                                           self.out_dim)
        return out


class Persistence(nn.Module):

    def __init__(self, out_timesteps, out_feature, kind="naive"):
        super(Persistence, self).__init__()

        self.out_timesteps = out_timesteps
        self.out_feature = out_feature
        self.kind = kind

    def forward(self, i):
        if self.kind == "naive":
            out = i[:, [-1], :].repeat(1, self.out_timesteps, 1)
        elif self.kind == "loop":
            out = i[:, -self.out_timesteps:, :]
        return out[..., self.out_feature]





class GeneralPersistence(nn.Module):

    def __init__(
        self,
        out_timesteps,
        out_feature,
        method="naive",
    ):
        super(GeneralPersistence, self).__init__()
        if method == "naive":
            self.model = Persistence(out_timesteps=out_timesteps,
                                     out_feature=out_feature,
                                     kind="naive")
        elif method == "loop":
            self.model = Persistence(out_timesteps=out_timesteps,
                                     out_feature=out_feature,
                                     kind="loop")
        else:
            raise ValueError("No such Persistence model.")
        self.loss_fn = torch.nn.MSELoss()

    def _get_loss(self, dl):
        cum_loss = 0
        cum_batches = 0
        for batch in dl:
            preds = self.model(batch["observed_data"])
            cum_loss += self.loss_fn(torch.flatten(preds),
                                     torch.flatten(batch["data_to_predict"]))
            cum_batches += 1
        mse = cum_loss / cum_batches
        return mse

    def training_step(self, batch):
        preds = self.model(batch["observed_data"])
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
            predictions.append(self.model(batch["observed_data"]))
            trajs.append(batch["data_to_predict"])
        return torch.cat(predictions, 0), torch.cat(trajs, 0)



class GeneralNeuralNetwork(nn.Module):

    def __init__(
        self,
        obs_dim,
        out_dim,
        out_timesteps,
        in_timesteps=None,
        nhidden=64,
        method="lstm",
    ):
        super(GeneralNeuralNetwork, self).__init__()
        if method == "lstm":
            self.model = LSTMNetwork(obs_dim, in_timesteps, nhidden, out_dim,
                                     out_timesteps)
        elif method == "mlp":
            self.model = MLPNetwork(obs_dim, in_timesteps, out_timesteps,
                                    nhidden, out_dim)
        else:
            raise ValueError("No such NN model.")
        self.loss_fn = torch.nn.MSELoss()

    def _get_loss(self, dl):
        cum_loss = 0
        cum_batches = 0
        for batch in dl:
            preds = self.model(batch["observed_data"],
                               batch["available_forecasts"])
            cum_loss += self.loss_fn(torch.flatten(preds),
                                     torch.flatten(batch["data_to_predict"]))
            cum_batches += 1
        mse = cum_loss / cum_batches
        return mse

    def training_step(self, batch):
        preds = self.model(batch["observed_data"],
                           batch["available_forecasts"])
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
                self.model(batch["observed_data"],
                           batch["available_forecasts"]))
            trajs.append(batch["data_to_predict"])
        return torch.cat(predictions, 0), torch.cat(trajs, 0)


