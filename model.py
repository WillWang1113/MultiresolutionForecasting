import logging

import torch
from torch import nn
from torchlaplace import laplace_reconstruct
import torchlaplace.inverse_laplace
from encoders import BiEncoder

logger = logging.getLogger()


class SphereSurfaceModel(nn.Module):
    # C^{b+k} -> C^{bxd} - In Riemann Sphere Co ords : b dim s reconstruction terms, k is latent encoding dimension, d is output dimension
    def __init__(
            self,
            s_dim,
            output_dim,
            latent_dim,
            #  out_timesteps,
            include_s_recon_terms=True,
            hidden_units=64):
        super(SphereSurfaceModel, self).__init__()
        self.s_dim = s_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.include_s_recon_terms = include_s_recon_terms
        print("include s recon terms:", include_s_recon_terms)
        dim_in = (2 * s_dim +
                  latent_dim) if include_s_recon_terms else (2 + latent_dim)
        dim_out = (2 * output_dim *
                   s_dim) if include_s_recon_terms else (2 * output_dim)
        self.dim_in = dim_in

        # # TODO: Test RNN
        # self.rnn = nn.RNN(dim_in, hidden_units, batch_first=True, nonlinearity="relu")
        # self.linear_out = nn.Linear(hidden_units, dim_out)

        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(dim_in, hidden_units),
            # nn.BatchNorm1d(hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            # nn.BatchNorm1d(hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, dim_out),
        )
        # self.out_timesteps = out_timesteps

        self.divide_point = (
            self.output_dim *
            self.s_dim) if self.include_s_recon_terms else self.output_dim

        self.phi_max = torch.pi / 2.0
        self.phi_min = -torch.pi / 2.0
        self.phi_scale = self.phi_max - self.phi_min

        self.theta_max = torch.pi
        self.theta_min = -torch.pi
        self.theta_scale = self.theta_max - self.theta_min
        self.nfe = 0

    def forward(self, i):

        # Take in initial conditon p and the Rieman representation
        # If include_s_recon_terms: inputs shape: [batchsize, 2 * s_dim + latent_dim]
        # else                      inputs shape: [batchsize, s_dim, 2 + latent_dim]
        self.nfe += 1
        out = self.linear_tanh_stack(i.view(-1, self.dim_in))

        theta = nn.Tanh()(
            out[..., :self.divide_point]) * torch.pi  # From - pi to + pi
        phi = (nn.Tanh()(out[..., self.divide_point:]) * self.phi_scale / 2.0 -
               torch.pi / 2.0 + self.phi_scale / 2.0
               )  # Form -pi / 2 to + pi / 2
        theta = theta.view(i.shape[0], 1, -1)
        phi = phi.view(i.shape[0], 1, -1)
        return theta, phi


class MyNeuralLaplace(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 latent_dim=2,
                 hidden_units=64,
                 s_recon_terms=33,
                 use_sphere_projection=True,
                 encode_obs_time=True,
                 include_s_recon_terms=True,
                 ilt_algorithm="fourier",
                 device="cpu",
                 encoder="rnn",
                 input_timesteps=None,
                 output_timesteps=None,
                 start_k=0):
        super(MyNeuralLaplace, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = BiEncoder(dimension_in=input_dim,
                                 latent_dim=latent_dim,
                                 hidden_units=hidden_units,
                                 encode_obs_time=encode_obs_time,
                                 encoder="rnn",
                                 timesteps=input_timesteps)

        self.use_sphere_projection = use_sphere_projection
        self.output_dim = output_dim
        self.start_k = start_k
        self.ilt_algorithm = ilt_algorithm
        self.include_s_recon_terms = include_s_recon_terms
        self.s_recon_terms = s_recon_terms

        self.laplace_rep_func = SphereSurfaceModel(
            s_dim=s_recon_terms,
            include_s_recon_terms=include_s_recon_terms,
            output_dim=output_dim,
            latent_dim=latent_dim,
            # out_timesteps=output_timesteps,
        )

        torchlaplace.inverse_laplace.device = device

    # TODO: pass available_forecasts directly into decoder
    def forward(self, observed_data, available_forecasts, observed_tp,
                tp_to_predict):
        # trajs_to_encode : (N, T, D) tensor containing the observed values.
        # tp_to_predict: Is the time to predict the values at.
        p = self.encoder(observed_data, available_forecasts, observed_tp)

        out = laplace_reconstruct(
            self.laplace_rep_func,
            p,
            tp_to_predict,
            ilt_reconstruction_terms=self.s_recon_terms,
            # recon_dim=self.latent_dim,
            recon_dim=self.output_dim,
            use_sphere_projection=self.use_sphere_projection,
            include_s_recon_terms=self.include_s_recon_terms,
            ilt_algorithm=self.ilt_algorithm,
            options={"start_k": self.start_k})
        # out = self.output_dense(out)
        # out = out.reshape(observed_data.shape[0], -1, self.output_dim)
        return out


class HierarchicalNeuralLaplace(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 input_timesteps,
                 output_timesteps,
                 latent_dim=2,
                 hidden_units=64,
                 s_recon_terms=[33, 67, 101],
                 avg_terms_list=[1, 1, 1],
                 use_sphere_projection=True,
                 include_s_recon_terms=True,
                 encode_obs_time=False,
                 ilt_algorithm="fourier",
                 encoder="dnn",
                 device="cpu",
                 pass_raw=False,
                 shared_encoder=False):
        super(HierarchicalNeuralLaplace, self).__init__()
        self.input_timesteps, _ = input_timesteps
        self.output_timesteps = output_timesteps
        self.use_sphere_projection = use_sphere_projection
        self.include_s_recon_terms = include_s_recon_terms
        self.ilt_algorithm = ilt_algorithm
        self.output_dim = output_dim
        self.pass_raw = pass_raw
        self.avg_terms_list = avg_terms_list
        self.shared_encoder = shared_encoder
        print(self.input_timesteps)
        print(self.output_timesteps)


        start_ks = [0]
        for i in range(len(s_recon_terms) - 1):
            start_ks.append(start_ks[-1] + s_recon_terms[i])
        print(start_ks)
        print(s_recon_terms)
        self.start_ks = start_ks
        self.s_recon_terms_list = s_recon_terms

        print(input_dim)
        if shared_encoder:
            self.encoders = nn.ModuleList([
                BiEncoder(dimension_in=input_dim,
                          latent_dim=latent_dim,
                          hidden_units=hidden_units,
                          encode_obs_time=encode_obs_time,
                          encoder=encoder,
                          timesteps=input_timesteps)
            ])
        else:
            self.encoders = nn.ModuleList([
                BiEncoder(dimension_in=input_dim,
                          latent_dim=latent_dim,
                          hidden_units=hidden_units,
                          encode_obs_time=encode_obs_time,
                          encoder=encoder,
                          timesteps=input_timesteps)
                for _ in range(len(s_recon_terms))
            ])
        # recon_steps = self.output_timesteps if pass_raw else self.output_timesteps + self.input_timesteps
        # assert len(avg_terms_list) == len(s_recon_terms)

        self.nlblk_list = nn.ModuleList([
            SphereSurfaceModel(
                s,
                output_dim,
                latent_dim,
                # self.output_timesteps //
                # a,  # different resolution different steps ?
                include_s_recon_terms,
                # hidden_units,
            ) for s in s_recon_terms
        ])

        torchlaplace.inverse_laplace.device = device


    # TODO: Loss on different level?
    def forward(self, observed_data, available_forecasts, observed_tp,
                tp_to_predict):
        all_fcsts, all_recons = [], []
        for i in range(len(self.avg_terms_list)):
            # avg the output timesteps
            avg_tp_to_predict = tp_to_predict[..., None].transpose(1, 2)
            avg_tp_to_predict = torch.nn.functional.avg_pool1d(
                avg_tp_to_predict, self.avg_terms_list[i],
                self.avg_terms_list[i])
            avg_tp_to_predict = avg_tp_to_predict.transpose(
                1, 2).squeeze().unsqueeze(0)

            out = observed_data
            fcsts, recons = 0, 0

            for j in range(i + 1):
                encoder = self.encoders[
                    0] if self.shared_encoder else self.encoders[j]
                nlblk = self.nlblk_list[j]
                if self.pass_raw:
                    p = encoder(observed_data, available_forecasts,
                                observed_tp)
                    fcst = laplace_reconstruct(
                        nlblk,
                        p,
                        avg_tp_to_predict,
                        ilt_reconstruction_terms=self.s_recon_terms_list[j],
                        recon_dim=self.output_dim,
                        use_sphere_projection=self.use_sphere_projection,
                        include_s_recon_terms=self.include_s_recon_terms,
                        ilt_algorithm=self.ilt_algorithm,
                        options={"start_k": self.start_ks[j]})
                    fcsts += fcst
                else:
                    all_tp = torch.cat([observed_tp, avg_tp_to_predict],
                                       axis=-1)
                    p = encoder(out, available_forecasts, observed_tp)
                    temp = laplace_reconstruct(
                        nlblk,
                        p,
                        all_tp,
                        ilt_reconstruction_terms=self.s_recon_terms_list[j],
                        recon_dim=self.output_dim,
                        use_sphere_projection=self.use_sphere_projection,
                        include_s_recon_terms=self.include_s_recon_terms,
                        ilt_algorithm=self.ilt_algorithm,
                        options={"start_k": self.start_ks[j]})

                    fcst, recon = temp[:, self.
                                       input_timesteps:, :], temp[:, :self.
                                                                  input_timesteps, :]
                    out = out - recon
                    fcsts += fcst
                    recons += recon

            all_fcsts.append(fcsts)
            all_recons.append(recons)
        return all_fcsts


    #TODO: fcst1 = 1, fcst2 = 1+2, fcst3 = 1+2+3
    @torch.no_grad()
    def predict(self, observed_data, available_forecasts, observed_tp,
                tp_to_predict):
        self.eval()
        all_fcsts, all_recons = [], []

        for i in range(len(self.avg_terms_list)):
            # avg the output timesteps
            avg_tp_to_predict = tp_to_predict[..., None].transpose(1, 2)
            avg_tp_to_predict = torch.nn.functional.avg_pool1d(
                avg_tp_to_predict, self.avg_terms_list[i],
                self.avg_terms_list[i])
            avg_tp_to_predict = avg_tp_to_predict.transpose(
                1, 2).squeeze().unsqueeze(0)

            out = observed_data
            fcsts, recons = 0, 0

            for j in range(i + 1):
                encoder = self.encoders[
                    0] if self.shared_encoder else self.encoders[j]
                nlblk = self.nlblk_list[j]
                if self.pass_raw:
                    p = encoder(observed_data, available_forecasts,
                                observed_tp)
                    fcst = laplace_reconstruct(
                        nlblk,
                        p,
                        avg_tp_to_predict,
                        ilt_reconstruction_terms=self.s_recon_terms_list[j],
                        recon_dim=self.output_dim,
                        use_sphere_projection=self.use_sphere_projection,
                        include_s_recon_terms=self.include_s_recon_terms,
                        ilt_algorithm=self.ilt_algorithm,
                        options={"start_k": self.start_ks[j]})
                    fcsts += fcst
                else:
                    all_tp = torch.cat([observed_tp, avg_tp_to_predict],
                                       axis=-1)
                    p = encoder(out, available_forecasts, observed_tp)
                    temp = laplace_reconstruct(
                        nlblk,
                        p,
                        all_tp,
                        ilt_reconstruction_terms=self.s_recon_terms_list[j],
                        recon_dim=self.output_dim,
                        use_sphere_projection=self.use_sphere_projection,
                        include_s_recon_terms=self.include_s_recon_terms,
                        ilt_algorithm=self.ilt_algorithm,
                        options={"start_k": self.start_ks[j]})

                    fcst, recon = temp[:, self.
                                       input_timesteps:, :], temp[:, :self.
                                                                  input_timesteps, :]
                    out = out - recon
                    fcsts += fcst
                    recons += recon

            all_fcsts.append(fcsts)
            all_recons.append(recons)
        return all_fcsts, all_recons

    @torch.no_grad()
    def decompose_predcit(self, observed_data, available_forecasts,
                          observed_tp, tp_to_predict):
        self.eval()
        all_fcsts, all_recons = [], []

        for i in range(len(self.avg_terms_list)):
            # avg the output timesteps
            avg_tp_to_predict = tp_to_predict[..., None].transpose(1, 2)
            avg_tp_to_predict = torch.nn.functional.avg_pool1d(
                avg_tp_to_predict, self.avg_terms_list[i],
                self.avg_terms_list[i])
            avg_tp_to_predict = avg_tp_to_predict.transpose(
                1, 2).squeeze().unsqueeze(0)

            out = observed_data
            fcsts, recons = [], []

            for j in range(i + 1):
                encoder = self.encoders[
                    0] if self.shared_encoder else self.encoders[j]
                nlblk = self.nlblk_list[j]
                if self.pass_raw:
                    p = encoder(observed_data, available_forecasts,
                                observed_tp)
                    fcst = laplace_reconstruct(
                        nlblk,
                        p,
                        avg_tp_to_predict,
                        ilt_reconstruction_terms=self.s_recon_terms_list[j],
                        recon_dim=self.output_dim,
                        use_sphere_projection=self.use_sphere_projection,
                        include_s_recon_terms=self.include_s_recon_terms,
                        ilt_algorithm=self.ilt_algorithm,
                        options={"start_k": self.start_ks[j]})
                    fcsts.append(fcst.cpu().numpy())
                else:
                    all_tp = torch.cat([observed_tp, avg_tp_to_predict],
                                       axis=-1)
                    p = encoder(out, available_forecasts, observed_tp)
                    temp = laplace_reconstruct(
                        nlblk,
                        p,
                        all_tp,
                        ilt_reconstruction_terms=self.s_recon_terms_list[j],
                        recon_dim=self.output_dim,
                        use_sphere_projection=self.use_sphere_projection,
                        include_s_recon_terms=self.include_s_recon_terms,
                        ilt_algorithm=self.ilt_algorithm,
                        options={"start_k": self.start_ks[j]})

                    fcst, recon = temp[:, self.
                                       input_timesteps:, :], temp[:, :self.
                                                                  input_timesteps, :]
                    out = out - recon
                    fcsts.append(fcst.cpu().numpy())
                    recons.append(recon.cpu().numpy())
                    # fcsts += fcst
                    # recons += recon

            all_fcsts.append(fcsts)
            all_recons.append(recons)
        return all_fcsts, all_recons


class GeneralHNL(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 input_timesteps,
                 output_timesteps,
                 latent_dim=2,
                 hidden_units=64,
                 s_recon_terms=[33, 67, 101],
                 use_sphere_projection=True,
                 include_s_recon_terms=True,
                 encode_obs_time=True,
                 encoder="dnn",
                 ilt_algorithm="fourier",
                 device="cpu",
                 avg_terms_list=[1, 1, 1],
                 **kwargs):
        super(GeneralHNL, self).__init__()

        self.model = HierarchicalNeuralLaplace(
            input_dim=input_dim,
            output_dim=output_dim,
            input_timesteps=input_timesteps,
            output_timesteps=output_timesteps,
            latent_dim=latent_dim,
            hidden_units=hidden_units,
            s_recon_terms=s_recon_terms,
            use_sphere_projection=use_sphere_projection,
            include_s_recon_terms=include_s_recon_terms,
            encode_obs_time=encode_obs_time,
            ilt_algorithm=ilt_algorithm,
            encoder=encoder,
            device=device,
            pass_raw=kwargs.get("pass_raw", False),
            avg_terms_list=avg_terms_list,
            shared_encoder=kwargs.get("shared_encoder", False))

        self.avg_terms_list = avg_terms_list
        self.loss_fn = torch.nn.MSELoss()

    def _get_loss(self, dl):
        cum_loss = 0
        cum_samples = 0
        for batch in dl:
            preds = self.model(batch["observed_data"],
                               batch["available_forecasts"],
                               batch["observed_tp"], batch["tp_to_predict"])
            loss = 0
            for i, avg_terms in enumerate(self.avg_terms_list):
                data_to_predict = batch["data_to_predict"].transpose(1, 2)
                data_to_predict = torch.nn.functional.avg_pool1d(
                    data_to_predict, avg_terms, avg_terms)
                data_to_predict = data_to_predict.transpose(1, 2)
                resolution_loss = self.loss_fn(torch.flatten(preds[i]),
                                               torch.flatten(data_to_predict))
                loss += resolution_loss
            loss /= len(self.avg_terms_list)
            # cum_loss += recon_loss
            cum_loss += loss * batch["observed_data"].shape[0]
            cum_samples += batch["observed_data"].shape[0]
        mse = cum_loss / cum_samples
        return mse

    def training_step(self, batch):
        preds = self.model(batch["observed_data"],
                           batch["available_forecasts"], batch["observed_tp"],
                           batch["tp_to_predict"])
        loss = 0
        for i, avg_terms in enumerate(self.avg_terms_list):
            data_to_predict = batch["data_to_predict"].transpose(1, 2)
            data_to_predict = torch.nn.functional.avg_pool1d(
                data_to_predict, avg_terms, avg_terms)
            data_to_predict = data_to_predict.transpose(1, 2)
            resolution_loss = self.loss_fn(torch.flatten(preds[i]),
                                           torch.flatten(data_to_predict))
            loss += resolution_loss
        loss /= len(self.avg_terms_list)
        return loss

    @torch.no_grad()
    def validation_step(self, dlval):
        self.model.eval()
        mse = self._get_loss(dlval)
        return mse, mse

    @torch.no_grad()
    def test_step(self, dltest):
        self.model.eval()
        mse = self._get_loss(dltest)
        return mse, mse

    @torch.no_grad()
    def predict(self, dl):
        self.model.eval()
        predictions, trajs = [], []
        for batch in dl:
            preds, _ = self.model.predict(batch["observed_data"],
                                          batch["available_forecasts"],
                                          batch["observed_tp"],
                                          batch["tp_to_predict"])
            predictions.append(preds)
            trajs.append(batch["data_to_predict"])

        out_preds = [torch.concat(f) for f in zip(*predictions)]

        return out_preds, torch.cat(trajs, 0)



class GeneralNeuralLaplace(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 input_timesteps,
                 output_timesteps,
                 latent_dim=2,
                 hidden_units=64,
                 s_recon_terms=33,
                 use_sphere_projection=True,
                 include_s_recon_terms=True,
                 encode_obs_time=True,
                 encoder="rnn",
                 ilt_algorithm="fourier",
                 device="cpu",
                 method="single",
                 **kwargs):
        super(GeneralNeuralLaplace, self).__init__()

        if method == "single" and isinstance(s_recon_terms, int):
            self.model = MyNeuralLaplace(
                input_dim,
                output_dim,
                latent_dim=latent_dim,
                hidden_units=hidden_units,
                s_recon_terms=s_recon_terms,
                use_sphere_projection=use_sphere_projection,
                encode_obs_time=encode_obs_time,
                include_s_recon_terms=include_s_recon_terms,
                ilt_algorithm=ilt_algorithm,
                device=device,
                encoder=encoder,
                input_timesteps=input_timesteps,
                output_timesteps=output_timesteps,
                start_k=kwargs.get("start_k", 0))
        elif method == "hierarchical" and isinstance(s_recon_terms, list):
            self.model = HierarchicalNeuralLaplace(
                input_dim=input_dim,
                output_dim=output_dim,
                input_timesteps=input_timesteps,
                output_timesteps=output_timesteps,
                latent_dim=latent_dim,
                hidden_units=hidden_units,
                s_recon_terms=s_recon_terms,
                use_sphere_projection=use_sphere_projection,
                include_s_recon_terms=include_s_recon_terms,
                encode_obs_time=encode_obs_time,
                ilt_algorithm=ilt_algorithm,
                encoder=encoder,
                device=device,
                pass_raw=kwargs.get("pass_raw", False),
                avg_terms_list=kwargs.get("avg_terms_list", [1, 1, 1]),
                shared_encoder=kwargs.get("shared_encoder", False))

        else:
            raise ValueError(
                "Neural Laplace method can only be 'single' or 'hierarchical'."
            )
        self.method = method
        self.loss_fn = torch.nn.MSELoss()

    def _get_loss(self, dl):
        cum_loss = 0
        cum_samples = 0
        for batch in dl:
            if self.method == "single":
                preds = self.model(batch["observed_data"],
                                   batch["available_forecasts"],
                                   batch["observed_tp"],
                                   batch["tp_to_predict"])
            else:
                preds, _ = self.model(batch["observed_data"],
                                      batch["available_forecasts"],
                                      batch["observed_tp"],
                                      batch["tp_to_predict"])
                # cum_loss += recon_loss
            cum_loss += self.loss_fn(torch.flatten(preds),
                                     torch.flatten(batch["data_to_predict"])
                                     ) * batch["observed_data"].shape[0]
            cum_samples += batch["observed_data"].shape[0]
        mse = cum_loss / cum_samples
        return mse

    def training_step(self, batch):
        if self.method == "single":
            preds = self.model(batch["observed_data"],
                               batch["available_forecasts"],
                               batch["observed_tp"], batch["tp_to_predict"])
            recon_loss = 0
        else:
            preds, recon_loss = self.model(batch["observed_data"],
                                           batch["available_forecasts"],
                                           batch["observed_tp"],
                                           batch["tp_to_predict"])
        # print(preds.shape)
        # print(batch["data_to_predict"].shape)
        loss = self.loss_fn(torch.flatten(preds),
                            torch.flatten(
                                batch["data_to_predict"])) + recon_loss
        return loss

    @torch.no_grad()
    def validation_step(self, dlval):
        self.model.eval()
        mse = self._get_loss(dlval)
        return mse, mse

    @torch.no_grad()
    def test_step(self, dltest):
        self.model.eval()
        mse = self._get_loss(dltest)
        return mse, mse

    @torch.no_grad()
    def predict(self, dl):
        self.model.eval()
        predictions, trajs = [], []
        for batch in dl:
            if self.method == "single":
                preds = self.model(batch["observed_data"],
                                   batch["available_forecasts"],
                                   batch["observed_tp"],
                                   batch["tp_to_predict"])
            else:
                # Return multi-resolution forecasts
                preds, _ = self.model.predict(batch["observed_data"],
                                              batch["available_forecasts"],
                                              batch["observed_tp"],
                                              batch["tp_to_predict"])
            predictions.append(preds)
            trajs.append(batch["data_to_predict"])
        if self.method == "single":
            out_preds = torch.cat(predictions, 0)
        else:
            out_preds = [torch.concat(f) for f in zip(*predictions)]

        # print(out_preds.shape)
        return out_preds, torch.cat(trajs, 0)

