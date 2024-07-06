import logging

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

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
        device='cpu'
    ):
        super(GeneralNeuralNetwork, self).__init__()
        self.device = device
        if method == "lstm":
            self.model = LSTMNetwork(obs_dim, in_timesteps, nhidden, out_dim,
                                     out_timesteps)
        elif method == "mlp":
            self.model = MLPNetwork(obs_dim, in_timesteps, out_timesteps,
                                    nhidden, out_dim)
        elif method == "tft":
            self.model = TemporalFusionTransformer(
                obs_dim, in_timesteps, out_timesteps, out_dim
            )
        elif method == "dlinear":
            self.model = DLinear(obs_dim, in_timesteps, out_timesteps, out_dim)
        elif method == "nbeats":
            self.model = NBeatsNet(
                device=device,
                hidden_layer_units=nhidden,
                forecast_length=out_timesteps,
                backcast_length=in_timesteps[0],
                extra_dims=obs_dim[-1],
            )
        else:
            raise ValueError("No such NN model.")
        self.loss_fn = torch.nn.MSELoss()

    def _get_loss(self, dl):
        cum_loss = 0
        cum_batches = 0
        for batch in dl:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(self.device)
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
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(self.device)
            predictions.append(
                self.model(batch["observed_data"],
                           batch["available_forecasts"]))
            trajs.append(batch["data_to_predict"])
        return torch.cat(predictions, 0), torch.cat(trajs, 0)



class GatedLinearUnit(nn.Module):
    def __init__(self, input_size, hidden_layer_size, dropout_rate, activation=None):
        super(GatedLinearUnit, self).__init__()

        self.input_size = input_size
        hidden_size = hidden_layer_size
        dropout = dropout_rate
        self.activation_name = activation

        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        self.W4 = torch.nn.Linear(self.input_size, hidden_size)
        self.W5 = torch.nn.Linear(self.input_size, hidden_size)

        if self.activation_name:
            self.activation = getattr(nn, self.activation_name)()

        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if "bias" not in n:
                torch.nn.init.xavier_uniform_(p)
            #                 torch.nn.init.kaiming_normal_(p, a=0, mode='fan_in', nonlinearity='sigmoid')
            elif "bias" in n:
                torch.nn.init.zeros_(p)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)

        if self.activation_name:
            output = self.sigmoid(self.W4(x)) * self.activation(self.W5(x))
        else:
            output = self.sigmoid(self.W4(x)) * self.W5(x)

        return output


class GateAddNormNetwork(nn.Module):
    def __init__(self, input_size, hidden_layer_size, dropout_rate, activation=None):
        super(GateAddNormNetwork, self).__init__()

        self.input_size = input_size
        hidden_size = hidden_layer_size
        dropout = dropout_rate
        self.activation_name = activation

        self.GLU = GatedLinearUnit(
            self.input_size,
            hidden_size,
            dropout,
            activation=self.activation_name,
        )

        self.LayerNorm = nn.LayerNorm(hidden_size)

    def forward(self, x, skip):
        output = self.LayerNorm(self.GLU(x) + skip)

        return output


class GatedResidualNetwork(nn.Module):
    def __init__(
        self,
        hidden_layer_size,
        input_size=None,
        output_size=None,
        dropout_rate=None,
        # additional_context=None,
        return_gate=False,
    ):
        super(GatedResidualNetwork, self).__init__()

        hidden_size = hidden_layer_size
        self.input_size = input_size if input_size else hidden_size
        self.output_size = output_size
        dropout = dropout_rate
        # self.additional_context = additional_context
        self.return_gate = return_gate

        self.W2 = torch.nn.Linear(self.input_size, hidden_size)
        self.W1 = torch.nn.Linear(hidden_size, hidden_size)

        # if self.additional_context:
        #     self.W3 = torch.nn.Linear(self.additional_context, hidden_size, bias=False)

        if self.output_size:
            self.skip_linear = torch.nn.Linear(self.input_size, self.output_size)
            self.glu_add_norm = GateAddNormNetwork(
                hidden_size, self.output_size, dropout
            )
        else:
            self.glu_add_norm = GateAddNormNetwork(hidden_size, hidden_size, dropout)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if ("W2" in name or "W3" in name) and "bias" not in name:
                torch.nn.init.kaiming_normal_(
                    p, a=0, mode="fan_in", nonlinearity="leaky_relu"
                )
            elif ("skip_linear" in name or "W1" in name) and "bias" not in name:
                torch.nn.init.xavier_uniform_(p)
            #                 torch.nn.init.kaiming_normal_(p, a=0, mode='fan_in', nonlinearity='sigmoid')
            elif "bias" in name:
                torch.nn.init.zeros_(p)

    def forward(self, x):
        # if self.additional_context:
        #     x, context = x
        #     # x_forward = self.W2(x)
        #     # context_forward = self.W3(context)
        #     # print(self.W3(context).shape)
        #     n2 = F.elu(self.W2(x) + self.W3(context))
        # else:
        n2 = F.elu(self.W2(x))

        # print('n2 shape {}'.format(n2.shape))

        n1 = self.W1(n2)

        # print('n1 shape {}'.format(n1.shape))

        if self.output_size:
            output = self.glu_add_norm(n1, self.skip_linear(x))
        else:
            output = self.glu_add_norm(n1, x)

        # print('output shape {}'.format(output.shape))

        return output


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0, scale=True):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=2)
        self.scale = scale

    def forward(self, q, k, v, mask=None):
        # print('---Inputs----')
        # print('q: {}'.format(q[0]))
        # print('k: {}'.format(k[0]))
        # print('v: {}'.format(v[0]))

        attn = torch.bmm(q, k.permute(0, 2, 1))
        # print('first bmm')
        # print(attn.shape)
        # print('attn: {}'.format(attn[0]))

        if self.scale:
            dimention = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32))
            attn = attn / dimention
        #    print('attn_scaled: {}'.format(attn[0]))

        if mask is not None:
            # fill = torch.tensor(-1e9).to(DEVICE)
            # zero = torch.tensor(0).to(DEVICE)
            attn = attn.masked_fill(mask == 0, -1e9)
        #    print('attn_masked: {}'.format(attn[0]))

        attn = self.softmax(attn)
        # print('attn_softmax: {}'.format(attn[0]))
        attn = self.dropout(attn)

        output = torch.bmm(attn, v)

        return output, attn


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout):
        super(InterpretableMultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = self.d_q = self.d_v = d_model // n_head
        self.dropout = nn.Dropout(p=dropout)

        self.v_layer = nn.Linear(self.d_model, self.d_v, bias=False)
        self.q_layers = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_q, bias=False) for _ in range(self.n_head)]
        )
        self.k_layers = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_k, bias=False) for _ in range(self.n_head)]
        )
        self.v_layers = nn.ModuleList([self.v_layer for _ in range(self.n_head)])
        self.attention = ScaledDotProductAttention()
        self.w_h = nn.Linear(self.d_v, self.d_model, bias=False)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" not in name:
                torch.nn.init.xavier_uniform_(p)
            #                 torch.nn.init.kaiming_normal_(p, a=0, mode='fan_in', nonlinearity='sigmoid')
            else:
                torch.nn.init.zeros_(p)

    def forward(self, q, k, v, mask=None):
        heads = []
        attns = []
        for i in range(self.n_head):
            qs = self.q_layers[i](q)
            ks = self.k_layers[i](k)
            vs = self.v_layers[i](v)
            # print('qs layer: {}'.format(qs.shape))
            head, attn = self.attention(qs, ks, vs, mask)
            # print('head layer: {}'.format(head.shape))
            # print('attn layer: {}'.format(attn.shape))
            head_dropout = self.dropout(head)
            heads.append(head_dropout)
            attns.append(attn)

        head = torch.stack(heads, dim=2) if self.n_head > 1 else heads[0]
        # print('concat heads: {}'.format(head.shape))
        # print('heads {}: {}'.format(0, head[0,0,Ellipsis]))
        attn = torch.stack(attns, dim=2)
        # print('concat attn: {}'.format(attn.shape))

        outputs = torch.mean(head, dim=2) if self.n_head > 1 else head
        # print('outputs mean: {}'.format(outputs.shape))
        # print('outputs mean {}: {}'.format(0, outputs[0,0,Ellipsis]))
        outputs = self.w_h(outputs)
        outputs = self.dropout(outputs)

        return outputs, attn


class VariableSelectionNetwork(nn.Module):
    def __init__(
        self,
        # timesteps,
        hidden_layer_size,
        dropout_rate,
        output_size,
        input_size=None,
        # additional_context=None,
    ):
        super(VariableSelectionNetwork, self).__init__()

        self.hidden_size = hidden_layer_size
        self.input_size = input_size
        self.output_size = output_size
        dropout = dropout_rate
        # self.additional_context = additional_context

        self.flattened_grn = GatedResidualNetwork(
            self.hidden_size,
            input_size=self.input_size,
            output_size=self.output_size,
            dropout_rate=dropout,
            # additional_context=self.additional_context,
        )

        self.per_feature_grn = nn.ModuleList(
            [
                GatedResidualNetwork(self.hidden_size, dropout_rate=dropout)
                for i in range(self.output_size)
            ]
        )

    def forward(self, x):
        # Non Static Inputs
        embedding = x
        # print('x')
        # print(x.shape)

        # time_steps = embedding.shape[0]
        time_steps = embedding.shape[1]
        flatten = embedding.view(-1, time_steps, self.hidden_size * self.output_size)
        # print('flatten')
        # print(flatten.shape)

        # static_context = static_context.unsqueeze(1)
        # print('static_context')
        # print(static_context.shape)

        # Nonlinear transformation with gated residual network.
        mlp_outputs = self.flattened_grn(flatten)
        # print('mlp_outputs')
        # print(mlp_outputs.shape)

        sparse_weights = F.softmax(mlp_outputs, dim=-1)
        sparse_weights = sparse_weights.unsqueeze(2)
        # print('sparse_weights')
        # print(sparse_weights.shape)

        trans_emb_list = []
        for i in range(self.output_size):
            e = self.per_feature_grn[i](embedding[Ellipsis, i])
            trans_emb_list.append(e)
        transformed_embedding = torch.stack(trans_emb_list, axis=-1)
        # print('transformed_embedding')
        # print(transformed_embedding.shape)

        combined = sparse_weights * transformed_embedding
        # print('combined')
        # print(combined.shape)

        # temporal_ctx = combined
        temporal_ctx = torch.sum(combined, dim=-1)
        # print('temporal_ctx')
        # print(temporal_ctx.shape)

        # # Static Inputs
        # else:
        #     embedding = x
        #     # print('embedding')
        #     # print(embedding.shape)

        #     flatten = torch.flatten(embedding, start_dim=1)
        #     # flatten = embedding.view(batch_size, -1)
        #     # print('flatten')
        #     # print(flatten.shape)

        #     # Nonlinear transformation with gated residual network.
        #     mlp_outputs = self.flattened_grn(flatten)
        #     # print('mlp_outputs')
        #     # print(mlp_outputs.shape)

        #     sparse_weights = F.softmax(mlp_outputs, dim=-1)
        #     sparse_weights = sparse_weights.unsqueeze(-1)
        #     #             print('sparse_weights')
        #     #             print(sparse_weights.shape)

        #     trans_emb_list = []
        #     for i in range(self.output_size):
        #         # print('embedding for the per feature static grn')
        #         # print(embedding[:, i:i + 1, :].shape)
        #         e = self.per_feature_grn[i](embedding[:, i : i + 1, :])
        #         trans_emb_list.append(e)
        #     transformed_embedding = torch.cat(trans_emb_list, axis=1)
        #     #             print('transformed_embedding')
        #     #             print(transformed_embedding.shape)

        #     combined = sparse_weights * transformed_embedding
        #     #             print('combined')
        #     #             print(combined.shape)

        #     temporal_ctx = torch.sum(combined, dim=1)
        # #             print('temporal_ctx')
        # #             print(temporal_ctx.shape)

        return temporal_ctx, sparse_weights


class TemporalFusionTransformer(nn.Module):
    def __init__(
        self,
        in_dim,
        in_timesteps,
        out_timesteps,
        out_dim,
        hidden_size=32,
        dropout=0.1,
        n_head=4,
    ) -> None:
        super().__init__()
        observed_data_dim, future_feature_dim = in_dim
        input_timesteps, _ = in_timesteps

        self.output_timesteps = out_timesteps
        self.hist_embed = nn.Linear(1, hidden_size)
        self.future_embed = nn.Linear(1, hidden_size)

        self.temporal_historical_vsn = VariableSelectionNetwork(
            # timesteps=input_timesteps,
            hidden_layer_size=hidden_size,
            input_size=hidden_size * observed_data_dim,
            output_size=observed_data_dim,
            dropout_rate=dropout,
        )

        self.temporal_future_vsn = VariableSelectionNetwork(
            # timesteps=out_timesteps,
            hidden_layer_size=hidden_size,
            input_size=hidden_size * future_feature_dim,
            output_size=future_feature_dim,
            dropout_rate=dropout,
        )

        self.historical_lstm = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.future_lstm = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.post_seq_encoder_gate_add_norm = GateAddNormNetwork(
            hidden_size,
            hidden_size,
            dropout,
            activation=None,
        )

        self.self_attn_layer = InterpretableMultiHeadAttention(
            n_head=n_head,
            d_model=hidden_size,
            dropout=dropout,
        )

        self.post_attn_gate_add_norm = GateAddNormNetwork(
            hidden_size,
            hidden_size,
            dropout,
            activation=None,
        )

        self.GRN_positionwise = GatedResidualNetwork(hidden_size, dropout_rate=dropout)

        self.post_tfd_gate_add_norm = GateAddNormNetwork(
            hidden_size,
            hidden_size,
            dropout,
            activation=None,
        )

        self.static_enrichment = GatedResidualNetwork(hidden_size, dropout_rate=dropout)

        self.output_feed_forward = torch.nn.Linear(hidden_size, out_dim)
        self.data_to_predict_dim = out_dim

    def forward(self, observed_data, available_forecasts: torch.Tensor):
        if self.output_timesteps != available_forecasts.shape[1]:
            interp_af = available_forecasts.permute(0, 2, 1)
            interp_af = torch.nn.functional.interpolate(
                interp_af, size=self.output_timesteps, mode="linear"
            )
            interp_af = interp_af.permute(0, 2, 1)
        else:
            interp_af = available_forecasts
        hidden_obs, hidden_futfeat = [], []
        for i in range(observed_data.shape[-1]):
            hidden_obs.append(self.hist_embed(observed_data[Ellipsis, i : i + 1]))

        for i in range(available_forecasts.shape[-1]):
            hidden_futfeat.append(
                self.future_embed(available_forecasts[Ellipsis, i : i + 1])
            )

        # hidden_obs = self.hist_embed(observed_data)
        # hidden_futfeat = self.future_embed(interp_af)
        # print(hidden_obs.shape)
        # print(hidden_futfeat.shape)
        hidden_obs = torch.stack(hidden_obs, dim=-1)
        hidden_futfeat = torch.stack(hidden_futfeat, dim=-1)

        select_obs, obs_weight = self.temporal_historical_vsn(hidden_obs)
        select_futfeat, futfeat_weight = self.temporal_future_vsn(hidden_futfeat)

        # select_obs, obs_weight = self.temporal_historical_vsn(hidden_obs)
        # select_futfeat, futfeat_weight = self.temporal_future_vsn(hidden_futfeat)

        # print(select_obs.shape)
        # print(select_futfeat.shape)

        lstm_obs, _ = self.historical_lstm(select_obs)
        lstm_futfeat, _ = self.future_lstm(select_futfeat)

        input_embeddings = torch.cat((select_obs, select_futfeat), axis=1)
        lstm_layer = torch.cat((lstm_obs, lstm_futfeat), axis=1)
        # print(lstm_layer.shape)
        # print(input_embeddings.shape)

        temporal_feature_layer = self.post_seq_encoder_gate_add_norm(
            lstm_layer, input_embeddings
        )

        enriched = self.static_enrichment(temporal_feature_layer)

        x, self_att = self.self_attn_layer(enriched, enriched, enriched)
        # x, self_att = self.self_attn_layer(
        #     enriched, enriched, enriched, mask=self.get_decoder_mask(enriched)
        # )

        x = self.post_attn_gate_add_norm(x, enriched)

        decoder = self.GRN_positionwise(x)

        # Final skip connection
        transformer_layer = self.post_tfd_gate_add_norm(decoder, temporal_feature_layer)

        outputs = self.output_feed_forward(
            transformer_layer[Ellipsis, -self.output_timesteps :, :]
        )

        return outputs

    def get_decoder_mask(self, self_attn_inputs):
        """Returns causal mask to apply for self-attention layer.
        Args:
        self_attn_inputs: Inputs to self attention layer to determine mask shape
        """
        len_s = self_attn_inputs.shape[1]
        bs = self_attn_inputs.shape[0]
        mask = torch.cumsum(torch.eye(len_s, device=self_attn_inputs.device), 0)
        mask = mask.repeat(bs, 1, 1).to(torch.float32)

        return mask



class NBeatsNet(nn.Module):
    SEASONALITY_BLOCK = "seasonality"
    TREND_BLOCK = "trend"
    GENERIC_BLOCK = "generic"

    def __init__(
        self,
        device=torch.device("cpu"),
        stack_types=(TREND_BLOCK, SEASONALITY_BLOCK),
        nb_blocks_per_stack=3,
        forecast_length=5,
        backcast_length=10,
        extra_dims=1,
        thetas_dim=(4, 8),
        share_weights_in_stack=False,
        hidden_layer_units=256,
        nb_harmonics=None,
    ):
        super(NBeatsNet, self).__init__()
        self.extra_dims = extra_dims
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.hidden_layer_units = hidden_layer_units
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.nb_harmonics = nb_harmonics
        self.stack_types = stack_types
        self.stacks = []
        self.thetas_dim = thetas_dim
        self.parameters = []
        self.device = device
        print("| N-Beats")
        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))
        self.parameters = nn.ParameterList(self.parameters)
        self.to(self.device)
        self._loss = None
        self._opt = None
        self._gen_intermediate_outputs = False
        self._intermediary_outputs = []

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        print(
            f"| --  Stack {stack_type.title()} (#{stack_id}) (share_weights_in_stack={self.share_weights_in_stack})"
        )
        blocks = []
        for block_id in range(self.nb_blocks_per_stack):
            block_init = NBeatsNet.select_block(stack_type)
            if self.share_weights_in_stack and block_id != 0:
                block = blocks[-1]  # pick up the last one when we share weights.
            else:
                block = block_init(
                    self.hidden_layer_units,
                    self.thetas_dim[stack_id],
                    self.device,
                    self.backcast_length,
                    self.forecast_length,
                    self.extra_dims,
                    self.nb_harmonics,
                )
                self.parameters.extend(block.parameters())
            print(f"     | -- {block}")
            blocks.append(block)
        return blocks

    @staticmethod
    def select_block(block_type):
        if block_type == NBeatsNet.SEASONALITY_BLOCK:
            return SeasonalityBlock
        elif block_type == NBeatsNet.TREND_BLOCK:
            return TrendBlock
        else:
            return GenericBlock

    def get_generic_and_interpretable_outputs(self):
        g_pred = sum(
            [
                a["value"][0]
                for a in self._intermediary_outputs
                if "generic" in a["layer"].lower()
            ]
        )
        i_pred = sum(
            [
                a["value"][0]
                for a in self._intermediary_outputs
                if "generic" not in a["layer"].lower()
            ]
        )
        outputs = {o["layer"]: o["value"][0] for o in self._intermediary_outputs}
        return g_pred, i_pred, outputs

    def forward(self, observed_data, available_forecasts):
        if self.forecast_length != available_forecasts.shape[1]:
            interp_af = available_forecasts.permute(0, 2, 1)
            interp_af = torch.nn.functional.interpolate(
                interp_af, size=self.forecast_length, mode="linear"
            )
            interp_af = interp_af.permute(0, 2, 1)
        else:
            interp_af = available_forecasts

        self._intermediary_outputs = []
        observed_data = squeeze_last_dim(observed_data)
        forecast = torch.zeros(
            size=(
                observed_data.size()[0],
                self.forecast_length,
            )
        )  # maybe batch size here.
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id]((observed_data, interp_af))
                observed_data = observed_data.to(self.device) - b
                forecast = forecast.to(self.device) + f
                block_type = self.stacks[stack_id][block_id].__class__.__name__
                layer_name = f"stack_{stack_id}-{block_type}_{block_id}"
                if self._gen_intermediate_outputs:
                    self._intermediary_outputs.append(
                        {"value": f.detach().numpy(), "layer": layer_name}
                    )
        return forecast.unsqueeze(-1)
        # return observed_data, forecast


def squeeze_last_dim(tensor):
    if len(tensor.shape) == 3 and tensor.shape[-1] == 1:  # (128, 10, 1) => (128, 10).
        return tensor[..., 0]
    return tensor


def seasonality_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p <= thetas.shape[1], "thetas_dim is too big."
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    s1 = torch.tensor(
        np.array([np.cos(2 * np.pi * i * t) for i in range(p1)])
    ).float()  # H/2-1
    s2 = torch.tensor(np.array([np.sin(2 * np.pi * i * t) for i in range(p2)])).float()
    S = torch.cat([s1, s2])
    return thetas.mm(S.to(device))


def trend_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p <= 4, "thetas_dim is too big."
    T = torch.tensor(np.array([t**i for i in range(p)])).float()
    return thetas.mm(T.to(device))


def linear_space(backcast_length, forecast_length, is_forecast=True):
    horizon = forecast_length if is_forecast else backcast_length
    return np.arange(0, horizon) / horizon


class Block(nn.Module):
    def __init__(
        self,
        units,
        thetas_dim,
        device,
        backcast_length=10,
        forecast_length=5,
        extra_dims=1,
        share_thetas=False,
        nb_harmonics=None,
    ):
        super(Block, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas
        self.fc1 = nn.Linear(extra_dims * forecast_length + backcast_length, units)
        # self.fc1 = nn.Linear(backcast_length*extra_dims, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.fc4 = nn.Linear(units, units)
        self.device = device
        self.backcast_linspace = linear_space(
            backcast_length, forecast_length, is_forecast=False
        )
        self.forecast_linspace = linear_space(
            backcast_length, forecast_length, is_forecast=True
        )
        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
        else:
            self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
            self.theta_f_fc = nn.Linear(units, thetas_dim, bias=False)

    def forward(self, x):
        b, f = x
        x = torch.concat([b.flatten(1), f.flatten(1)], dim=-1)
        x = squeeze_last_dim(x)
        x = F.relu(self.fc1(x.to(self.device)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

    def __str__(self):
        block_type = type(self).__name__
        return (
            f"{block_type}(units={self.units}, thetas_dim={self.thetas_dim}, "
            f"backcast_length={self.backcast_length}, forecast_length={self.forecast_length}, "
            f"share_thetas={self.share_thetas}) at @{id(self)}"
        )


class SeasonalityBlock(Block):
    def __init__(
        self,
        units,
        thetas_dim,
        device,
        backcast_length=10,
        forecast_length=5,
        extra_dim=1,
        nb_harmonics=None,
    ):
        if nb_harmonics:
            super(SeasonalityBlock, self).__init__(
                units,
                nb_harmonics,
                device,
                backcast_length,
                forecast_length,
                extra_dim,
                share_thetas=True,
            )
        else:
            super(SeasonalityBlock, self).__init__(
                units,
                forecast_length,
                device,
                backcast_length,
                forecast_length,
                extra_dim,
                share_thetas=True,
            )

    def forward(self, x):
        x = super(SeasonalityBlock, self).forward(x)
        backcast = seasonality_model(
            self.theta_b_fc(x), self.backcast_linspace, self.device
        )
        forecast = seasonality_model(
            self.theta_f_fc(x), self.forecast_linspace, self.device
        )
        return backcast, forecast


class TrendBlock(Block):
    def __init__(
        self,
        units,
        thetas_dim,
        device,
        backcast_length=10,
        forecast_length=5,
        extra_dim=1,
        nb_harmonics=None,
    ):
        super(TrendBlock, self).__init__(
            units,
            thetas_dim,
            device,
            backcast_length,
            forecast_length,
            extra_dim,
            share_thetas=True,
        )

    def forward(self, x):
        x = super(TrendBlock, self).forward(x)
        backcast = trend_model(self.theta_b_fc(x), self.backcast_linspace, self.device)
        forecast = trend_model(self.theta_f_fc(x), self.forecast_linspace, self.device)
        return backcast, forecast


class GenericBlock(Block):
    def __init__(
        self,
        units,
        thetas_dim,
        device,
        backcast_length=10,
        forecast_length=5,
        extra_dim=1,
        nb_harmonics=None,
    ):
        super(GenericBlock, self).__init__(
            units, thetas_dim, device, backcast_length, forecast_length, extra_dim
        )

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x):
        # no constraint for generic arch.
        x = super(GenericBlock, self).forward(x)

        theta_b = self.theta_b_fc(x)
        theta_f = self.theta_f_fc(x)

        backcast = self.backcast_fc(theta_b)  # generic. 3.3.
        forecast = self.forecast_fc(theta_f)  # generic. 3.3.

        return backcast, forecast
    


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    """
    DLinear
    """

    def __init__(self, in_dim, in_timesteps, out_timesteps, out_dim):
        super(DLinear, self).__init__()
        hist_dim_in, avail_fcst_dim_in = in_dim
        hist_tsteps, avail_fcst_tsteps = in_timesteps

        self.seq_len = hist_tsteps
        self.pred_len = out_timesteps

        # Decompsition Kernel Size
        kernel_size = out_timesteps // 4 - 1
        self.decompsition = series_decomp(kernel_size)
        # self.individual = configs.individual
        self.channels = hist_dim_in

        # if self.individual:
        self.Linear_Seasonal = nn.ModuleList()
        self.Linear_Trend = nn.ModuleList()
        self.Linear_Decoder = nn.ModuleList()
        for i in range(self.channels):
            self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
            self.Linear_Seasonal[i].weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
            )
            self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
            self.Linear_Trend[i].weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
            )
            self.Linear_Decoder.append(nn.Linear(self.seq_len, self.pred_len))
        # else:
        #     self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
        #     self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
        #     self.Linear_Decoder = nn.Linear(self.seq_len,self.pred_len)
        #     self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        #     self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        if avail_fcst_dim_in is not None:
            concat_dim = avail_fcst_dim_in * avail_fcst_tsteps + out_timesteps * out_dim

        self.linear_out = nn.Linear(concat_dim, out_dim * out_timesteps)
        self.out_timesteps = out_timesteps
        self.out_dim = out_dim

    def forward(self, observed_data, available_forecasts):
        # x: [Batch, Input length, Channel]
        x = observed_data
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = (
            seasonal_init.permute(0, 2, 1),
            trend_init.permute(0, 2, 1),
        )
        # if self.individual:
        seasonal_output = torch.zeros(
            [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
            dtype=seasonal_init.dtype,
        ).to(observed_data.device)
        trend_output = torch.zeros(
            [trend_init.size(0), trend_init.size(1), self.pred_len],
            dtype=trend_init.dtype,
        ).to(observed_data.device)
        for i in range(self.channels):
            seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
            trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        # else:
        #     seasonal_output = self.Linear_Seasonal(seasonal_init)
        #     trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        x = x.permute(0, 2, 1)
        x = (
            self.linear_out(
                torch.concat([x.flatten(1), available_forecasts.flatten(1)], axis=1)
            ).reshape(-1, self.out_timesteps, self.out_dim)
            + x
        )

        return x  # to [Batch, Output length, Channel]
