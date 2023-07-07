# import shelve
# from functools import partial
# import numpy as np
# import pandas as pd
# import scipy.io as sio
# import torch
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from ddeint import ddeint
# from torchlaplace.data_utils import basic_collate_fn
# from sklearn.preprocessing import StandardScaler

# Real-world dataset
# def solete_60min(device, double=False, features=["WIND_SPEED[m1s]"], window_width=48):
#     df = pd.read_csv("SOLETE_data/SOLETE_Pombo_60min.csv",
#                      index_col=0,
#                      parse_dates=True,
#                      infer_datetime_format=True)
#     df = df.sort_index()
#     df = df[features].values
#     trajs = []
#     start = 0
#     while start + window_width < len(df):
#         end = start + window_width
#         trajs.append(df[start:end])
#         start += 1
#     if len(trajs[-1]) != len(trajs[-2]):
#         trajs.pop()
#     trajs = np.stack(trajs, axis=0)
#     t = torch.linspace(20 / window_width, 20, window_width)
#     if double:
#         t = t.to(device).double()
#         trajs = torch.from_numpy(trajs).to(device).double()
#     else:
#         t = t.to(device)
#         trajs = torch.from_numpy(trajs).to(device)
#     return trajs, t.unsqueeze(0)

# def solete_5min(device, features=["WIND_SPEED[m1s]"]):
#     df = pd.read_csv("SOLETE_data/SOLETE_Pombo_5min.csv",
#                      index_col=0,
#                      parse_dates=True,
#                      infer_datetime_format=True)
#     df = df.sort_index()
#     df = df[features].values
#     trajs = []

#     start = 0
#     while start + 48 < len(df):
#         end = start + 48
#         trajs.append(df[start:end])
#         start += 12
#     if len(trajs[-1]) != len(trajs[-2]):
#         trajs.pop()
#     trajs = np.stack(trajs, axis=0)
#     print(trajs.shape)
#     t = torch.linspace(20 / (48), 20, 48)
#     return torch.from_numpy(trajs).to(device).float(), t.to(device).float()

# def traj_sine(device,
#               double=False,
#               trajectories_to_sample=100,
#               t_nsamples=200):
#     """Generate sine data in the trajectories fashion

#     Args:
#         device (int): device
#         double (bool, optional): dtype. Defaults to False.
#         trajectories_to_sample (int, optional): samples. Defaults to 100.
#         t_nsamples (int, optional): time steps in each sample. Defaults to 200.

#     Returns:
#         tuple: trajs, t
#     """
#     t_end = 20.0
#     t_begin = t_end / t_nsamples
#     if double:
#         ti = torch.linspace(t_begin, t_end, t_nsamples).to(device).double()
#     else:
#         ti = torch.linspace(t_begin, t_end, t_nsamples).to(device)

#     def sampler(t, x0=0):
#         return torch.sin(t + x0)
#         return torch.sin(t + x0) + torch.sin(
#             2 * (t + x0)) + 0.5 * torch.sin(11 * (t + x0))

#     x0s = torch.linspace(0, 2 * torch.pi, trajectories_to_sample)
#     trajs = []
#     for x0 in x0s:
#         trajs.append(sampler(ti, x0))
#     y = torch.stack(trajs)
#     trajectories = y.view(trajectories_to_sample, -1, 1)
#     print(trajectories.shape)
#     print(ti.shape)
#     return trajectories, ti

# def time_sine(device,
#               double=False,
#               trajectories_to_sample=100,
#               t_nsamples=201):
#     """Generate sine data in time series fashion

#     Args:
#         device (_type_): _description_
#         double (bool, optional): _description_. Defaults to False.
#         trajectories_to_sample (int, optional): _description_. Defaults to 100.
#         t_nsamples (int, optional): _description_. Defaults to 201.

#     Returns:
#         _type_: _description_
#     """
#     # (total_length - window_width) / stride - 1 = n_windows
#     t_end = 20.0
#     t_begin = t_end / t_nsamples
#     window_width = t_nsamples - (trajectories_to_sample + 1)
#     if double:
#         ti = torch.linspace(t_begin, t_end, t_nsamples).to(device).double()
#     else:
#         ti = torch.linspace(t_begin, t_end, t_nsamples).to(device)

#     def sampler(t):
#         return torch.sin(t)
#         return torch.sin(t) + torch.sin(2 * (t)) + 0.5 * torch.sin(11 * (t))

#     traj = sampler(ti)
#     # traj = torch.cat([traj.reshape(-1,1), ti.reshape(-1,1)], axis=-1)
#     start = 0
#     trajs, t = [], []
#     for i in range(trajectories_to_sample):
#         end = window_width + start
#         trajs.append(traj[start:end].unsqueeze(-1))
#         t.append(ti[start:end])
#         start += 1
#     if trajs[-1].shape != trajs[0].shape:
#         trajs.pop()
#         t.pop()
#     y, t = torch.stack(trajs), torch.stack(t)
#     print(y.shape)
#     print(t.shape)

#     return y, t

# def generate_data(
#     device,
#     dataset,
#     features=["WIND_SPEED[m1s]"],
#     double=False,
#     batch_size=128,
#     extrap=0,
#     percent_missing_at_random=0.0,
#     normalize=True,
#     test_set_out_of_distribution=True,
#     noise_std=None,
#     observe_step=1,
#     predict_step=1,
# ):
#     # trajectories, t = sine(device)
#     if dataset == "solete_5min":
#         trajectories, t = solete_5min(device, features=features)
#     elif dataset == "solete_60min":
#         trajectories, t = solete_60min(device, features=features)
#     elif dataset == "traj_sine":
#         trajectories, t = traj_sine(device=device, double=double)
#     elif dataset == "time_sine":
#         trajectories, t = time_sine(device=device, double=double)
#     else:
#         raise ValueError("no such dataset")
#     if not extrap:
#         bool_mask = torch.FloatTensor(
#             *trajectories.shape).uniform_() < (1.0 - percent_missing_at_random)
#         if double:
#             float_mask = (bool_mask).float().double().to(device)
#         else:
#             float_mask = (bool_mask).float().to(device)
#         trajectories = float_mask * trajectories

#     # # normalize
#     # if normalize:
#     #     samples = trajectories.shape[0]
#     #     dim = trajectories.shape[2]
#     #     traj = (torch.reshape(trajectories, (-1, dim)) - torch.reshape(
#     #         trajectories,
#     #         (-1, dim)).mean(0)) / torch.reshape(trajectories, (-1, dim)).std(0)
#     #     trajectories = torch.reshape(traj, (samples, -1, dim))

#     if noise_std:
#         trajectories += torch.randn(trajectories.shape).to(device) * noise_std

#     train_split = int(0.8 * trajectories.shape[0])
#     test_split = int(0.9 * trajectories.shape[0])

#     if test_set_out_of_distribution:
#         train_trajectories = trajectories[:train_split]
#         train_t = t[:train_split]

#         val_trajectories = trajectories[train_split:test_split]
#         val_t = t[train_split:test_split]

#         test_trajectories = trajectories[test_split:]
#         test_t = t[test_split:]
#     else:
#         traj_index = torch.randperm(trajectories.shape[0])
#         train_trajectories = trajectories[traj_index[:train_split]]
#         train_t = t[traj_index[:train_split]]
#         val_trajectories = trajectories[traj_index[train_split:test_split]]
#         val_t = t[traj_index[train_split:test_split]]
#         test_trajectories = trajectories[traj_index[test_split:]]
#         test_t = t[traj_index[test_split:]]

#     if normalize:
#         len_train, len_val, len_test = len(train_trajectories), len(val_trajectories), len(test_trajectories)
#         dim = trajectories.shape[2]
#         train_mean = torch.reshape(train_trajectories, (-1, dim)).mean(0)
#         train_std = torch.reshape(train_trajectories, (-1, dim)).std(0)
#         train_trajectories = (torch.reshape(train_trajectories, (-1, dim)) -
#                               train_mean) / train_std
#         val_trajectories = (torch.reshape(val_trajectories, (-1, dim)) -
#                               train_mean) / train_std
#         test_trajectories = (torch.reshape(test_trajectories, (-1, dim)) -
#                               train_mean) / train_std
#         train_trajectories = train_trajectories.reshape((len_train, -1, dim))
#         val_trajectories = val_trajectories.reshape((len_val, -1, dim))
#         test_trajectories = test_trajectories.reshape((len_test, -1, dim))

#     print("train shape:\t", train_trajectories.shape)
#     print("valid shape:\t", val_trajectories.shape)
#     print("test shape:\t", test_trajectories.shape)
#     test_plot_traj = test_trajectories[0]

#     input_dim = train_trajectories.shape[2]
#     output_dim = input_dim

#     dltrain = DataLoader(
#         train_trajectories,
#         batch_size=batch_size,
#         shuffle=True,
#         collate_fn=lambda batch: basic_collate_fn(
#             batch,
#             t,
#             # train_t,
#             data_type="train",
#             extrap=extrap,
#             observe_step=observe_step,
#             predict_step=predict_step,
#         ),
#     )
#     dlval = DataLoader(
#         val_trajectories,
#         batch_size=batch_size,
#         shuffle=False,
#         collate_fn=lambda batch: basic_collate_fn(
#             batch,
#             t,
#             # val_t,
#             data_type="test",
#             extrap=extrap,
#             observe_step=observe_step,
#             predict_step=predict_step,
#         ),
#     )
#     dltest = DataLoader(
#         test_trajectories,
#         batch_size=batch_size,
#         shuffle=False,
#         collate_fn=lambda batch: basic_collate_fn(
#             batch,
#             t,
#             # test_t,
#             data_type="test",
#             extrap=extrap,
#             observe_step=observe_step,
#             predict_step=predict_step,
#         ),
#     )
#     return (
#         input_dim,
#         output_dim,
#         dltrain,
#         dlval,
#         dltest,
#         t,
#         train_trajectories,
#         val_trajectories,
#         test_trajectories,
#         train_t,
#         val_t,
#         test_t,
#     )

# # sine_w_t(0, t_nsamples=301)

# # sine(0)

# # (
# #     input_dim,
# #     output_dim,
# #     dltrain,
# #     dlval,
# #     dltest,
# #     t,
# #     train_trajectories,
# #     val_trajectories,
# #     test_trajectories,
# #     train_t,
# #     val_t,
# #     test_t,
# # ) = generate_data(0, dataset="sine_w_t", extrap=1)

# # for b in dltrain:
# #     for k in b:
# #         print(k)
# #         print(b[k].shape)
# #     break

###########################
# Neural Laplace: Learning diverse classes of differential equations in the Laplace domain
# Author: Samuel Holt
###########################
import shelve
from functools import partial
import numpy as np
import scipy.io as sio
import torch
from ddeint import ddeint
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import pandas as pd
from utils import setup_seed

# from torchlaplace.data_utils import (
#     basic_collate_fn
# )
from NeuralLaplace.torchlaplace.data_utils import basic_collate_fn

from pathlib import Path

local_path = Path(__file__).parent

# DE Datasets


def lotka_volterra_system_with_delay(device,
                                     double=False,
                                     trajectories_to_sample=100,
                                     t_nsamples=200):

    def model(Y, t, d):
        x, y = Y(t)
        xd, yd = Y(t - d)
        return np.array([0.5 * x * (1 - yd), -0.5 * y * (1 - xd)])

    subsample_to_points = t_nsamples
    compute_points = 1000
    tt = np.linspace(2, 30, compute_points)
    sample_step = int(compute_points / subsample_to_points)
    trajectories_list = []

    evaluate_points = int(np.floor(np.sqrt(trajectories_to_sample)))
    x0s1d = np.linspace(0.1, 2, evaluate_points)
    try:
        with shelve.open("datasets") as db:
            trajectories = db[
                f"lotka_volterra_system_with_delay_trajectories_{evaluate_points}"]
    except KeyError:
        for x0 in tqdm(x0s1d):
            for y0 in x0s1d:
                yy = ddeint(model,
                            lambda t: np.array([x0, y0]),
                            tt,
                            fargs=(0.1, ))
                trajectories_list.append(yy)
        trajectories = np.stack(trajectories_list)
        with shelve.open("datasets") as db:
            db[f"lotka_volterra_system_with_delay_trajectories_{evaluate_points}"] = trajectories
    trajectoriesn = trajectories[:, ::sample_step]
    tt = tt[::sample_step]
    if double:
        trajectories = torch.from_numpy(trajectoriesn).to(device).double()
        t = torch.from_numpy(tt).to(device).double()
    else:
        trajectories = torch.from_numpy(trajectoriesn).to(
            torch.float32).to(device)
        t = torch.from_numpy(tt).to(torch.float32).to(device)
    return trajectories, t


def sine(device,
         double=False,
         trajectories_to_sample=100,
         t_nsamples=200,
         num_pi=4):
    t_nsamples_ref = 1000
    t_nsamples = int(t_nsamples_ref / 4 * num_pi)

    t_end = num_pi * np.pi
    t_begin = t_end / t_nsamples

    if double:
        ti = torch.linspace(t_begin, t_end, t_nsamples).to(device).double()
    else:
        ti = torch.linspace(t_begin, t_end, t_nsamples).to(device)

    def sampler(t, x0=0):
        # return torch.sin(t + x0)
        return torch.sin(t + x0) + torch.sin(
            2 * (t + x0)) + 0.5 * torch.sin(12 * (t + x0))

    x0s = torch.linspace(0, 16 * torch.pi, trajectories_to_sample)
    trajs = []
    for x0 in x0s:
        trajs.append(sampler(ti, x0))
    y = torch.stack(trajs)
    trajectories = y.view(trajectories_to_sample, -1, 1)
    return trajectories, ti


def time_sine(device,
              double=False,
              trajectories_to_sample=100,
              t_nsamples=201):
    """Generate sine data in time series fashion

    Args:
        device (_type_): _description_
        double (bool, optional): _description_. Defaults to False.
        trajectories_to_sample (int, optional): _description_. Defaults to 100.
        t_nsamples (int, optional): _description_. Defaults to 201.

    Returns:
        _type_: _description_
    """
    # (total_length - window_width) / stride - 1 = n_windows
    t_end = 20.0
    t_begin = t_end / t_nsamples
    window_width = t_nsamples - (trajectories_to_sample + 1)
    if double:
        ti = torch.linspace(t_begin, t_end, t_nsamples).to(device).double()
    else:
        ti = torch.linspace(t_begin, t_end, t_nsamples).to(device)

    def sampler(t):
        return torch.sin(t)
        return torch.sin(t) + torch.sin(2 * (t)) + 0.5 * torch.sin(11 * (t))

    traj = sampler(ti)
    # traj = torch.cat([traj.reshape(-1,1), ti.reshape(-1,1)], axis=-1)
    start = 0
    trajs, t = [], []
    for i in range(trajectories_to_sample):
        end = window_width + start
        trajs.append(traj[start:end].unsqueeze(-1))
        t.append(ti[start:end])
        start += 1
    if trajs[-1].shape != trajs[0].shape:
        trajs.pop()
        t.pop()
    y, t = torch.stack(trajs), torch.stack(t)
    if double:
        y = y.to(device).double()
        t = t.to(device).double()
    else:
        y = y.to(device)
        t = y.to(device)

    return y, t


# TODO: add transform or not
def solete(device,
           double=False,
           energy="solar",
           resolution="5min",
           transformed=False,
           window_width=24 * 12 * 2):
    df = pd.read_csv(f"datasets/SOLETE_new_{resolution}.csv",
                    index_col=0,
                    parse_dates=True,
                    infer_datetime_format=True)
    df = df.sort_index()
    if energy == "solar":
        features = ['TEMPERATURE[degC]', 'POA Irr[kW1m2]', 'P_Solar[kW]']
        if transformed:
            features[-1] += '-LNT'
        else:
            features[-1] += '[pu]'

        
    elif energy == "wind":
        if resolution == "5min" or resolution == "60min":
            features = ["WIND_SPEED[m1s]", "P_Gaia[kW]"]
            # features = ["u", "v", "P_Gaia[kW]"]
        else:
            features = ["WIND_SPEED[m1s]", "P_synthetic[kW]"]
            # features = ["u", "v", "P_synthetic[kW]"]
        if transformed:
            features[-1] += '-LNT'
        else:
            features[-1] += '[pu]'
        df = df['2018-08':'2019-05']
    print(features)
    

    df = df[features].values
    trajs = []
    start = 0
    while start + window_width < len(df):
        end = start + window_width
        trajs.append(df[start:end])
        start += 48
    if len(trajs[-1]) != len(trajs[-2]):
        trajs.pop()
    trajs = np.stack(trajs, axis=0)
    # t = torch.linspace(20 / window_width, 20, window_width)
    t = torch.arange(window_width) / 12
    if double:
        t = t.to(device).double()
        trajs = torch.from_numpy(trajs).to(device).double()
    else:
        t = t.to(device)
        trajs = torch.from_numpy(trajs).to(device)
    return trajs, t.unsqueeze(0)


#  Real-world dataset
def solete_wind(
        device,
        double=False,
        # features=[
        #     "TEMPERATURE[degC]", "WIND_SPEED[m1s]", "P_Sythetic(kW)"
        # ],
        # features=["TEMPERATURE[degC]"],
        features=["u", "v", "P_Gaia[kW]"],
        window_width=24 * 12 * 2):
    df = pd.read_csv("datasets/SOLETE_clean_5min.csv",
                     index_col=0,
                     parse_dates=True,
                     infer_datetime_format=True)
    df = df.sort_index()
    df = df['2018-08':'2019-05']
    df = df[features].values
    trajs = []
    start = 0
    while start + window_width < len(df):
        end = start + window_width
        trajs.append(df[start:end])
        start += window_width // 8
    if len(trajs[-1]) != len(trajs[-2]):
        trajs.pop()
    trajs = np.stack(trajs, axis=0)
    # t = torch.linspace(20 / window_width, 20, window_width)
    t = torch.arange(window_width) / 12
    if double:
        t = t.to(device).double()
        trajs = torch.from_numpy(trajs).to(device).double()
    else:
        t = t.to(device)
        trajs = torch.from_numpy(trajs).to(device)
    return trajs, t.unsqueeze(0)


#  Real-world dataset
def solete_solar(
        device,
        double=False,
        features=['TEMPERATURE[degC]', 'POA Irr[kW1m2]', 'P_Solar[kW]'],
        window_width=24 * 12 * 2):
    df = pd.read_csv("datasets/SOLETE_clean_5min.csv",
                     index_col=0,
                     parse_dates=True,
                     infer_datetime_format=True)
    df = df.sort_index()
    df = df[features].values
    trajs = []
    start = 0
    while start + window_width < len(df):
        end = start + window_width
        trajs.append(df[start:end])
        start += window_width // 8
    if len(trajs[-1]) != len(trajs[-2]):
        trajs.pop()
    trajs = np.stack(trajs, axis=0)
    # t = torch.linspace(20 / window_width, 20, window_width)
    t = torch.arange(window_width) / 12
    if double:
        t = t.to(device).double()
        trajs = torch.from_numpy(trajs).to(device).double()
    else:
        t = t.to(device)
        trajs = torch.from_numpy(trajs).to(device)
    return trajs, t.unsqueeze(0)


def guangdong(device, double, features=["value"], window_width=24 * 4 * 2):
    df = pd.read_csv("datasets/gd_wind_site.csv")
    df = df[features].values[:96 * 300]
    trajs = []
    # trajs, ts = [], []
    start = 0
    while start + window_width < len(df):
        end = start + window_width
        trajs.append(df[start:end])
        # ts.append(t[start:end])
        start += window_width // 2
    if len(trajs[-1]) != len(trajs[-2]):
        trajs.pop()
        # ts.pop()
    trajs = np.stack(trajs, axis=0)
    # ts = torch.stack(ts, axis=0)
    t = torch.arange(window_width) / 96
    # t = torch.linspace(window_width / window_width, 20, window_width)
    if double:
        # ts = ts.to(device).double()
        t = t.to(device).double()
        trajs = torch.from_numpy(trajs).to(device).double()
    else:
        # ts = ts.to(device)
        t = t.to(device)
        trajs = torch.from_numpy(trajs).to(device)
    # trajs = torch.cat([trajs, ts.unsqueeze(-1)], axis=-1)
    return trajs, t.unsqueeze(0)


def GEF(device, double, features=["temp", "load"], window_width=24 * 2):
    df = pd.read_csv("datasets/GEF_11_14.csv")
    df = df[features].values
    trajs = []
    # trajs, ts = [], []
    start = 0
    while start + window_width < len(df):
        end = start + window_width
        trajs.append(df[start:end])
        # ts.append(t[start:end])
        start += window_width // window_width
    if len(trajs[-1]) != len(trajs[-2]):
        trajs.pop()
        # ts.pop()
    trajs = np.stack(trajs, axis=0)
    # ts = torch.stack(ts, axis=0)
    t = torch.arange(window_width) / 48
    # t = torch.linspace(window_width / window_width, 20, window_width)
    if double:
        # ts = ts.to(device).double()
        t = t.to(device).double()
        trajs = torch.from_numpy(trajs).to(device).double()
    else:
        # ts = ts.to(device)
        t = t.to(device)
        trajs = torch.from_numpy(trajs).to(device)
    # trajs = torch.cat([trajs, ts.unsqueeze(-1)], axis=-1)
    return trajs, t.unsqueeze(0)


def generate_data_set(name,
                      device,
                      double=False,
                      batch_size=128,
                      extrap=0,
                      trajectories_to_sample=100,
                      percent_missing_at_random=0.0,
                      normalize=True,
                      test_set_out_of_distribution=True,
                      noise_std=None,
                      t_nsamples=200,
                      observe_stride=1,
                      predict_stride=1,
                      observe_steps=0.5,
                      seed=0,**kwargs):
    setup_seed(seed)
    if name == "lotka_volterra_system_with_delay":
        trajectories, t = lotka_volterra_system_with_delay(
            device, double, trajectories_to_sample, t_nsamples)
    elif name == "sine":
        trajectories, t = sine(device, double, trajectories_to_sample,
                               t_nsamples)
    elif name == "time_sine":
        trajectories, t = time_sine(device, double, trajectories_to_sample,
                                    t_nsamples)
    elif name == "solete_solar":
        trajectories, t = solete_solar(device, double)
    elif name == "solete_wind":
        trajectories, t = solete_wind(
            device,
            double,
        )
    elif name == "solete":
        print(kwargs)
        trajectories, t = solete(
            device,
            double,
            energy=kwargs.get("solete_energy"),
            resolution=kwargs.get("solete_resolution"),
            transformed=kwargs.get("solete_transformed"),
            window_width=kwargs.get("solete_window_width")
        )
    elif name == "gef":
        trajectories, t = GEF(
            device,
            double,
        )
    elif name == "guangdong":
        trajectories, t = guangdong(
            device,
            double,
        )

    else:
        raise ValueError("Unknown Dataset To Test")

    if not extrap:
        bool_mask = torch.FloatTensor(
            *trajectories.shape).uniform_() < (1.0 - percent_missing_at_random)
        if double:
            float_mask = (bool_mask).float().double().to(device)
        else:
            float_mask = (bool_mask).float().to(device)
        trajectories = float_mask * trajectories

    if noise_std:
        trajectories += torch.randn(trajectories.shape).to(device) * noise_std

    train_split = int(0.8 * trajectories.shape[0])
    test_split = int(0.9 * trajectories.shape[0])
    if test_set_out_of_distribution:
        train_trajectories = trajectories[:train_split, :, :]
        val_trajectories = trajectories[train_split:test_split, :, :]
        test_trajectories = trajectories[test_split:, :, :]
        if name.__contains__("time"):
            train_t = t[:train_split]
            val_t = t[train_split:test_split]
            test_t = t[test_split:]
        else:
            train_t = t
            val_t = t
            test_t = t

    else:
        traj_index = torch.randperm(trajectories.shape[0])
        train_trajectories = trajectories[traj_index[:train_split], :, :]
        val_trajectories = trajectories[
            traj_index[train_split:test_split], :, :]
        test_trajectories = trajectories[traj_index[test_split:], :, :]
        if name.__contains__("time"):
            train_t = t[traj_index[:train_split]]
            val_t = t[traj_index[train_split:test_split]]
            test_t = t[traj_index[test_split:]]
        else:
            train_t = t
            val_t = t
            test_t = t
    if normalize:
        len_train, len_val, len_test = len(train_trajectories), len(
            val_trajectories), len(test_trajectories)
        dim = trajectories.shape[2]
        train_mean = torch.reshape(train_trajectories, (-1, dim)).mean(0)
        train_std = torch.reshape(train_trajectories, (-1, dim)).std(0)
        train_trajectories = (torch.reshape(train_trajectories, (-1, dim)) -
                              train_mean) / train_std
        val_trajectories = (torch.reshape(val_trajectories,
                                          (-1, dim)) - train_mean) / train_std
        test_trajectories = (torch.reshape(test_trajectories,
                                           (-1, dim)) - train_mean) / train_std
        train_trajectories = train_trajectories.reshape((len_train, -1, dim))
        val_trajectories = val_trajectories.reshape((len_val, -1, dim))
        test_trajectories = test_trajectories.reshape((len_test, -1, dim))
    else:
        train_std = 1
        train_mean = 0

    rand_idx = torch.randperm(len(train_trajectories)).tolist()
    train_trajectories = train_trajectories[rand_idx]
    dltrain = DataLoader(
        train_trajectories,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: basic_collate_fn(
            batch,
            train_t,
            data_type="train",
            extrap=extrap,
            observe_stride=observe_stride,
            predict_stride=predict_stride,
            observe_steps=observe_steps),
    )
    dlval = DataLoader(
        val_trajectories,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: basic_collate_fn(
            batch,
            val_t,
            data_type="test",
            extrap=extrap,
            observe_stride=observe_stride,
            predict_stride=predict_stride,
            observe_steps=observe_steps),
    )
    dltest = DataLoader(
        test_trajectories,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: basic_collate_fn(
            batch,
            test_t,
            data_type="test",
            extrap=extrap,
            observe_stride=observe_stride,
            predict_stride=predict_stride,
            observe_steps=observe_steps),
    )

    b = next(iter(dltrain))
    input_dim = b["observed_data"].shape[-1]
    output_dim = b["data_to_predict"].shape[-1]
    input_timesteps = b["observed_data"].shape[1]
    output_timesteps = b["data_to_predict"].shape[1]
    return (input_dim, output_dim, dltrain, dlval, dltest, input_timesteps,
            output_timesteps, train_mean, train_std)


# (
#     input_dim,
#     output_dim,
#     dltrain,
#     dlval,
#     dltest,
#     input_timesteps,
#     output_timesteps,
# ) = generate_data_set("solete_solar",
#                       0,
#                       extrap=1,
#                       normalize=False,
#                       batch_size=3)

# for b in dltrain:
#     for k in b:
#         print(k)
#         print(b[k].shape)
#     break

# print(
#     input_dim,
#     output_dim,
#     dltrain,
#     dlval,
#     dltest,
#     input_timesteps,
#     output_timesteps,
# )