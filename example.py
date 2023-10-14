###########################
# Neural Laplace: Learning diverse classes of differential equations in the Laplace domain
# Author: Samuel Holt
###########################
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import logging
import pickle
from pathlib import Path
from time import strftime
import pandas as pd
import numpy as np
import torch

# torch.cuda.empty_cache()
# torch.cuda.memory_summary(device=None, abbreviated=False)
from sklearn.metrics import mean_squared_error
from dataset import generate_data_set

# from model import GeneralNeuralLaplace
from model import GeneralHNL
# from baseline_models.ode_models import GeneralLatentODE
# from baseline_models.original_latent_ode import GeneralLatentODEOfficial
from utils import train_and_test, setup_seed, init_weights

datasets = ["sine", "solete", "nrel", "mfred", "australia"]

file_name = Path(__file__).stem


def experiment_with_all_baselines(
        dataset, double, batch_size, extrapolate, epochs, seed, run_times,
        learning_rate, weight_decay, trajectories_to_sample,
        time_points_to_sample, observe_stride, predict_stride,
        avail_fcst_stride, observe_steps, noise_std, normalize_dataset,
        encode_obs_time, encoder, latent_dim, hidden_units,
        include_s_recon_terms, pass_raw, s_recon_terms_list, avg_terms_list,
        patience, device, use_sphere_projection, ilt_algorithm, solete_energy,
        solete_resolution, transformed, window_width, shared_encoder,
        add_external_feature, **kwargs):
    # Compares against all baselines, returning a pandas DataFrame of the test RMSE extrapolation error with std across input seed runs
    # Also saves out training meta-data in a ./results folder (such as training loss array and NFE array against the epochs array)
    # observe_samples = (time_points_to_sample // 2) // observe_step
    # logger.info(f"Experimentally observing {observe_samples} samples")

    df_list_baseline_results = []

    for seed in range(seed, seed + run_times):
        setup_seed(seed)
        Path(f"./results/{dataset}").mkdir(parents=True, exist_ok=True)
        path = f"./results/{dataset}/{path_run_name}-{seed}.pkl"

        (input_dim, output_dim, sample_rate, t, dltrain, dlval, dltest,
         input_timesteps, output_timesteps, train_mean, train_std,
         feature) = generate_data_set(
             name=dataset,
             device=device,
             double=double,
             batch_size=batch_size,
             trajectories_to_sample=trajectories_to_sample,
             extrap=extrapolate,
             normalize=normalize_dataset,
             noise_std=noise_std,
             t_nsamples=time_points_to_sample,
             observe_stride=observe_stride,
             predict_stride=predict_stride,
             avail_fcst_stride=avail_fcst_stride,
             observe_steps=observe_steps,
             seed=seed,
             add_external_feature=add_external_feature,
             solete_energy=solete_energy,
             solete_resolution=solete_resolution,
             transformed=transformed,
             avg_terms=1,
             window_width=window_width)
        logger.info(f"input steps:\t {input_timesteps}")
        logger.info(f"output steps:\t {output_timesteps}")
        if avg_terms_list is not None:
            logger.info(f"Calculate s_recon_terms")
            avg_terms_list.sort(reverse=True)
            s_recon_terms_list = []
            for i, avg_terms in enumerate(avg_terms_list):
                desired_f = sample_rate / avg_terms
                T = 2 * t.cpu().numpy().max()
                s_terms = int(T * desired_f)
                if i == 0:
                    s_recon_terms_list.append(s_terms)
                else:
                    s_recon_terms_list.append(s_terms -
                                              sum(s_recon_terms_list))
            for i in range(len(s_recon_terms_list)):
                if s_recon_terms_list[i] % 2 == 0:
                    s_recon_terms_list[i] += 1
            print(s_recon_terms_list)
        elif s_recon_terms_list is not None:
            s_recon_terms_list = s_recon_terms_list
        elif s_recon_terms_list is None and avg_terms_list is None:
            raise ValueError(
                "please enter s_recon_terms_list or avg_terms_list.")
        
        logger.info(f"s_recon_terms list:\t {s_recon_terms_list}")
        saved_dict = {}
        saved_dict["dataset"] = dataset
        saved_dict["train_mean"] = train_mean
        saved_dict["train_std"] = train_std
        saved_dict["sample_rate"] = sample_rate
        saved_dict["add_external_feature"] = add_external_feature

        saved_dict["model_hyperparams"] = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "latent_dim": latent_dim,
            "hidden_units": hidden_units,
            "s_recon_terms": s_recon_terms_list,
            "use_sphere_projection": use_sphere_projection,
            "include_s_recon_terms": include_s_recon_terms,
            "ilt_algorithm": ilt_algorithm,
            "encode_obs_time": encode_obs_time,
            "encoder": encoder,
            "device": device,
            "input_timesteps": input_timesteps,
            "output_timesteps": output_timesteps,
            "avg_terms_list": avg_terms_list,
            "pass_raw": pass_raw,
            "shared_encoder": shared_encoder
        }
        saved_dict["data_params"] = {
            "name": dataset,
            "device": device,
            "double": double,
            "batch_size": batch_size,
            "trajectories_to_sample": trajectories_to_sample,
            "extrap": extrapolate,
            "normalize": normalize_dataset,
            "noise_std": noise_std,
            "t_nsamples": time_points_to_sample,
            "observe_stride": observe_stride,
            "predict_stride": predict_stride,
            "avail_fcst_stride": avail_fcst_stride,
            "observe_steps": observe_steps,
            "seed": seed,
            "add_external_feature": add_external_feature,
            "solete_energy": solete_energy,
            "solete_resolution": solete_resolution,
            "transformed": transformed,
            "avg_terms": 1,
            "window_width": window_width
        }

        # Pre-save
        with open(path, "wb") as f:
            pickle.dump(saved_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        model_name, system = "Hierarchical NL", GeneralHNL(
            input_dim=input_dim,
            output_dim=output_dim,
            latent_dim=latent_dim,
            hidden_units=hidden_units,
            s_recon_terms=s_recon_terms_list,
            use_sphere_projection=use_sphere_projection,
            include_s_recon_terms=include_s_recon_terms,
            ilt_algorithm=ilt_algorithm,
            encode_obs_time=encode_obs_time,
            encoder=encoder,
            device=device,
            input_timesteps=input_timesteps,
            output_timesteps=output_timesteps,
            avg_terms_list=avg_terms_list,
            pass_raw=pass_raw,
            shared_encoder=shared_encoder).to(device),

        try:
            logger.info(
                f"Training & testing for : {model_name} \t | seed: {seed}")

            if double:
                system.double()
            else:
                system.float()
            logger.info("num_params={}".format(
                sum(p.numel() for p in system.model.parameters())))
            init_weights(system.model, seed)
            optimizer = torch.optim.Adam(system.model.parameters(),
                                         lr=learning_rate,
                                         weight_decay=weight_decay)
            lr_scheduler_step = 20
            lr_decay = 0.5
            scheduler = None
            train_losses, val_losses, train_nfes, _ = train_and_test(
                system,
                dltrain,
                dlval,
                dltest,
                optimizer,
                device,
                scheduler,
                epochs=epochs,
                patience=patience,
            )
            val_preds, val_trajs = system.predict(dlval)
            test_preds, test_trajs = system.predict(dltest)

            assert test_trajs.shape == test_preds[-1].shape
            test_rmse = mean_squared_error(
                test_trajs.detach().cpu().numpy().flatten(),
                test_preds[-1].detach().cpu().numpy().flatten())
            logger.info(f"Result: {model_name} - TEST RMSE: {test_rmse}")
            df_list_baseline_results.append({
                'method': model_name,
                'test_rmse': test_rmse,
                'seed': seed
            })
            # train_preds, train_trajs = system.predict(dltrain)

            saved_dict[model_name] = {
                "seed": seed,
                "model_state_dict": system.model.state_dict(),
                "train_losses": train_losses.detach().cpu().numpy(),
                "val_losses": val_losses.detach().cpu().numpy(),
                # "train_nfes": train_nfes.detach().cpu().numpy(),
                # "train_epochs": train_epochs.detach().cpu().numpy(),
                # "train_preds": train_preds.detach().cpu().numpy(),
                # "train_trajs": train_trajs.detach().cpu().numpy(),
                # "val_preds": val_preds,
                # "val_trajs": val_trajs.detach().cpu().numpy(),
                "test_preds": test_preds,
                "test_trajs": test_trajs.detach().cpu().numpy(),
            }
            # Checkpoint
            with open(path, "wb") as f:
                pickle.dump(saved_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.error(e)
            logger.error(f"Error for model: {model_name}")
            raise e
        path = f"./results/{dataset}/{path_run_name}-{seed}.pkl"
        with open(path, "wb") as f:
            pickle.dump(saved_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Process results for experiment
    df_results = pd.DataFrame(df_list_baseline_results)
    test_rmse_df = df_results.groupby('method').agg(['mean',
                                                     'std'])['test_rmse']
    logger.info("Test RMSE of experiment")
    logger.info(test_rmse_df)
    return test_rmse_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Run all baselines for an experiment (including Neural Laplace)")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="sine",
        help=f"Available datasets: {datasets}",
    )
    parser.add_argument("--double", action="store_true")  # Default False
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--extrapolate", action="store_false")  # Default True
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_times", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--trajectories_to_sample", type=int, default=1000)
    parser.add_argument("--time_points_to_sample", type=int, default=200)
    parser.add_argument("--observe_stride", type=int, default=1)
    parser.add_argument("--predict_stride", type=int, default=1)
    parser.add_argument("--avail_fcst_stride", type=int, default=12)
    parser.add_argument("--observe_steps", type=int, default=100)
    parser.add_argument("--noise_std", type=float, default=0.0)
    parser.add_argument("--normalize_dataset",
                        action="store_false")  # Default True
    parser.add_argument("--encode_obs_time",
                        action="store_true")  # Default False
    parser.add_argument("--encoder", type=str, default="dnn")
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--hidden_units", type=int, default=42)
    parser.add_argument("--include_s_recon_terms",
                        action="store_false")  # Default True
    parser.add_argument("--pass_raw", action="store_true")  # Default False
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_sphere_projection",
                        action="store_false")  # Default True
    parser.add_argument("--ilt_algorithm", type=str, default="fourier")
    parser.add_argument("--solete_energy", type=str, default="solar")
    parser.add_argument("--solete_resolution", type=str, default="5min")
    parser.add_argument("--transformed", action="store_true")  # Default False
    parser.add_argument("--window_width", type=int,
                        default=24 * 12 * 2)  # Default False
    parser.add_argument('--s_recon_terms_list',
                        nargs='+',
                        type=int,
                        default=[25, 57, 41])
    parser.add_argument('--avg_terms_list', nargs='+', type=int, default=None)
    parser.add_argument('--shared_encoder',
                        action="store_true")  # Default False
    parser.add_argument('--add_external_feature',
                        action="store_true")  # Default False
    args = parser.parse_args()

    assert args.dataset in datasets
    device = torch.device(
        'cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    Path("./logs").mkdir(parents=True, exist_ok=True)
    # path_run_name = "nwf_dnnFE_addloss_{}-{}-{}-{}-{}-{}-{}-{}-{}".format(
    #     f"{args.dataset}", f"obs_{args.observe_steps}", "hnl", args.encoder,
    #     f"sterms_{args.avg_terms_list}", f"latent_{args.latent_dim}",
    #     f"shared_encoder{args.shared_encoder}", f"pass_raw{args.pass_raw}",
    #     f"wd_{args.weight_decay}")
    path_run_name = "{}-{}".format(file_name, strftime("%Y%m%d-%H%M%S"))

    logging.basicConfig(
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(f"logs/{path_run_name}_log.txt"),
            logging.StreamHandler()
        ],
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger()

    logger.info(f"Using {device} device")
    test_rmse_df = experiment_with_all_baselines(device=device, **vars(args))
