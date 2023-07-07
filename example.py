###########################
# Neural Laplace: Learning diverse classes of differential equations in the Laplace domain
# Author: Samuel Holt
###########################
import argparse
import logging
import pickle
from pathlib import Path
from time import strftime
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from dataset import generate_data_set

from model import GeneralNeuralLaplace
# from baseline_models.ode_models import GeneralLatentODE
# from baseline_models.original_latent_ode import GeneralLatentODEOfficial
from benchmarks import GeneralNODE, GeneralNeuralNetwork, GeneralPersistence
from utils import train_and_test, setup_seed, init_weights

datasets = [
    "lotka_volterra_system_with_delay",
    "sine",
    "time_sine",
    "solete_wind",
    "solete_solar",
    "solete",
    "gef",
    "guangdong",
]

file_name = Path(__file__).stem


def experiment_with_all_baselines(
    dataset,
    batch_size,
    extrapolate,
    epochs,
    seed,
    run_times,
    learning_rate,
    weight_decay,
    ode_solver_method,
    trajectories_to_sample,
    time_points_to_sample,
    observe_stride,
    predict_stride,
    observe_steps,
    noise_std,
    normalize_dataset,
    encode_obs_time,
    latent_dim,
    hidden_units,
    s_recon_terms,
    patience,
    device,
    use_sphere_projection,
    ilt_algorithm,
    solete_energy,
    solete_resolution,
    solete_transformed,
    solete_window_width,
):
    # Compares against all baselines, returning a pandas DataFrame of the test RMSE extrapolation error with std across input seed runs
    # Also saves out training meta-data in a ./results folder (such as training loss array and NFE array against the epochs array)
    # observe_samples = (time_points_to_sample // 2) // observe_step
    # logger.info(f"Experimentally observing {observe_samples} samples")

    df_list_baseline_results = []

    for seed in range(seed, seed + run_times):
        setup_seed(seed)
        Path("./results").mkdir(parents=True, exist_ok=True)
        path = f"./results/{path_run_name}-{seed}.pkl"

        (input_dim, output_dim, dltrain, dlval, dltest, input_timesteps,
         output_timesteps, train_mean, train_std) = generate_data_set(
             dataset,
             device,
             double=True,
             batch_size=batch_size,
             trajectories_to_sample=trajectories_to_sample,
             extrap=extrapolate,
             normalize=normalize_dataset,
             noise_std=noise_std,
             t_nsamples=time_points_to_sample,
             observe_stride=observe_stride,
             predict_stride=predict_stride,
             observe_steps=observe_steps,
             seed=seed,
             solete_energy=solete_energy,
             solete_resolution=solete_resolution,
             solete_transformed=solete_transformed,
             solete_window_width=solete_window_width)

        saved_dict = {}

        saved_dict["dataset"] = dataset
        saved_dict["trajectories_to_sample"] = trajectories_to_sample
        saved_dict["extrapolate"] = extrapolate
        saved_dict["normalize_dataset"] = normalize_dataset
        saved_dict["input_dim"] = input_dim
        saved_dict["output_dim"] = output_dim
        saved_dict["train_mean"] = train_mean
        saved_dict["train_std"] = train_std

        # Pre-save
        with open(path, "wb") as f:
            pickle.dump(saved_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        for model_name, system in [
            (
                "Neural Laplace",
                GeneralNeuralLaplace(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    latent_dim=latent_dim,
                    hidden_units=hidden_units,
                    s_recon_terms=s_recon_terms,
                    use_sphere_projection=use_sphere_projection,
                    ilt_algorithm=ilt_algorithm,
                    encode_obs_time=encode_obs_time,
                    device=device,
                    timesteps=input_timesteps).to(device),
            ),
                # (
                #     f"NODE ({ode_solver_method})",
                #     GeneralNODE(
                #         obs_dim=input_dim,
                #         nhidden=128,
                #         method=ode_solver_method,
                #         extrap=extrapolate,
                #     ).to(device),
                # ),
                # (
                #     f"ANODE ({ode_solver_method})",
                #     GeneralNODE(
                #         obs_dim=input_dim,
                #         nhidden=128,
                #         method=ode_solver_method,
                #         extrap=extrapolate,
                #         augment_dim=1,
                #     ).to(device),
                # ),
            # (
            #     "LSTM",
            #     GeneralNeuralNetwork(obs_dim=input_dim,
            #                          out_dim=output_dim,
            #                          out_timesteps=output_timesteps,
            #                          in_timesteps=input_timesteps,
            #                          nhidden=hidden_units,
            #                          method="lstm").to(device),
            # ),
            # (
            #     "MLP",
            #     GeneralNeuralNetwork(obs_dim=input_dim,
            #                          out_dim=output_dim,
            #                          out_timesteps=output_timesteps,
            #                          in_timesteps=input_timesteps,
            #                          nhidden=hidden_units,
            #                          method="mlp").to(device),
            # ),
            # (
            #     "Persistence",
            #     GeneralPersistence(out_timesteps=output_timesteps,
            #                        method="naive").to(device),
            # ),
        ]:
            try:
                logger.info(
                    f"Training & testing for : {model_name} \t | seed: {seed}")
                system.double()
                logger.info("num_params={}".format(
                    sum(p.numel() for p in system.model.parameters())))
                if model_name != "Persistence":
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
                test_rmse = mean_squared_error(
                    test_trajs[:,
                               -test_preds.shape[1]:, :].detach().cpu().numpy(
                               ).flatten(),
                    test_preds.detach().cpu().numpy().flatten())
                logger.info(f"Result: {model_name} - TEST RMSE: {test_rmse}")
                df_list_baseline_results.append({
                    'method': model_name,
                    'test_rmse': test_rmse,
                    'seed': seed
                })
                # train_preds, train_trajs = system.predict(dltrain)

                saved_dict[model_name] = {
                    "test rmse": test_rmse,
                    "seed": seed,
                    "model_state_dict": system.model.state_dict(),
                    "train_losses": train_losses.detach().cpu().numpy(),
                    "val_losses": val_losses.detach().cpu().numpy(),
                    "train_nfes": train_nfes.detach().cpu().numpy(),
                    # "train_epochs": train_epochs.detach().cpu().numpy(),
                    # "train_preds": train_preds.detach().cpu().numpy(),
                    # "train_trajs": train_trajs.detach().cpu().numpy(),
                    "val_preds": val_preds.detach().cpu().numpy(),
                    "val_trajs": val_trajs.detach().cpu().numpy(),
                    "test_preds": test_preds.detach().cpu().numpy(),
                    "test_trajs": test_trajs.detach().cpu().numpy(),
                }
                # Checkpoint
                with open(path, "wb") as f:
                    pickle.dump(saved_dict,
                                f,
                                protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                pass
                logger.error(e)
                logger.error(f"Error for model: {model_name}")
                raise e
        path = f"./results/{path_run_name}-{seed}.pkl"
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
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--interpolate", action="store_false")  # Default True
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_times", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--ode_solver_method", type=str, default="euler")
    parser.add_argument("--trajectories_to_sample", type=int, default=1000)
    parser.add_argument("--time_points_to_sample", type=int, default=200)
    parser.add_argument("--observe_stride", type=int, default=1)
    parser.add_argument("--predict_stride", type=int, default=1)
    parser.add_argument("--observe_steps", type=int, default=100)
    parser.add_argument("--noise_std", type=float, default=0.0)
    parser.add_argument("--normalize_dataset",
                        action="store_false")  # Default True
    parser.add_argument("--encode_obs_time",
                        action="store_true")  # Default False
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--hidden_units", type=int, default=42)
    parser.add_argument("--s_recon_terms", type=int,
                        default=33)  # (ANGLE_SAMPLES * 2 + 1)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_sphere_projection",
                        action="store_false")  # Default True
    parser.add_argument("--ilt_algorithm", type=str, default="fourier")
    parser.add_argument("--solete_energy", type=str, default="solar")
    parser.add_argument("--solete_resolution", type=str, default="5min")
    parser.add_argument("--solete_transformed",
                        action="store_true")  # Default False
    parser.add_argument("--solete_window_width", type=int,
                        default=24 * 12 * 2)  # Default False
    args = parser.parse_args()

    assert args.dataset in datasets
    device = torch.device(
        'cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    Path("./logs").mkdir(parents=True, exist_ok=True)
    path_run_name = "large-{}-{}-{}".format(f"observe_{args.observe_steps}",
                                      f"{args.dataset}-{args.solete_energy}",
                                      args.solete_transformed)
    # path_run_name = "{}-{}-{}".format(f"observe_{args.observe_steps}",
    #                                   strftime("%Y%m%d-%H%M"), args.dataset)
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
    test_rmse_df = experiment_with_all_baselines(
        dataset=args.dataset,
        batch_size=args.batch_size,
        extrapolate=args.interpolate,
        epochs=args.epochs,
        seed=args.seed,
        run_times=args.run_times,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        ode_solver_method=args.ode_solver_method,
        trajectories_to_sample=args.trajectories_to_sample,
        time_points_to_sample=args.time_points_to_sample,
        observe_stride=args.observe_stride,
        predict_stride=args.predict_stride,
        observe_steps=args.observe_steps,
        noise_std=args.noise_std,
        normalize_dataset=args.normalize_dataset,
        encode_obs_time=args.encode_obs_time,
        latent_dim=args.latent_dim,
        hidden_units=args.hidden_units,
        s_recon_terms=args.s_recon_terms,
        patience=args.patience,
        device=device,
        use_sphere_projection=args.use_sphere_projection,
        ilt_algorithm=args.ilt_algorithm,
        solete_energy=args.solete_energy,
        solete_resolution=args.solete_resolution,
        solete_transformed=args.solete_transformed,
        solete_window_width=args.solete_window_width,
    )
