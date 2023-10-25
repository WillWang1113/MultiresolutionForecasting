import logging
from copy import deepcopy
from time import time
from torch import nn

import numpy as np
import torch
import random

logger = logging.getLogger()


def train_and_test(
    system,
    dltrain,
    dlval,
    dltest,
    optim,
    device,
    scheduler=None,
    epochs=1000,
    patience=None,
    gradient_clip=1,
):
    if not patience:
        patience = epochs
    best_loss = float("inf")
    waiting = 0
    durations = []
    train_losses = []
    val_losses = []
    train_nfes = []
    epoch_num = []
    for epoch in range(epochs):
        iteration = 0
        epoch_train_loss_it_cum = 0
        epoch_nfe_cum = 0

        system.model.train()
        start_time = time()

        for batch in dltrain:
            optim.zero_grad()
            train_loss = system.training_step(batch)
            train_loss.backward()
            # Optional gradient clipping
            torch.nn.utils.clip_grad_norm_(system.model.parameters(),
                                           gradient_clip)
            optim.step()
            epoch_train_loss_it_cum += train_loss.item()

            iteration += 1

            epoch_train_loss_it_cum += train_loss.item()

            iteration += 1
        epoch_train_loss = epoch_train_loss_it_cum / iteration
        epoch_nfes = epoch_nfe_cum / iteration

        epoch_duration = time() - start_time
        durations.append(epoch_duration)
        train_losses.append(epoch_train_loss)
        train_nfes.append(epoch_nfes)
        epoch_num.append(epoch)

        # Validation step
        system.model.eval()
        val_loss, val_mse = system.validation_step(dlval)
        val_loss, val_mse = val_loss.item(), val_mse.item()
        val_losses.append(val_loss)
        logger.info("[epoch={}] | train_loss={:.5f}\t| val_loss={:.5f}".format(
            epoch, np.sqrt(epoch_train_loss), np.sqrt(val_loss)))

        # Learning rate scheduler
        if scheduler:
            scheduler.step()

        # Early stopping procedure
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = deepcopy(system.model.state_dict())
            best_epoch = epoch
            waiting = 0
        elif waiting > patience:
            break
        else:
            print("val up!", waiting, "/", patience)
            waiting += 1

    logger.info(f"epoch_duration_mean={np.mean(durations):.5f}")

    # Load best model
    system.model.load_state_dict(best_model)

    return (
        # test_rmse,
        torch.Tensor(train_losses).to(device),
        torch.Tensor(val_losses).to(device),
        torch.Tensor(train_nfes).to(device),
        torch.Tensor(epoch_num).to(device),
    )


def setup_seed(seed: int = 9):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_weights(m, seed):
    for m in m.modules():
        setup_seed(seed)
        if isinstance(m, nn.BatchNorm1d):
            m.reset_parameters()

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.01)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            m.reset_parameters()
        elif isinstance(m, (nn.GRU, nn.LSTM, nn.RNN)):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0.01)
