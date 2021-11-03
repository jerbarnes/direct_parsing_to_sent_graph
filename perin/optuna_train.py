#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import optuna

import argparse
import os
import datetime

import torch
import torch.utils.data
from transformers import AutoConfig

from model.model import Model
from data.dataset import Dataset
from utility.initialize import initialize
from utility.log import Log
from utility.schedule.multi_scheduler import multi_scheduler_wrapper
from utility.autoclip import AutoClip
from data.batch import Batch
from config.params import Params
from utility.predict import predict
from utility.adamw import AdamW
from utility.loss_weight_learner import LossWeightLearner

from transformers import logging
logging.set_verbosity_error()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="path to config file")
    parser.add_argument("--data_directory", type=str, default="/cluster/projects/nn9851k/davisamu/sent_graph_followup/data")
    parser.add_argument("--dist_backend", default="nccl", type=str)
    parser.add_argument("--dist_url", default="localhost", type=str)
    parser.add_argument("--home_directory", type=str, default="/cluster/projects/nn9851k/davisamu/sent_graph_followup/data")
    parser.add_argument("--name", default="norec", type=str, help="name of this run.")
    parser.add_argument("--save_checkpoints", dest="save_checkpoints", action="store_true", default=False)
    parser.add_argument("--log_wandb", dest="log_wandb", action="store_true", default=False)
    parser.add_argument("--validate_each", type=int, default=10, help="Validate every ${N}th epoch.")
    parser.add_argument("--wandb_log_mode", type=str, default=None, help="How to log the model weights, supported values: {'all', 'gradients', 'parameters', None}")
    parser.add_argument("--workers", type=int, default=1, help="number of CPU workers per GPU.")
    args = parser.parse_args()

    params = Params()
    params.load(args)
    params.load_state_dict(vars(args))

    encoder_config = AutoConfig.from_pretrained(params.encoder)
    params.hidden_size = encoder_config.hidden_size
    params.n_encoder_layers = encoder_config.num_hidden_layers

    return params


def run(args):
    initialize(args, init_wandb=False)

    dataset = Dataset(args, verbose=False)
    dataset.load_datasets(args)

    model = Model(dataset, args)
    optimizer = torch.optim.AdamW(model.get_params_for_optimizer(args), betas=(0.9, args.beta_2))
    scheduler = multi_scheduler_wrapper(optimizer, args, len(dataset.train))
    # autoclip = AutoClip([p for name, p in model.named_parameters() if "loss_weights" not in name])
    if args.balance_loss_weights:
        loss_weight_learner = LossWeightLearner(args, model, 1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    errors = []

    for epoch in range(50):

        #
        # TRAINING
        #

        model.train()
        model.zero_grad()

        i = 0

        for batch in dataset.train:
            batch = Batch.to(batch, device)
            total_loss, losses, stats = model(batch)

            if args.balance_loss_weights:
                loss_weight_learner.compute_grad(losses, epoch)
            total_loss.backward()

            if (i + 1) % args.accumulation_steps == 0:
                #_ = autoclip()

                if args.balance_loss_weights:
                    loss_weight_learner.step(epoch)
                scheduler(epoch)
                optimizer.step()
                model.zero_grad()

            del total_loss, losses
            i += 1

        if epoch >= 45:
            f1 = predict(model, dataset.val, args.validation_data, args.raw_validation_data, args, None, directory, 0, mode="validation", epoch=epoch)
            errors.append(f1)

    return sum(errors) / len(errors)


# Define an objective function to be minimized.
class Objective:
    def __init__(self, args):
        self.args = args

    def __call__(self, trial):
        args.query_length = 2
        args.balance_loss_weights = False
        args.grad_norm_lr = 3.0e-4
        args.decoder_learning_rate = trial.suggest_loguniform('decoder_learning_rate', 3.0e-5, 6.0e-4)
        args.encoder_learning_rate = trial.suggest_loguniform('encoder_learning_rate', 3.0e-6, 3.0e-5)            # initial encoder learning rate
        args.encoder_weight_decay = 0.1
        args.label_smoothing = 0.0
        args.encoder_delay_steps = 500
        args.warmup_steps = 1000
        args.char_embedding = True
        args.dropout_word = 0.1
        args.focal = True
        args.hidden_size_edge_presence = 256
        args.hidden_size_anchor = 256
        self.args.dropout_anchor = trial.suggest_float('dropout_anchor', 0.25, 0.75)
        self.args.dropout_edge_presence = trial.suggest_float('dropout_edge_presence', 0.5, 0.95)
        self.args.dropout_label = trial.suggest_float('dropout_label', 0.5, 0.95)
        args.batch_size = 16
        args.beta_2 = 0.98
        args.layerwise_lr_decay = 0.9

        f1_score = run(self.args)

        return f1_score  # An objective value linked with the Trial object.


def main_worker(directory, args):
    study = optuna.create_study(direction="maximize")  # Create a new study.
    study.optimize(Objective(args), n_trials=100, gc_after_trial=True, show_progress_bar=True)  # Invoke optimization of the objective function.


if __name__ == "__main__":
    args = parse_arguments()

    timestamp = f"{datetime.datetime.today():%m-%d-%y_%H-%M-%S}"
    directory = f"/cluster/home/davisamu/home/sent_graph_followup/perin/outputs/test_{timestamp}"
    os.mkdir(directory)
    os.mkdir(f"{directory}/test_predictions")

    main_worker(directory, args)
