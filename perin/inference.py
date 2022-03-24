#!/usr/bin/env python3
# coding=utf-8

import argparse
import torch

from model.model import Model
from data.dataset import Dataset
from utility.initialize import initialize
from config.params import Params
from utility.predict import predict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_directory", type=str, default="/home/samueld/mrp_update/mrp")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint)
    args = Params().load_state_dict(checkpoint["args"]).init_data_paths(args.data_directory)
    args.log_wandb = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    directory = initialize(args, create_directory=True, init_wandb=False, directory_prefix="inference_")

    dataset = Dataset(args)
    dataset.load_datasets(args)

    model = Model(dataset, args).to(device)
    model.load_state_dict(checkpoint["model"])

    print("inference of validation data", flush=True)
    predict(model, dataset.val, args.validation_data, args, directory, 0, run_evaluation=True, epoch=0)

    print("inference of test data", flush=True)
    predict(model, dataset.test, args.test_data, args, f"{directory}/test_predictions", 0)
