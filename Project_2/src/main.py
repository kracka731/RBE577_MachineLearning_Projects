import argparse
import torch
from lib.models import PushPlanner, NNModel
from helpers.utils import (
    load_data,
    prepare_dataloader,
    evaluate_planner,
    save_checkpoint,
    load_checkpoint,
)
from helpers.config import load_config
from tqdm import tqdm
from colorama import init, Fore, Style
import os
from lib.physics import PushPhysics
import numpy as np
import sys


# Initialize colorama
init()


def print_header(text: str):
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{text}{Style.RESET_ALL}")


def print_success(text: str):
    print(f"{Fore.GREEN}{text}{Style.RESET_ALL}")


def print_info(text: str):
    print(f"{Fore.YELLOW}{text}{Style.RESET_ALL}")


def print_error(text: str):
    print(f"{Fore.RED}{text}{Style.RESET_ALL}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train push planning model")
    parser.add_argument("--config", type=str, default=f"{os.path.dirname(os.path.abspath(__file__))}/config/custom.yaml", help="Path to config file")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training",
    )
    parser.add_argument("--model", type=str, default="nn", help="Choose model to run: {nn, physics, nn+physics}")
    return parser.parse_args()

def split_data(X, y, train_percent, valid_percent):
    # train_percent: amount of dataset to be training
    # valid_percent: amount of dataset to be validation
    # The rest of the datset will be for testing

    length = len(X)
    num_train = int(length*train_percent)
    num_valid = int(length*valid_percent)
    num_test = length - num_train - num_valid

    x_train = X[0:num_train,:]
    x_valid = X[num_train+1:num_valid+num_train,:]
    x_test = X[num_valid+num_train+1:num_valid+num_train+num_test, :]

    y_train = y[0:num_train,:]
    y_valid = y[num_train+1:num_valid+num_train,:]
    y_test = y[num_valid+num_train+1:num_valid+num_train+num_test, :]

    return x_train, x_valid, x_test, y_train, y_valid, y_test

def nn_train(config, planner, loaders):

    [train_dload, valid_dload, test_dload] = loaders

    print_header("Starting Neural Network Training")
    
    #pbar is progress bar: number of epochs
    pbar = tqdm(range(config.training["num_epochs"]), desc="Training Progress")

    accuracy_list = []
    loss_list = []
    avg_acc = 0.0
    avg_loss = 1.0

    for epoch in pbar:

        planner.train_epoch(train_dload, config.data["batch_size"])
        
        # Test model with validation set to have option to check learning
        # Doesn't make any specific changes as is due to no checkpoint saving
        test_loss, correct = planner.test(valid_dload)

        accuracy_list.extend([100*correct])
        loss_list.extend([test_loss])

        avg_acc = sum(accuracy_list) / len(accuracy_list)
        avg_loss = sum(loss_list) / len(loss_list)
        
        # Every 10 epochs, give an update of the average accuracy and loss
        if epoch % 10 == 0:
            print(f"At epoch {epoch}")
            print(f"Average Error: \n Accuracy: {(avg_acc):>0.1f}%, Avg loss: {avg_loss:>8f} \n")


    print_success("\nTraining completed!")

    print_header("Testing Prediction")
    test_loss, accuracy = planner.test(test_dload)
    print(f"overall avg loss is {test_loss} at {100*accuracy}% accuracy")

def physics_pred(config, planner, loaded_dataset):

    # ToDO: Call Physics Push Planner
    physics = PushPhysics.from_config(config.model['physics'])
    # model = PushPlanner() #TODO: add inputs to initialization

    # Test prediction
    print_header("Testing Prediction")
    
    # x, y = loaded_dataset.dataset[0:32]
    pred_list = []
    for batch, (X, y) in enumerate(loaded_dataset):
        # print(f"batch: {batch}")
        predictions = physics.compute_motion(X)
        # print(f"predictions: {predictions}")
        mse = torch.mean((predictions - y) ** 2)
        # print(f"MSE for one batch: {mse}")

        pred_list.extend(predictions)

    # print(len(pred_list))

    return pred_list

def main():

    # Parse command line arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    device = config.get_device()
    print_info(f"Using device: {device}")

    # Load data
    print_header("Loading Data")

    planner = PushPlanner(config.model, config.physics_sampling)

    if args.model == "nn":
        
        x_data, y_data = load_data(config)

        # Cut into training and validation sets.
        x_train, x_validate, x_test, y_train, y_validate, y_test = split_data(x_data, y_data, 0.7, 0.1)

        train_dataloader = prepare_dataloader(x_train, y_train, config)
        valid_dataloader = prepare_dataloader(x_validate, y_validate, config)
        test_dataloader  = prepare_dataloader(x_test, y_test, config)

        loaders = [train_dataloader, valid_dataloader, test_dataloader]

        nn_train(config, planner, loaders)

    elif args.model == "physics":
        x_data, y_data = load_data(config)

        loaded_dataset = prepare_dataloader(x_data, y_data, config)
        print("Loaded Dataset: ",loaded_dataset)

        phyics_predictions = physics_pred(config, planner, loaded_dataset)

    elif args.model == "nn+physics":
        # Perform same process as nn above, but use physics model as input to the nn
        pass

    else:
        sys.exit(f"(System Exit) Invalid model entered: {args.model}")

    # ToDO: Make Some predictions with model



if __name__ == "__main__":
    main()
