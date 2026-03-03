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
import numpy as np


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

    

def main():

    # Parse command line arguments
    args = parse_args()


    # Load configuration
    config = load_config(args.config)
    device = config.get_device()
    print_info(f"Using device: {device}")

    # Load data
    print_header("Loading Data")
    x_data, y_data = load_data(config)
    # print_info(f"Loaded data shapes: x={x_data.shape}, y={y_data.shape}")
    # print(f"row 1 of x: {x_data[1, :]}")
    x_rows, x_cols = x_data.shape
    y_rows, y_cols = y_data.shape
    # dataloader = prepare_dataloader(x_data, y_data, config)

    # Cut into training and validation sets.
    x_train, x_validate, x_test, y_train, y_validate, y_test = split_data(x_data, y_data, 0.7, 0.1)


    loaded_train_dataset = prepare_dataloader(x_train, y_train, config)
    loaded_valid_dataset = prepare_dataloader(x_validate, y_validate, config)
    loaded_test_dataset  = prepare_dataloader(x_test, y_test, config)


    # print("Loaded Train Dataset: ",loaded_train_dataset)

    # ToDO: Call Physics Push Planner
    planner = PushPlanner(config.model, config.physics_sampling)

    print_header("Starting Training")
    #pbar is progress bar: number of epochs
    pbar = tqdm(range(config.training["num_epochs"]), desc="Training Progress")

    # ToDO: Implement training loop
    accuracy_list = []
    loss_list = []
    for epoch in pbar:
        # print(f"Epoch {epoch+1}\n------------")

        # TODO: use planner.train_epoch() instead???
        for batch, (X, y) in enumerate(loaded_train_dataset):
            X, y = X.to(device), y.to(device)
            pred = planner.model(X)
            # print("pred: ",pred)
            loss = planner.loss(predictions=pred, targets=y)
            # print("loss: ",loss)

            #Deposit gradients of the loss w.r.t. each parameter
            #through backpropogation
            loss.backward()

            planner.optimize_push()

        # Test model with validation set to ensure learning
        test_loss, correct = planner.test(loaded_valid_dataset)

        accuracy_list.extend([100*correct,int(epoch+1)])
        loss_list.extend([test_loss,int(epoch+1)])

        # Test prediction
        # print_header("Testing Prediction")
    

    print_success("\nTraining completed!")
    # ToDo: Test the Model
    test_loss, accuracy = planner.test(loaded_test_dataset)
    print(f"overall avg loss is {test_loss} at {100*accuracy}% accuracy")

    # ToDO: Make Some predictions with model


if __name__ == "__main__":
    main()
