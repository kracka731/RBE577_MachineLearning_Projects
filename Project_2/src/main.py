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


    pass
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



    # print("Loaded Train Dataset: ",loaded_train_dataset)

    # ToDO: Call Physics Push Planner
    # model = PushPlanner() #TODO: add inputs to initialization

    print_header("Starting Training")
    #pbar is progress bar: number of epochs
    pbar = tqdm(range(config.training["num_epochs"]), desc="Training Progress")

    #TODO Figure out how to source these parameters instead from the config file
    nn_model = NNModel(input_dim=x_cols,output_dim=y_cols,hidden_dims=[32, 64, 128, 128, 64, 32])

    optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)

    # ToDO: Implement training loop
    for epoch in pbar:
        # print(f"Epoch {epoch+1}\n------------")

        for batch, (X, y) in enumerate(loaded_train_dataset):
            pred = nn_model(X)
            # print("pred: ",pred)
            loss = nn_model.loss(predictions=pred, targets=y)
            # print("loss: ",loss)

            #Deposit gradients of the loss w.r.t. each parameter
            #through backpropogation
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        # Test prediction
    print_header("Testing Prediction")

    nn_model.eval()
    accuracy_list = []
    loss_list = []
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in loaded_valid_dataset:
            pred = nn_model(X)

            test_loss += nn_model.loss(predictions=pred, targets=y).item()

            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    test_loss /= len(loaded_valid_dataset)
    correct /= len(x_validate)

    accuracy_list.extend([100*correct,int(epoch+1)])
    loss_list.extend([test_loss,int(epoch+1)])

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    

    print_success("\nTraining completed!")


    # ToDo: Test the Model

    # ToDO: Make Some predictions with model
    # Make predictions with the model
    with torch.no_grad():
        # Move tensors to appropriate device
        # Do forward and calculate error
        pass


if __name__ == "__main__":
    main()
