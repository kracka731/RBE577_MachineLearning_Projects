import argparse
import torch
from lib.models import PushPlanner, NNModel, NNPhysicsModel
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
from main import parse_args, split_data
import matplotlib.pyplot as plt


def get_loaders(x_data, y_data, config):
    # Cut into training and validation sets.
    x_train, x_validate, x_test, y_train, y_validate, y_test = split_data(x_data, y_data, 0.7, 0.1)

    train_dataloader = prepare_dataloader(x_train, y_train, config)
    valid_dataloader = prepare_dataloader(x_validate, y_validate, config)
    test_dataloader  = prepare_dataloader(x_test, y_test, config)
    loaders = [train_dataloader, valid_dataloader, test_dataloader]
    return loaders

def track_train(config, planner, loaders):
    [train_dload, valid_dload, test_dload] = loaders
    
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

    test_loss, accuracy = planner.test(test_dload)
    return loss_list, accuracy_list, test_loss, accuracy

def test(planner, loaders):
    [train_dload, valid_dload, test_dload] = loaders
    test_loss, accuracy = planner.test(test_dload)

def plot(num_epochs, phys_loss, nn_loss, nn_accuracy, nnphys_loss, nnphys_accuracy):
    fig, axs = plt.subplots(1, 2)
    pos = [x for x in range(num_epochs)]
    phys_loss = [phys_loss for i in range(num_epochs)]
    axs[0].plot(pos,  phys_loss, label='Physics')
    axs[0].plot(pos,  nn_loss, label='NNModel')
    axs[0].plot(pos,  nnphys_loss, label='NNPhysicsModel')
    axs[0].set_title('Loss over course of Training')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(pos,  nn_accuracy, label='NNModel')
    axs[1].plot(pos,  nnphys_accuracy, label='NNModel')
    axs[1].set_title('Accuracy over course of Training')
    axs[1].set_ylabel('% Accuracy')
    axs[1].legend()
    
    return fig, axs

def main():
    """Initialize all 3 models, train, graph loss, compare"""
    # Parse command line arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    device = config.get_device()

    # Intialize pure physics
    physics = PushPhysics.from_config(config.model['physics'])
    nn = PushPlanner(config.model, config.physics_sampling, override_model="nn")
    nn_phys = PushPlanner(config.model, config.physics_sampling, override_model="nn+physics")

    # Load data
    x_data, y_data = load_data(config)
    nn_loaders = get_loaders(x_data, y_data, config)
    phys_pred = physics.compute_motion(torch.from_numpy(x_data))
    combined_x = np.hstack([x_data, phys_pred.cpu().numpy()])
    nnphys_loaders = get_loaders(combined_x, y_data, config)

    # Training  
    print("Training pure neural network")
    nn_loss, nn_accuracy, nn_t_loss, nn_t_accuracy = track_train(config, nn, nn_loaders)

    print("Training combined physics + neural network")
    nnphys_loss, nnphys_accuracy, nnph_t_loss, nnph_t_accuracy = track_train(config, nn_phys, nnphys_loaders)
    print("Training complete")

    phys_loss = torch.mean((phys_pred - torch.from_numpy(y_data).to(device))**2)

    print(f"Test loss for Physis: {phys_loss.cpu()} loss and accuracy for NN: {nn_t_loss} {100*nn_t_accuracy}, and NNPhysics {nnph_t_loss} {100*nnph_t_accuracy}")

    # Graph predictions vs ground truth 

    # Graph loss at each epoch
    num_epochs = config.training["num_epochs"]
    fig, axs = plot(num_epochs, phys_loss.cpu(), nn_loss, nn_accuracy, nnphys_loss, nnphys_accuracy)
    plt.show()



if __name__ == "__main__":
    main()
