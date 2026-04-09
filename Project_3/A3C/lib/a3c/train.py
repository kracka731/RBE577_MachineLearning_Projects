import os

import torch
import torch.multiprocessing as mp
import torch.optim as optim

from helpers.config import load_config
from helpers.logger import A3CLogger
from helpers.utils import get_kuka_action_dim, get_network_input_shape
from lib.a3c.agent import worker_process
from lib.a3c.model import ActorCritic
from lib.a3c.shared_optim import SharedAdam


def build_global_model(config, device):
    """Create the shared global actor-critic model."""
    # TODO: Build the shared global actor-critic network
    state_dim = get_network_input_shape(config)
    action_dim = get_kuka_action_dim(config)
    net = config['network']

    # Hint: The global model should use the same architecture as each worker's
    # local model, but this instance must also be prepared for parameter sharing.
    model = ActorCritic(state_dim, action_dim, 
                        net["shared_layers"], net["critic_hidden_layers"], 
                        net["actor_hidden_layers"], init_type=net["init_type"]) 

    # TODO: Move the global model parameters into shared memory
    model.to(device)
    model.share_memory()  # Replace with your implementation

    return model


def save_final_checkpoint(global_net, optimizer, config):
    """Save the final shared model state."""
    model_path = os.path.join(
        config["logging"]["model_dir"], "a3c_kuka_model_final.pth"
    )
    torch.save(
        {
            "model_state_dict": global_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "episode": config["hyperparameters"]["max_episodes"],
        },
        model_path,
    )
    return model_path


def train_a3c():
    config = load_config()
    device = torch.device(config["device"])
    logger = A3CLogger(config)

    # TODO: Set up PyTorch multiprocessing before workers are launched
    # Hint: Use the start method expected by the shared-memory A3C setup.
    mp.set_start_method('spawn', force=True)

    # TODO: Create the shared training objects used by all workers

    # interval statistics for logging.
    global_net = build_global_model(config, device) 
    # optimizer = SharedAdam(global_net.parameters(), lr=config['hyperparameters']['lr']) 
    optimizer = optim.Adam(global_net.parameters(), lr=config['hyperparameters']['lr'])  

    global_ep = mp.Value('i', 0)  # shared data. signed integer with init value 0. when using, do lock manually!! 
    lock = mp.Lock()  # used to ensure only 1 process can access/modify shared resources at a time
    manager = mp.Manager()  # use for sharing complex data. handles all synchronization, so you don't have to use lock manually
    shared_stats = None  # FIXME: idk what this is supposed to be used for

    os.makedirs(config["logging"]["model_dir"], exist_ok=True)

    logger.info("Starting A3C training for Kuka pick and place task...")
    logger.info(
        f"Using {config['hyperparameters']['num_workers']} workers on {device}"
    )

    processes = []
    for worker_id in range(config["hyperparameters"]["num_workers"]):
        # TODO: Launch one worker process for each worker id
        args = tuple([worker_id,
            global_net,
            optimizer,
            global_ep,
            config["hyperparameters"]['max_episodes'],
            lock,
            config,
            device,
            shared_stats,
            None])
        p = mp.Process(target=worker_process, args=args)

        # TODO: Start the worker and keep track of the process handle
        p.start()
        processes.append(p)

    # TODO: Wait for all worker processes to finish
    for p in processes:
        p.join()

    # TODO: Save the final checkpoint and clean up shared manager resources
    model_path = None  # Replace with your implementation
    logger.info(f"Final model saved to {model_path}. Training complete!")
    logger.close()
