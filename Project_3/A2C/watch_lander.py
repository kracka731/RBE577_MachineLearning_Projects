import argparse

from train import load_config, run_lunar_lander
from eval import load_actor_from_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved LunarLander policy.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to the saved actor checkpoint. Defaults to config checkpoint_path.",
    )
    config = load_config()
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or config["checkpoint_path"]

    #run_lunar_lander(None, "random_lunar_lander_example1.mp4", config=config)
    actor = load_actor_from_checkpoint(config, checkpoint_path)
    run_lunar_lander(actor, "a2c_ex3.mp4", config=config)
