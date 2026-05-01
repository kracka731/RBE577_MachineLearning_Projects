import robomimic
from robomimic.config import config_factory
import os

demo_fp = 'demonstrations/'


def make_config(algo_name="bc"):

    # create BC config
    config = config_factory(algo_name=algo_name)

    combined_data = []
    for root, dirs, files in os.walk(demo_fp):
        for file in files:
            fp = os.path.join(root, file)
            print(f"Added demonstration from: {fp}")
            data = {
                    "path": fp,
                    "weight": 1.0,                    
                }
            combined_data.append(data)

    # configure datasets
    config.train.data = combined_data

    # normalize weights by dataset size for balanced sampling
    config.train.normalize_weights_by_ds_size = True

    # other training settings...
    config.train.batch_size = 100
    config.train.num_epochs = 1000

    return config

make_config("bc")
