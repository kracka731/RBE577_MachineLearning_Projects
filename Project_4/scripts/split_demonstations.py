"""split the existing demonstrations via masks that can then be used to select the number
of demonstrations used for training. how to use: 
ex: modify config file: 
"train": {
    "data": [
        {
            "path": "/work/demonstrations/merged_converted.hdf5",
            "filter_key": "train_5_demos"
        }
    ]
}
ex: modify command line args: 
python train.py --config training_configs/diffusion_policy.json --dataset.filter_key train_5_demos
"""

import h5py
import numpy as np

input_dataset = 'demonstrations/merged_converted.hdf5'
SPLITS = [5, 10, 20, 50]

def create_demo_splits(dataset_path, splits=SPLITS):
    with h5py.File(dataset_path, "a") as f:
        # Get all demo keys sorted consistently
        demo_keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        print(f"Total demos found: {len(demo_keys)}")
        
        f.require_group("mask")
        
        for n in splits:
            if n > len(demo_keys):
                print(f"Skipping split '{n}' — only {len(demo_keys)} demos available")
                continue
            
            selected = demo_keys[:n]
            encoded = np.array([k.encode("utf-8") for k in selected])
            
            key = f"train_{n}_demos"
            if key in f["mask"]:
                del f["mask"][key]
            
            f["mask"].create_dataset(key, data=encoded)
            print(f"Created mask '{key}' with demos: {selected}")

create_demo_splits(input_dataset, splits=SPLITS)

with h5py.File(input_dataset, "r") as f:
    for key in f["mask"].keys():
        demos = [d.decode("utf-8") for d in f["mask"][key][:]]
        print(f"{key}: {demos}")