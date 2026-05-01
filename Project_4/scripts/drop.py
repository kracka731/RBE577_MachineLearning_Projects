import os
import h5py

input_root = "demonstrations"
output_file = "combined.hdf5"

def copy_item(name, obj, target_group):
    if isinstance(obj, h5py.Dataset):
        target_group.create_dataset(name, data=obj[()])
    elif isinstance(obj, h5py.Group):
        new_group = target_group.require_group(name)
        for key, val in obj.attrs.items():
            new_group.attrs[key] = val

def merge_hdf5_files(input_root, output_file):
    with h5py.File(output_file, "w") as out_f:
        for root, dirs, files in os.walk(input_root):
            if "demo.hdf5" in files:
                file_path = os.path.join(root, "demo.hdf5")
                
                # Use folder name as group name
                group_name = os.path.basename(root)
                print(f"Adding {file_path} as group '{group_name}'")

                with h5py.File(file_path, "r") as in_f:
                    out_group = out_f.create_group(group_name)
                    
                    # Copy all contents
                    def recursive_copy(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            in_f.copy(name, out_group)
                        elif isinstance(obj, h5py.Group):
                            out_group.require_group(name)

                    in_f.visititems(recursive_copy)

if __name__ == "__main__":
    merge_hdf5_files(input_root, output_file)