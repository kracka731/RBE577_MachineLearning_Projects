"""The purpose of this script is to take all of the separate demonstrations and combine them into one file that
can be used to split and train a model."""
import h5py
import os

input_root = '/demonstrations'
output_file = 'merged_data.hdf5'

# Create a new HDF5 file (or open an existing one) to store the combined data
with h5py.File(output_file, 'w') as combined_h5:

    for root, subdirs, files in os.walk(input_root):
        if "demo.hdf5" in files:
            file_path = os.path.join(root, "demo.hdf5")

            # Open each input HDF5 file
            with h5py.File(file_path, 'r') as input_h5:
                
                # Iterate through datasets in the input file
                for dataset_name in input_h5:
                    # Copy each dataset to the combined HDF5 file
                    input_dataset = input_h5[dataset_name]
                    output_dataset = combined_h5.create_dataset(
                        f'{file_path}/{dataset_name}',  # Store datasets with a unique name
                        data=input_dataset[:]
                    )

# Close the combined HDF5 file (it's important to do this)
combined_h5.close()

print(f'Combined HDF5 file saved as: {output_file}')