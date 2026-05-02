# # Source - https://stackoverflow.com/a/41586571
# # Posted by Martin Thoma, modified by community. See post 'Timeline' for change history
# # Retrieved 2026-05-01, License - CC BY-SA 4.0

# import h5py
# filename = "../demonstrations/1776983448_1210191/demo.hdf5"

# with h5py.File(filename, "r") as f:
#     # Print all root level object names (aka keys) 
#     # these can be group or dataset names 
#     print("Keys: %s" % f.keys())
#     # get first object name/key; may or may NOT be a group
#     a_group_key = list(f.keys())[0]

#     # get the object type for a_group_key: usually group or dataset
#     print(type(f[a_group_key])) 

#     # If a_group_key is a group name, 
#     # this gets the object names in the group and returns as a list
#     data = list(f[a_group_key])

#     # If a_group_key is a dataset name, 
#     # this gets the dataset values and returns as a list
#     # data = list(f[a_group_key])
#     # # preferred methods to get dataset values:
#     ds_obj = f[a_group_key]      # returns as a h5py dataset object
#     # ds_arr = f[a_group_key][()]  # returns as a numpy array

#     print(f"data:\n {ds_obj}")


# ---------------------

# # Source - https://stackoverflow.com/q/28170623
# # Posted by Sameer Damir, modified by community. See post 'Timeline' for change history
# # Retrieved 2026-05-01, License - CC BY-SA 3.0

# import h5py    
# import numpy as np    
# f1 = h5py.File("../demonstrations/1776983448_1210191/demo.hdf5",'r+')    

# --------------------

# import h5py

# # Open the HDF5 file in read-only mode
# with h5py.File('../demonstrations/1776983448_1210191/demo.hdf5', 'r') as file:
#     # List all groups and datasets at the root level
#     print("Keys in the file:", list(file.keys()))
    
#     # Access a specific dataset
#     dataset = file['/group1/dataset1']
    
#     # Read the entire dataset into a NumPy array
#     data_array = dataset[...]
    
#     # Display the shape and datatype of the dataset
#     print("Dataset shape:", data_array.shape)
#     print("Dataset datatype:", data_array.dtype)

# ----------------------


# import h5py
# filename = "../demonstrations/1776983448_1210191/demo.hdf5"
# mode = 'r' # read only

# f = h5py.File(filename, mode)

# for key in f.keys():
#     print(f"key: {key}") #Names of the root level object names in HDF5 file - can be groups or datasets.
#     print(f"key type: {type(f[key])}") # get the object type: usually group or dataset

# #Get the HDF5 group; key needs to be a group name from above
# group = f[key]

# #Checkout what keys are inside that group.
# print(f"keys inside group '{group}':")
# for key in group.keys():
#     print(key)

# # This assumes group[some_key_inside_the_group] is a dataset, 
# # and returns a np.array:
#     data = group["demo_1"][()]
# #Do whatever you want with data

# #After you are done
# f.close()

# --------------


import h5py
import glob

input_files = glob.glob("../demonstrations/*/demo.hdf5")
input_files = glob.glob("../demonstrations/merged_raw.hdf5")

demo_counter = 0

for file in input_files:
    with h5py.File(file, "r") as fin:
        print(f"fin groups: {fin.keys()}")
        print(f"env name: {fin['data'].attrs['env']}")
        
        for demo in fin["data"]:
            src = fin["data"][demo]
            # print(f"src: {src}")
            print(f"stuff: {src.keys()}")
            # for key in src:
            #     # src.copy(key, dst)
            #     print(f"key: {key}")


