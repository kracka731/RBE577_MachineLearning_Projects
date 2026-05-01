import h5py
import glob

input_files = glob.glob("demonstrations/*/demo.hdf5")
output_file = "demonstrations/merged_raw.hdf5"

demo_counter = 0

with h5py.File(output_file, "w") as fout:
    data_group = fout.create_group("data")

    for file in input_files:
        with h5py.File(file, "r") as fin:
            for demo in fin["data"]:
                src = fin["data"][demo]
                dst = data_group.create_group(f"demo_{demo_counter}")

                for key in src:
                    src.copy(key, dst)

                demo_counter += 1

print(f"Merged {demo_counter} demos")