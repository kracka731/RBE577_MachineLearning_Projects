import h5py
import glob

input_files = glob.glob("../demonstrations/*/demo.hdf5")
output_file = "../demonstrations/merged_raw.hdf5"

demo_counter = 0

with h5py.File(output_file, "w") as f_dest:

    datagroup_dst = f_dest.create_group("data")


    for file in input_files:

        with h5py.File(file, "r") as f_src:

            env_name_dst = datagroup_dst.attrs.create("env", f_src['data'].attrs['env'])
            env_info_dst = datagroup_dst.attrs.create("env_info", f_src['data'].attrs['env_info'])
            repository_version_dst = datagroup_dst.attrs.create("repository_version", f_src['data'].attrs['repository_version'])

            print(f"data_group_dst: {f_dest.keys()}")
            print(f"src: {f_src.keys()}")


            for demo in f_src["data"]:

                demo_datagroup_src = f_src["data"][f"{demo}"]

                datagroup_dst.copy(demo_datagroup_src, datagroup_dst, f"demo_{demo_counter}")
                print(f"datagroup_dst keys: {datagroup_dst.keys()}")
                print(f"demo_datagroup_dst keys: {datagroup_dst.keys()}")

                demo_counter += 1


print(f"Merged {demo_counter} demos")