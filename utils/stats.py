import h5py
import numpy as np
import argparse

def compute_statistics(hdf5_path, dataset_name):
    with h5py.File(hdf5_path, 'r') as f:
        if dataset_name not in f:
            raise ValueError(f"Dataset {dataset_name} not found in the HDF5 file.")

        data = f[dataset_name][:]

        if data.ndim != 3:
            raise ValueError("Expected 3-dimensional data.")

        mean_val = np.mean(data)
        std_dev_val = np.std(data)

        return mean_val, std_dev_val

def main():
    parser = argparse.ArgumentParser(description="Compute mean and standard deviation of 3D HDF5 image data.")
    parser.add_argument('hdf5_path', type=str, help="Path to the HDF5 file.")
    parser.add_argument('dataset_name', type=str, help="Name of the dataset within the HDF5 file.")

    args = parser.parse_args()

    mean_val, std_dev_val = compute_statistics(args.hdf5_path, args.dataset_name)

    print(f"Mean: {mean_val}")
    print(f"Standard Deviation: {std_dev_val}")

if __name__ == "__main__":
    main()
