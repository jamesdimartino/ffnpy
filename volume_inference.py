import os
import h5py
import numpy as np
import shutil
from absl import flags, app
from itertools import product

FLAGS = flags.FLAGS

# Define flags for input file and output directory
flags.DEFINE_string('input_file', None, 'Path to the input HDF5 file')
flags.DEFINE_string('output_dir', None, 'Directory to save the final results')

def compute_mean_std(volume):
    """Compute the mean and standard deviation of the volume."""
    return np.mean(volume), np.std(volume)

def get_model_parameters(output_dir):
    """Determine base_dir and model_name based on the start of the output directory name."""
    # Extract the base directory name from the full path
    base_dir_name = os.path.basename(output_dir)

    if base_dir_name.startswith('3-mo_molecular_1'):
        return '3m_1/', '5109463_3250000'
    elif base_dir_name.startswith('3-mo_molecular_2'):
        return '3m_2/', '7491809_4350000'
    elif base_dir_name.startswith('24-mo_molecular_1'):
        return '24m3v2/', '939109_400000'
    elif base_dir_name.startswith('24-mo_molecular_3'):
        return '24m3v2/', '939109_400000'
    else:
        raise ValueError(f"Unknown output_dir prefix: {base_dir_name}")

def process_subvolume(corner, subvol_size, temp_dir, final_result_dir, input_file):
    """Process each subvolume by running the inference and then saving results."""
    temp2_dir = os.path.join(temp_dir, 'checkpoints/')
    if not os.path.exists(temp2_dir):
        os.makedirs(temp2_dir)

    final_result_path = os.path.join(final_result_dir, 'int_results', f'seg_{corner[0]}_{corner[1]}_{corner[2]}.npy')
    temp_result_path = os.path.join(temp_dir, f'seg_{corner[0]}_{corner[1]}_{corner[2]}.npz')
    if not os.path.exists(os.path.join(final_result_dir, 'int_results')):
        os.makedirs(os.path.join(final_result_dir, 'int_results'))

    # Check if the result file already exists and contains data
    if os.path.exists(temp_result_path):
        with np.load(temp_result_path) as data:
            if data.files:  # Check if the npz file has any arrays
                print(f"Skipping inference for {temp_result_path} as it already exists and contains data.")
                return

    with h5py.File(input_file, 'r') as f:
        subvol = f['raw'][corner[0]:corner[0]+subvol_size[0],
                          corner[1]:corner[1]+subvol_size[1],
                          corner[2]:corner[2]+subvol_size[2]]

        # Check if the subvolume is entirely empty
        if np.all(subvol == 0):
            # If the subvolume is empty, create an empty 3D array of zeros
            empty_output_array = np.zeros(subvol_size, dtype=subvol.dtype)
            np.save(final_result_path, empty_output_array)
            return

        mean, std = compute_mean_std(subvol)

        # Get the model parameters based on output_dir
        base_dir, model_name = get_model_parameters(final_result_dir)

        # Construct the command for inference
        cmd = (
            f"python inference.py "
            f"--image_mean={int(mean)} "
            f"--image_stddev={int(std)} "
            f"--image_path={input_file}:raw "
            f"--checkpoints_path={temp2_dir} "
            f"--seg_result_path={temp_dir} "
            f"--corner={corner[0]},{corner[1]},{corner[2]} "
            f"--subvol_size={subvol_size[0]},{subvol_size[1]},{subvol_size[2]} "
            f"--base_dir={base_dir} "
            f"--model_name={model_name}"
        )

        # Execute the inference command
        os.system(cmd)

        # The results from inference are saved directly in the temp_dir and final_result_dir

def main(argv):
    del argv  # Unused

    input_file = FLAGS.input_file
    output_dir = FLAGS.output_dir

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a temporary directory for intermediate files
    temp_dir = os.path.join(output_dir, 'temp/')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Open the HDF5 file to get the shape of the volume
    with h5py.File(input_file, 'r') as f:
        z_length, y_length, x_length = f['raw'].shape

    # Define the subvolume size (z_length remains full)
    subvol_size = [z_length, 512, 512]

    # Generate the list of x and y positions with steps of 512
    x_positions = list(range(0, x_length - 512 + 1, 512))
    y_positions = list(range(0, y_length - 512 + 1, 512))

    # Iterate over all combinations of x and y positions
    for x, y in product(x_positions, y_positions):
        corner = [0, y, x]
        print(corner)
        process_subvolume(corner, subvol_size, temp_dir, output_dir, input_file)

    # Cleanup the temporary directory after processing all subvolumes
if __name__ == '__main__':
    flags.mark_flag_as_required('input_file')
    flags.mark_flag_as_required('output_dir')
    app.run(main)
