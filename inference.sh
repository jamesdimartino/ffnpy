#!/bin/bash

# Ensure that the script is called with two arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_file> <output_dir>"
    exit 1
fi

# Assign the input arguments to variables
input_file=$1
output_dir=$2

# Ensure the output directory exists
mkdir -p "${output_dir}"
echo $output_dir
# Compute the overall shape from the input file using Python
shape=($(python -c "
import h5py
import sys
with h5py.File('$input_file', 'r') as f:
    print(f['raw'].shape)
"))

ogdepth=${shape[0]//[(),]/}
ogheight=${shape[1]//[(),]/}
ogwidth=${shape[2]//[(),]/}

# Run the vol_inf.py script with the computed shape
python vol_inf.py --input_file="${input_file}" --output_dir="${output_dir}"

# Create a directory to store intermediate numpy results
int_results="${output_dir}/int_results"
mkdir -p "${int_results}"

# Iterate over all npz files in the temp directory and run the extractor
for npz_file in "${output_dir}/temp/seg_"*.npz; do
    # Extract the base name from the npz file (remove the directory and extension)
    base_name=$(basename "$npz_file" .npz)

    # Run the extractor and save the result in the intermediate results directory
    python extractor.py "${npz_file}" "${int_results}/${base_name}.npy"
done

# Run the stitch and save process using the computed shape
python stitch_save.py "${int_results}" "${output_dir}/final_result.tif" --original_shape ${ogdepth} ${ogheight} ${ogwidth} --tile_shape ${ogdepth} 512 512

# Cleanup: remove the temp directory if no longer needed
