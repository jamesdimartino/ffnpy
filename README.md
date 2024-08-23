# FFN-PyTorch
---
## Training
FFN will not work without volume-specific training. It is reasonable to train with a 500-600 edge cube - 150 million voxels, or a ~531 edge cube, of that should be annotated according to Google. Using less than this (i.e. a half-annotated 500 edge cube) will still work. Additionally, one can train a model, obtain results for the volume with the model, and use these results to train the model again. This will lead to performance improvements over multiple iterations. To train, obtain a raw volume in hdf5 format and annotations for the volume in hdf5 format. Run data/partition.py and data/build_coordinates.py:
```bash
python data/partition.py --input_volume <input_volume_path.h5:dataset> --output_volume <output_volume_path.h5:dataset> --min_size <min_size>

python data/build_coordinates.py --partition_volumes <dataset:partition_volume_path.h5:dataset> --coordinate_output <coordinate_file_path.npy>
```
There are a number of parameters that can be altered. A min_size of >40,000 voxels is reasonable in most cases. For the various margins and radii that can be changed throughout the training process, the default values are consistent with one another. To change these, see the Google ffn github for instructions.
Now coordinates and datasets have been prepared for training. In order to train, we need a few more things, most importantly the standard deviation and mean of the image. To obtain these:
```bash
python util/stats.py <filepath.h5> <dataset>
```
Now, we can compose our train command:
```bash
python train.py --train_coords <coordinate_file_path.npy> --data_volumes <raw_image_volume.h5:dataset> --label_volumes <label_volume:dataset> --checkpoints <checkpoint_path> --starting_model <model_dir/model_num.pkl> --image_mean <mean> --image_stddev <stddev>
```

Importantly, label_volumes is NOT the partition volume generated earlier, it is the original label volume that was produced. The checkpoints should be a directory where you want the models saved as train.py runs. There are additional hyperparameters that can be altered; see the Google ffn github for more information. 
Once training has been completed, you must select a model to use for inference. The trained models are saved every 500,000 training iterations (this can be altered at line 271). They are named like <seed_ID>_<number_of_iterations>.pkl. The highest number of iterations will generally be the best trained and should be selected for use during inference.

---
## Inference
Inference is conducted on smaller subvolumes that can then be rejoined. The general workflow of inference is to run inference.py on a subvolume, producing an npz file that contains the segmentations, then running extractor.py to obtain the numpy file containing the segmentations:
```bash
python inference.py --image_mean <mean> --image_stddev <stddev> --image_path <raw_filepath:dataset> --checkpoints_path <checkpoints_filepath> --seg_result_path <result_filepath> --base_dir <dir_with_model> --model_name <model_name> --corner <upper_subvol_corner> --subvol_size <size_of_subvol>

python extractor.py <input_path.npz> <output_path.npy>
```
This process, especially when using large subvolumes, can be long. Instead, I have produced an end-to-end pipeline that simply requires an input file containing the entire raw volume and an output directory. To use it, call:
```bash
bash inference.sh <input_volume> <output_dir>
```
Inference.sh calls volume_inference.py, which automatically subdivides large volumes into Zx512x512 subvolumes and calls inference.py for each one, skipping empty subvolumes and creating empty label subvolumes in such cases. The results of this process are stored in output_dir/int_results, with the final, stitched result stored as output_dir/final_result.tif. Importantly, the x and y dimensions should both be divisible by 512, and the dataset containing the raw data should be called 'raw'. Once inference has been run for all subvolumes, they are extracted into numpy files and then stitched together using stitch_save.py. The merging logic for stitch_save.py is not ideal in this application and implementing a graphing based approach to this is a future goal. Merging across stitches is turned off by default.
If you are using inference.sh to run automated inference and want to change the model being used for inference, there is a dictionary of sorts at line 18 of volume_inference.py, which is the function get_model_parameters. Simply change the name of the directory that stores the models, the name of the model you wish to use for inference, and the name of the volume you want to perform inference on.

---
## Explanations of scripts

### inference.py

Runs inference for a given subvolume. Incoming file must be in hdf5 format. Output will be a probability map npz file and a segmentation npz file in the output directory, as well as any checkpoints created during the process. See Januszewski et al. 2019 for explanation of logic. Not advisable to change the hyperparameters around too much without reading the paper.

### volume_inference.py

Calls inference.py for a large volume of EM data by generating prompts that will call inference.py for smaller subvolumes of the larger volume. Takes an input hdf5 filepath and outputs probability maps and segmentations to the output directory. This assumes that the x and y dimensions are divisible by 512, as each subvolume is the full z stack, cropped to 512x512 in the x and y directions. Mean and standard deviation is also computed on a per-subvolume basis. If a subvolume is completely empty due to there being a buffer for the entire subvolume, an empty .npy file is put in the results file with the same naming conventions that inference.py uses (seg_zcorner_xcorner_ycorner.npy). Also contains a dictionary to handle using different models for different volumes - the file paths for this directory should be changed once the git is installed.

### extractor.py

The output of the inference function is .npz files, which have a segmentation array stored within them. This function stores that array as a numpy file with the same name as the parent npz.

### stitch_save.py

This function stitches the subvolume results back together. It takes a directory with numpy files labeled with the upper corners and uses the file names to assemble a large array the size of the original volume. Labels IDs are increased to ensure that unique labels in each subvolume remain unique in the finished result. It also has functionality that allows for merging over stitches, with a threshold for the number of consecutive voxels that are different on either side in order to trigger the two labels being merged. Unless segmentation quality is very good, this will not work and will lead to a large amount of excessive merging. Implementing an object tracking approach to fix this is a future goal.

### inference.sh

Runs the pipeline end-to-end for a large volume, calling volume inference, extracting the numpy arrays from every npz file, and stitching together the result.
