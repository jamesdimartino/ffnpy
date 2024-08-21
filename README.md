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
Once training has been completed, you must select a model to use for inference. The trained models are saved every 500,000 training iterations (this can be altered at line 271). They are named like <seedID>_<number_of_iterations>.pkl. The highest number of iterations will generally be the best trained and should be selected for use during inference.

---
## Inference
Inference is conducted on smaller subvolumes that can then be rejoined. The general workflow of inference is to run inference.py on a subvolume, producing an npz file that contains the segmentations, then running extractor.py to obtain the numpy file containing the segmentations:
```bash
python inference.py --image_mean <mean> --image_stddev <stddev> --image_path <raw_filepath:dataset> --checkpoints_path <checkpoints_filepath> --seg_result_path <result_filepath> --base_dir <dir_with_model> --model_name <model_name> --corner <upper_subvol_corner> --subvol_size <size_of_subvol>

python extractor.py <input_path.npz> <output_path.npy>
```
This process, especially when using large subvolumes, can be long. Instead, I have produced an end-to-end pipeline that simply requires an input file of the subvolume and an output directory. To use it, call:
```bash
bash inference.sh <input_volume> <output_dir>
```
Inference.sh calls volume_inference.py, which automatically subdivides large volumes into Zx512x512 subvolumes and calls inference.py for each one, skipping empty subvolumes and creating empty label subvolumes in such cases. The results of this process are stored in output_dir/int_results, with the final, stitched result stored as output_dir/final_result.tif. Importantly, the x and y dimensions should both be divisible by 512, and the dataset containing the raw data should be called 'raw'. Once inference has been run for all subvolumes, they are extracted into numpy files and then stitched together using stitch_save.py. The merging logic for stitch_save.py is not ideal in this application and implementing a graphing based approach to this is a future goal. Merging across stitches is turned off by default.
