import numpy as np
import argparse

def load_segmentation(args):
  npz = np.load(args.segmentation_file)
  seg = npz['segmentation']
  return seg

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Load segmentation from FFN subvolume.')
    parser.add_argument('segmentation_file', type=str, help='File containing FFN subvolume')
    parser.add_argument('output_file', type=str, help='Filepath for segmentation output')
    return parser.parse_args()

def main():
    args = parse_args()
    output = load_segmentation(args)
    np.save(args.output_file, output)
    print(f"Finished")
if __name__ == "__main__":
    main()
