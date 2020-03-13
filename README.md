# Checkpoint2Numpy

A simple utility for extracting tensors from a TensorFlow checkpoint.  
A checkpoint file is a serialized data containing weight matrices and their node names.  
This utility will read a checkpoint file and export them to individual `.npy` files,  
along with a `.json` text file that contains metadata such as node names and checksums.

## Getting Started

### Setup

- Python 3
- Tensorflow 1 or 2
- NumPy

### Run the app

To run Checkpoint2Numpy, run `run.py`. On the dialog box, select your `.index` or `.ckpt` file.

```bash
python run.py
```
