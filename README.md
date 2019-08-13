# Checkpoint2Numpy
Extract tensors from TensorFlow checkpoint file. 
A checkpoint file is a serialized data containing weight matrices and their node names. 
This utility will read a checkpoint file and export them to individual `.npy` files, 
along with a `.json` text file that contains metadata such as node names and checksums.

## Getting Started
### Setup
To run Checkpoint2Numpy, Python3 and TensorFlow and NumPy packages are required. 

On native Python 3 environment: 
```
$ pip install -r requirements.txt
```

On Anaconda Python 3 environment: 
```
$ conda install --file requirements.txt
```

### Running Checkpoint2Numpy
To run Checkpoint2Numpy, run `run.py`.
```
$ python run.py
```
