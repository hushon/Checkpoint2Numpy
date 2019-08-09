'''
Checkpoint To Numpy

This script extracts numpy arrays from a TensorFlow checkpoint file.

Requirements: 
    Python 3
    TensorFlow
    NumPy

Supported checkpoint formats:
    TF checkpoint V1 and V2 formats.
    For V1 format, select .ckpt file.
    For V2 format, select .index/.meta/.data file.

About exported files: 
    Tensors are exported as .npy files.
    You can read the binary with NumPy.
    JSON file contains list of metadata; such as tensor names and checksum.
'''

import os
import json
from tkinter import Tk, filedialog
import hashlib

import numpy as np
import tensorflow as tf

def save_as_json(path, data):
    '''
    Save data as JSON file format
    Args:
        path (str): directory and filename to .json file
        data (dict): dictionary-type data to be written into json
    '''
    if not os.path.exists(os.path.dirname(path)): 
        os.mkdir(os.path.dirname(path))

    with open(path, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)
    print('Saved as {}'.format(path))

def read_json(path):
    '''
    Read JSON file
    Args:
        path (str): path to .json file
    '''
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def md5_checksum(path):
    '''
    calculates MD5 checksum of given file
    Args:
        path (str): path to target file
    Returns: 
        checksum (str): a md5 checksum string
    '''

    with open(path, 'rb') as file:
        checksum = hashlib.md5(file.read()).hexdigest()
    return checksum

def checkpoint_to_dictionary(checkpoint_path):
    '''
    read checkpoint file and return weights as dictionary.
    Args:
        checkpoint_path (str): path to checkpoint file.
    Returns:
        (dict) a dictionary containing tensor names as keys and numpy arrays as values.
    Example:
        To load checkpoint V1 formats, pass full path to .ckpt file: 
            x = checkpoint_to_dictionary('./checkpoint/pix2pix.model-88500.ckpt')
        To load checkpoint V2 formats, pass path up to prefix of .index/.meta/data files.
            x = checkpoint_to_dictionary('./checkpoint/pix2pix.model-88500')
    '''

    reader = tf.train.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = sorted(reader.get_variable_to_shape_map())

    tensor_dict = {name:reader.get_tensor(name) for name in var_to_shape_map}

    return tensor_dict

def main():

    root = Tk()
    checkpoint_path = filedialog.askopenfilename(title='Select checkpoint file', filetypes = (("index files","*.index"),("ckpt files","*.ckpt"),("all files","*.*")))
    checkpoint_path = os.path.normpath(checkpoint_path)
    root.withdraw()

    root = Tk()
    save_dir = filedialog.askdirectory(title='Export NumPy binaries to...')
    save_dir = os.path.normpath(save_dir)
    root.withdraw()

    # if checkpoint V2 format, remove file extension
    if os.path.splitext(checkpoint_path)[1] in ['.index', '.meta', '.data']:
        checkpoint_path = os.path.splitext(checkpoint_path)[0]

    # read checkpoint file
    vars_dict = checkpoint_to_dictionary(checkpoint_path)

    # print summary about tensors
    print('Number of tensors found: {}'.format(len(vars_dict)))
    print('| Tensor Name | Shape | DType |')
    print('=======')
    for name, tensor in vars_dict.items():
        print('| {} | {} | {} |'.format(name, tensor.shape, tensor.dtype))

    # save npy file and metadata
    metadata = []

    for name, tensor in vars_dict.items():

        filename = name[:].replace('/', '_') + '.npy'
        np.save(os.path.join(save_dir, filename), tensor)

        metadata.append(
            {
            'tensor_name': name, 
            'filename': filename, 
            'md5_checksum': md5_checksum(os.path.join(save_dir, filename))
            })

    metadata_filename = os.path.basename(checkpoint_path)+'_metadata.json'
    save_as_json(os.path.join(save_dir, metadata_filename), metadata)

if __name__ == '__main__':
    main()