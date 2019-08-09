'''
Run ckpt2npy.py with file dialog window.
'''

import os
import argparse
from tkinter import Tk, filedialog
from ckpt2npy import main

if __name__ == '__main__':

    root = Tk()
    checkpoint_path = filedialog.askopenfilename(title='Select checkpoint file', filetypes = (("index files","*.index"),("ckpt files","*.ckpt"),("all files","*.*")))
    checkpoint_path = os.path.normpath(checkpoint_path)
    root.withdraw()

    root = Tk()
    save_dir = filedialog.askdirectory(title='Export NumPy binaries to...')
    save_dir = os.path.normpath(save_dir)
    root.withdraw()

    args = argparse.ArgumentParser().parse_args()
    args.checkpoint_path = checkpoint_path
    args.dest = save_dir

    main(args)
