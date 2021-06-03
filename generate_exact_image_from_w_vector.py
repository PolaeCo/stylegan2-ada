import argparse
from argparse import Namespace
import time
import os
import sys
import pprint
import numpy as np
from PIL import Image
import json
from pathlib import Path

from numpy import newaxis

import subprocess
import pickle
import re

import scipy
from numpy import linalg

import dnnlib 
import dnnlib.tflib as tflib

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import moviepy.editor
from opensimplex import OpenSimplex

import warnings # mostly numpy warnings for me
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


def generate_images_in_w_space(ws, Gs, truncation_psi, outdir, prefix, save_npy, save_video, framerate=6, vidname="out", verbose=False, class_idx=None, output_fpath=None):

    Gs_kwargs = {
        'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
        'randomize_noise': False
    }
    if truncation_psi is not None:
        Gs_kwargs['truncation_psi'] = truncation_psi

    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    label = np.zeros([1] + Gs.input_shapes[1][1:])
    if class_idx is not None:
        label[:, class_idx] = 1

    for w_idx, w in enumerate(ws):
        if verbose:
            print('Generating image for step %d/%d ...' % (w_idx, len(ws)))
        noise_rnd = np.random.RandomState(1) # fix noise
        tflib.set_vars({var: noise_rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images = Gs.components.synthesis.run(w, **Gs_kwargs) # [minibatch, height, width, channel]
        # images = Gs.run(w,label, **Gs_kwargs) # [minibatch, height, width, channel]

        if output_fpath is None:
            output_fpath = f'{outdir}/{prefix}{w_idx:05d}.png'

        Image.fromarray(images[0], 'RGB').save(output_fpath)
        if save_npy:
            np.save(f'{outdir}/vectors/{prefix}{w_idx:05d}.npz',w)
            # np.savetxt(f'{outdir}/vectors/{prefix}{w_idx:05d}.txt',w.reshape(w.shape[0], -1))

    if save_video:
        cmd="ffmpeg -loglevel quiet -y -r {} -i {}/{}%05d.png -vcodec libx264 -pix_fmt yuv420p {}/{}.mp4".format(framerate,outdir,prefix,outdir,vidname)
        if verbose:
          print(cmd)
        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL)
        

def generate_exact_image_from_w_vector(network_pkl: str, vector_fpath: str, output_fpath: str):

    tflib.init_tf()
    with dnnlib.util.open_url(network_pkl) as fp:
        _G, _D, Gs = pickle.load(fp)

    def newaxis_reshape(w):
      return w[newaxis, :, :]

    ws = [newaxis_reshape(np.load(vector_fpath))]

    truncation_psi = 0.4
    outdir = '' # not needed bc we have an output_fpath specified; we're only saving out 1 img
    prefix = '' # not needed bc we have an output_fpath specified; we're only saving out 1 img
    save_npy = False
    save_video = False

    # make sure output folder exists, otherwise saving won’t work
    output_basedir = os.path.dirname(output_fpath)
    if not os.path.exists(output_basedir):
        os.makedirs(output_basedir)

    generate_images_in_w_space(ws, Gs, truncation_psi, outdir, prefix, save_npy, save_video, framerate=6, vidname="out", verbose=False, class_idx=None, output_fpath=output_fpath)



_examples = '''examples:
  python %(prog)s --output_path=out.jpg --vector_fpath=dlatents.npz \\
      --network=toonify.pkl
'''


#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate an exact image from the given W vector.',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--network',     help='Network pickle filename', dest='network_pkl', required=True)
    parser.add_argument('--vector_fpath',      help='The W vector to generate from, as an .npy file.', dest='vector_fpath', required=True)
    parser.add_argument('--output_fpath',      help='The filename of the output image', required=True, dest='output_fpath', metavar='DIR')

    time_before = time.time()
    generate_exact_image_from_w_vector(**vars(parser.parse_args()))
    time_after = time.time()
    print('Total time to generate took {:.4f} seconds.'.format(time_after - time_before))
    

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------