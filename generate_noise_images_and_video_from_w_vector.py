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


def slerp(val, low, high):
        out = np.zeros([low.shape[0],low.shape[1],low.shape[2]])

        for i in range(low.shape[1]):
            omega = np.arccos(np.clip(np.dot(low[0][i]/np.linalg.norm(low[0][i]), high[0][i]/np.linalg.norm(high[0][i])), -1, 1))
            so = np.sin(omega)
            if so == 0:
                out[i] = (1.0-val) * low[0][i] + val * high[0][i] # L'Hopital's rule/LERP
            out[0][i] = np.sin((1.0-val)*omega) / so * low[0][i] + np.sin(val*omega) / so * high[0][i]
        return out

def slerp_interpolate(zs, steps):
    out = []
    for i in range(len(zs)-1):
        for index in range(steps):
            fraction = index/float(steps)
            out.append(slerp(fraction,zs[i],zs[i+1]))
    return out

def generate_images_in_w_space(ws, Gs, truncation_psi, outdir, prefix, save_npy, save_video, framerate=6, vidname="out", verbose=False, class_idx=None):

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
        Image.fromarray(images[0], 'RGB').save(f'{outdir}/{prefix}{w_idx:05d}.png')
        if save_npy:
            np.save(f'{outdir}/vectors/{prefix}{w_idx:05d}.npz',w)
            # np.savetxt(f'{outdir}/vectors/{prefix}{w_idx:05d}.txt',w.reshape(w.shape[0], -1))

    if save_video:
        cmd="ffmpeg -loglevel quiet -y -r {} -i {}/{}%05d.png -vcodec libx264 -pix_fmt yuv420p {}/{}.mp4".format(framerate,outdir,prefix,outdir,vidname)
        if verbose:
          print(cmd)
        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL)
        

def parse_json(json_path):
    with open(json_path) as f:
      json_dict = json.load(f)
    return json_dict

def initialize(network_path, verbose):

    # LOAD TF PKL MODEL
    tflib.init_tf()
    with dnnlib.util.open_url(network_path) as fp:
        _G, _D, Gs = pickle.load(fp)

    # LOAD USER00's W-VECTOR
    w_user00 = np.load('w5-user00.npy')
    w_user00_mat = w_user00[np.newaxis, :, :]

    if verbose:
        print(f'Loaded network from {network_path}.')
    
    return Gs, w_user00_mat

def process(input_path, output_path, verbose, preloaded_params, num_steps):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    Gs = preloaded_params[0]
    w_user00_mat = preloaded_params[1]
    
    w_last = np.load(input_path)
    if verbose:
        print(f'Loaded input vector from {input_path}.')

    
    # RESHAPE FROM (18, 512) to (1,18,512)
    w_last_mat = w_last[np.newaxis, :, :]

    truncation_psi = 0.4


    # GEN FROM W-SPACE: SLERP VIDEO
    time_0 = time.time()    
    print(f'Generating spherical interpolation mp4, of {num_steps} frames...')

    ws_slerp = slerp_interpolate([w_user00_mat, w_last_mat], num_steps)

    generate_images_in_w_space(ws_slerp, Gs, truncation_psi, output_path,'slerp', False, True, vidname='slerp',class_idx=None, verbose=verbose)
    time_1 = time.time()

    print(f'Wrote out slerp.mp4 to {output_path}. Generating the interpolation video took {time_1 - time_0} seconds.')


    # GEN FROM W-SPACE: SAVE 10 NOISE IMGS
    print('Generating 10 noise imgs...')

    ws_noise = []
    for i in range(10):
      noise = np.random.normal(0, (i+1)/10.0, w_last_mat.shape)
      ws_noise.append(w_last_mat + noise)
    generate_images_in_w_space(ws_noise, Gs, truncation_psi, output_path, 'noise', False, False, class_idx=None, verbose=verbose)
    time_2 = time.time()

    print(f'Wrote out 10 noise imgs to {output_path}. Generating the 10 images took {time_2 - time_1} seconds.')

    # Emit DONE signal
    print('DONE.')
    Path(f'{output_path}/clonegan_seq_imgs_video.done').touch()


def doLoop(preloaded_params, json_path, sleep_time, verbose, num_steps):
        
    while True:

        if os.path.exists(json_path):
            # parse json
            json_dict = parse_json(json_path)

            #del json
            if verbose:
                print(f'Deleting JSON.')
            os.remove(json_path)

            process(json_dict['input_path'], json_dict['output_path'], verbose, preloaded_params, num_steps)

        else:
            SLEEP_TIME_IN_SECS = sleep_time/1000.0
            if verbose:
                print(f'JSON not found. Sleeping {SLEEP_TIME_IN_SECS} seconds.')
            time.sleep(SLEEP_TIME_IN_SECS)


def start(preloaded_network, json_path, sleep_time, verbose, num_steps):
    # doLoop(preloaded_network, json_path, sleep_time, verbose, num_steps)
    try:
        doLoop(preloaded_network, json_path, sleep_time, verbose, num_steps)
    except:
        if verbose:
            print(f'Exception thrown during loop. Deleting JSON and re-entering loop.')
        os.remove(json_path)
        start(preloaded_network, json_path, sleep_time, verbose, num_steps)


def main():

    parser = argparse.ArgumentParser(
        description='Return the 10 images and an mp4 from the inputted vector file.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--network_path',     help='Network filepath as a .pt file', dest='network_path', required=True)
    parser.add_argument('--json_path',      help='File path to json arg file', dest='json_path', required=True)
    parser.add_argument('--sleep_time',      help='Sleep time in milliseconds', dest='sleep_time', type=int, default=50)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--steps',  help='Number of steps in the slerp video', dest='num_steps', type=int, default=30, required=False)

    args = parser.parse_args()

    preloaded_params = initialize(args.network_path, args.verbose)
    start(preloaded_params, args.json_path, args.sleep_time, args.verbose, args.num_steps)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
