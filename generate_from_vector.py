from pretrained_networks import *

import time
import argparse
import os
import numpy as np
from PIL import Image

from pathlib import Path


def generate_from_vector(network_pkl: str, vector_fpath: str, output_fpath: str):

    time_0= time.time()
    # Load network pkl
    _, _, Gs = load_networks(network_pkl)

    time_1 = time.time()

    print('Time to load model took {:.4f} seconds.'.format(time_1 - time_0))


    # load numpy array
    z = np.load(vector_fpath)
    # check if it's actually npz:
    if 'dlatents' in z:
        z = z['dlatents']

    time_2 = time.time()

    print('Time to load z took {:.4f} seconds.'.format(time_2 - time_1))


    synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=False), minibatch_size=8)
    time_3 = time.time()
#     print('Time to synthesis_kwargs took {:.4f} seconds.'.format(time_3 - time_2))


    images = Gs.components.synthesis.run(z, randomize_noise=False, **synthesis_kwargs)
    time_4 = time.time()
    print('Time to gen img from z took {:.4f} seconds.'.format(time_4 - time_3))

    Image.fromarray(images.transpose((0,2,3,1))[0], 'RGB').save(output_fpath)
    time_5 = time.time()
    print('Time to save out output img took {:.4f} seconds.'.format(time_5 - time_4))





_examples = '''examples:

  python %(prog)s --output_path=out.jpg --vector_fpath=dlatents.npz \\
      --network=toonify.pkl
'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate an image from the given vector in the latent space of a pretrained network pickle.',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--network',     help='Network pickle filename', dest='network_pkl', required=True)
    parser.add_argument('--vector_fpath',      help='The vector to generate from, as an .npz file. You can get an .npz vector from running the projector.py script.', dest='vector_fpath', required=True)
    parser.add_argument('--output_fpath',      help='The filename of the output image', required=True, dest='output_fpath', metavar='DIR')

    time_before = time.time()
    generate_from_vector(**vars(parser.parse_args()))
    time_after = time.time()
    print('Total time for generate_from_vector took {:.4f} seconds.'.format(time_after - time_before))
    

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
