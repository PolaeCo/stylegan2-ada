# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import argparse
import os
import pickle
import imageio
import time
import json

import numpy as np
import PIL.Image
import tensorflow as tf
import tqdm

import dnnlib
import dnnlib.tflib as tflib


class Projector:
    def __init__(self):
        self.num_steps                  = 1000
        self.dlatent_avg_samples        = 10000
        self.initial_learning_rate      = 0.1
        self.initial_noise_factor       = 0.05
        self.lr_rampdown_length         = 0.25
        self.lr_rampup_length           = 0.05
        self.noise_ramp_length          = 0.75
        self.regularize_noise_weight    = 1e5
        self.verbose                    = True

        self._Gs                    = None
        self._minibatch_size        = None
        self._dlatent_avg           = None
        self._dlatent_std           = None
        self._noise_vars            = None
        self._noise_init_op         = None
        self._noise_normalize_op    = None
        self._dlatents_var          = None
        self._dlatent_noise_in      = None
        self._dlatents_expr         = None
        self._images_float_expr     = None
        self._images_uint8_expr     = None
        self._target_images_var     = None
        self._lpips                 = None
        self._dist                  = None
        self._loss                  = None
        self._reg_sizes             = None
        self._lrate_in              = None
        self._opt                   = None
        self._opt_step              = None
        self._cur_step              = None

    def _info(self, *args):
        if self.verbose:
            print('Projector:', *args)

    def set_num_steps(self, num_steps):
        self.num_steps = num_steps

    def set_network(self, Gs, dtype='float16'):
        if Gs is None:
            self._Gs = None
            return
        self._Gs = Gs.clone(randomize_noise=False, dtype=dtype, num_fp16_res=0, fused_modconv=True)

        # Compute dlatent stats.
        self._info(f'Computing W midpoint and stddev using {self.dlatent_avg_samples} samples...')
        latent_samples = np.random.RandomState(123).randn(self.dlatent_avg_samples, *self._Gs.input_shapes[0][1:])
        dlatent_samples = self._Gs.components.mapping.run(latent_samples, None)  # [N, L, C]
        dlatent_samples = dlatent_samples[:, :1, :].astype(np.float32)           # [N, 1, C]
        self._dlatent_avg = np.mean(dlatent_samples, axis=0, keepdims=True)      # [1, 1, C]
        self._dlatent_std = (np.sum((dlatent_samples - self._dlatent_avg) ** 2) / self.dlatent_avg_samples) ** 0.5
        self._info(f'std = {self._dlatent_std:g}')

        # Setup noise inputs.
        self._info('Setting up noise inputs...')
        self._noise_vars = []
        noise_init_ops = []
        noise_normalize_ops = []
        while True:
            n = f'G_synthesis/noise{len(self._noise_vars)}'
            if not n in self._Gs.vars:
                break
            v = self._Gs.vars[n]
            self._noise_vars.append(v)
            noise_init_ops.append(tf.assign(v, tf.random_normal(tf.shape(v), dtype=tf.float32)))
            noise_mean = tf.reduce_mean(v)
            noise_std = tf.reduce_mean((v - noise_mean)**2)**0.5
            noise_normalize_ops.append(tf.assign(v, (v - noise_mean) / noise_std))
        self._noise_init_op = tf.group(*noise_init_ops)
        self._noise_normalize_op = tf.group(*noise_normalize_ops)

        # Build image output graph.
        self._info('Building image output graph...')
        self._minibatch_size = 1
        self._dlatents_var = tf.Variable(tf.zeros([self._minibatch_size] + list(self._dlatent_avg.shape[1:])), name='dlatents_var')
        self._dlatent_noise_in = tf.placeholder(tf.float32, [], name='noise_in')
        dlatents_noise = tf.random.normal(shape=self._dlatents_var.shape) * self._dlatent_noise_in
        self._dlatents_expr = tf.tile(self._dlatents_var + dlatents_noise, [1, self._Gs.components.synthesis.input_shape[1], 1])
        self._images_float_expr = tf.cast(self._Gs.components.synthesis.get_output_for(self._dlatents_expr), tf.float32)
        self._images_uint8_expr = tflib.convert_images_to_uint8(self._images_float_expr, nchw_to_nhwc=True)

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        proc_images_expr = (self._images_float_expr + 1) * (255 / 2)
        sh = proc_images_expr.shape.as_list()
        if sh[2] > 256:
            factor = sh[2] // 256
            proc_images_expr = tf.reduce_mean(tf.reshape(proc_images_expr, [-1, sh[1], sh[2] // factor, factor, sh[2] // factor, factor]), axis=[3,5])

        # Build loss graph.
        self._info('Building loss graph...')
        self._target_images_var = tf.Variable(tf.zeros(proc_images_expr.shape), name='target_images_var')
        if self._lpips is None:
            with dnnlib.util.open_url('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metrics/vgg16_zhang_perceptual.pkl') as f:
                self._lpips = pickle.load(f)
        self._dist = self._lpips.get_output_for(proc_images_expr, self._target_images_var)
        self._loss = tf.reduce_sum(self._dist)

        # Build noise regularization graph.
        self._info('Building noise regularization graph...')
        reg_loss = 0.0
        for v in self._noise_vars:
            sz = v.shape[2]
            while True:
                reg_loss += tf.reduce_mean(v * tf.roll(v, shift=1, axis=3))**2 + tf.reduce_mean(v * tf.roll(v, shift=1, axis=2))**2
                if sz <= 8:
                    break # Small enough already
                v = tf.reshape(v, [1, 1, sz//2, 2, sz//2, 2]) # Downscale
                v = tf.reduce_mean(v, axis=[3, 5])
                sz = sz // 2
        self._loss += reg_loss * self.regularize_noise_weight

        # Setup optimizer.
        self._info('Setting up optimizer...')
        self._lrate_in = tf.placeholder(tf.float32, [], name='lrate_in')
        self._opt = tflib.Optimizer(learning_rate=self._lrate_in)
        self._opt.register_gradients(self._loss, [self._dlatents_var] + self._noise_vars)
        self._opt_step = self._opt.apply_updates()

    def start(self, target_images):
        assert self._Gs is not None

        # Prepare target images.
        self._info('Preparing target images...')
        target_images = np.asarray(target_images, dtype='float32')
        target_images = (target_images + 1) * (255 / 2)
        sh = target_images.shape
        assert sh[0] == self._minibatch_size
        if sh[2] > self._target_images_var.shape[2]:
            factor = sh[2] // self._target_images_var.shape[2]
            target_images = np.reshape(target_images, [-1, sh[1], sh[2] // factor, factor, sh[3] // factor, factor]).mean((3, 5))

        # Initialize optimization state.
        self._info('Initializing optimization state...')
        dlatents = np.tile(self._dlatent_avg, [self._minibatch_size, 1, 1])
        tflib.set_vars({self._target_images_var: target_images, self._dlatents_var: dlatents})
        tflib.run(self._noise_init_op)
        self._opt.reset_optimizer_state()
        self._cur_step = 0

    def step(self):
        assert self._cur_step is not None
        if self._cur_step >= self.num_steps:
            return 0, 0

        # Choose hyperparameters.
        t = self._cur_step / self.num_steps
        dlatent_noise = self._dlatent_std * self.initial_noise_factor * max(0.0, 1.0 - t / self.noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
        learning_rate = self.initial_learning_rate * lr_ramp

        # Execute optimization step.
        feed_dict = {self._dlatent_noise_in: dlatent_noise, self._lrate_in: learning_rate}
        _, dist_value, loss_value = tflib.run([self._opt_step, self._dist, self._loss], feed_dict)
        tflib.run(self._noise_normalize_op)
        self._cur_step += 1
        return dist_value, loss_value

    @property
    def cur_step(self):
        return self._cur_step

    @property
    def dlatents(self):
        return tflib.run(self._dlatents_expr, {self._dlatent_noise_in: 0})

    @property
    def noises(self):
        return tflib.run(self._noise_vars)

    @property
    def images_float(self):
        return tflib.run(self._images_float_expr, {self._dlatent_noise_in: 0})

    @property
    def images_uint8(self):
        return tflib.run(self._images_uint8_expr, {self._dlatent_noise_in: 0})

#----------------------------------------------------------------------------

# def project(network_pkl: str, target_fname: str, outdir: str, save_video: bool, seed: int, num_steps: int, initial_noise_factor: float, initial_learning_rate: float, fps: int):

#----------------------------------------------------------------------------

def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

#----------------------------------------------------------------------------

_examples = '''examples:

  python %(prog)s --output_path=out --image_path=targetimg.png \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/ffhq.pkl
'''

#----------------------------------------------------------------------------


def recyclable_part(seed, network_pkl, num_steps):
    # BEGIN RECYCLABLE PORTION 

    time_recyclable_start = time.time()

    # Load networks.
    tflib.init_tf({'rnd.np_random_seed': seed})
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as fp:
        _G, _D, Gs = pickle.load(fp)


    # Initialize projector.
    proj = Projector()
    proj.set_num_steps(num_steps)

    proj.set_network(Gs)

    time_recyclable_end = time.time()
    print('Time to do recyclable / pre-init part took {:.4f} seconds.'.format(time_recyclable_end - time_recyclable_start))

    return proj, Gs

def configurable_part(target_fname, outdir, proj, Gs):
    time_start = time.time()
    FPS = 6

    # Load target image.
    target_pil = PIL.Image.open(target_fname)

    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil= target_pil.convert('RGB')
    target_pil = target_pil.resize((Gs.output_shape[3], Gs.output_shape[2]), PIL.Image.ANTIALIAS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)
    target_float = target_uint8.astype(np.float32).transpose([2, 0, 1]) * (2 / 255) - 1

    # Set target img in Projector.
    proj.start([target_float])

    # Setup output directory.
    os.makedirs(outdir, exist_ok=True)
    target_pil.save(f'{outdir}/target.png')
    writer = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=FPS, codec='libx264', bitrate='16M')

    # Run projector.
    with tqdm.trange(proj.num_steps) as t:
        for step in t:
            assert step == proj.cur_step
            writer.append_data(proj.images_uint8[0])
            dist, loss = proj.step()
            t.set_postfix(dist=f'{dist[0]:.4f}', loss=f'{loss:.2f}')


    # Save results.
    PIL.Image.fromarray(proj.images_uint8[0], 'RGB').save(f'{outdir}/proj.png')

    np.savez(f'{outdir}/dlatents.npz', dlatents=proj.dlatents)
    if writer is not None:
        writer.close()

    time_end = time.time()
    print('Time to do configurable part of projection took {:.4f} seconds.'.format(time_end - time_start))

def parse_json(json_path):
    with open(json_path) as f:
      json_dict = json.load(f)
    return json_dict


def main():

    parser = argparse.ArgumentParser(
        description='Project given image to the latent space of pretrained network pickle.',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--network',     help='Network pickle filename', dest='network_pkl', required=True)
    parser.add_argument('--json_path',      help='File path to json arg file', dest='json_path', required=True)

    args = parser.parse_args()

#### BEGIN RECYCLABLE PART
    network_pkl = args.network_pkl
    num_steps = 80
    seed = 303

    proj, Gs = recyclable_part(seed, network_pkl, num_steps)

#### END RECYCLABLE PART

    while (True):

        if os.path.exists(args.json_path):
            # parse json
            json_dict = parse_json(args.json_path)

            # del json
            print(f'Deleting JSON.')
            os.remove(args.json_path)

            configurable_part(json_dict['input_path'], json_dict['output_path'], proj, Gs)

        else:
            SLEEP_TIME_IN_SECS = 0.05
            print(f'JSON not found. Sleeping {SLEEP_TIME_IN_SECS} seconds.')
            time.sleep(SLEEP_TIME_IN_SECS)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
