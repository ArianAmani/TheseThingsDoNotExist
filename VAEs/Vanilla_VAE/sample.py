import tensorflow as tf
import os
import numpy as np


def sample(vae, num_samples=10, device='cpu'):
    if device == 'gpu':
        with tf.device('/device:GPU:0'):
            samples = vae.sample(num_samples)

    elif device == 'cpu':
        samples = vae.sample(num_samples)

    return samples # (num_samples, img_size, img_size, 3) pixels in range [0, 1]