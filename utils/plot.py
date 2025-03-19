"""
Plotting utilities to visualize training logs.
"""

import imageio
import os
import torchvision.utils as v_utils


def plot_samples_per_epoch(gen_batch, output_dir, epoch, iteration, nsample, name):
    """
    Plot and save output samples per epoch
    """
    fname = f"samples_epoch_{epoch}_{iteration}_{name}.jpg"
    fpath = os.path.join(output_dir, fname)
    nrow = gen_batch.shape[0] // nsample

    image = v_utils.make_grid(gen_batch, nrow=nrow, padding=2, normalize=True)
    # v_utils.save_image(image, fpath)
    return image


def plot_val_samples(gen_batch, output_dir, fname, nrow):
    """
    Plot and dsave output samples for validations
    """
    fpath = os.path.join(output_dir, fname)
    image = v_utils.make_grid(gen_batch, nrow=nrow, padding=2, normalize=True)
    v_utils.save_image(image, fpath)
    return image


def plot_image(img, output_dir, fname):
    """
    img in tensor format
    """

    fpath = os.path.join(output_dir, fname)

    v_utils.save_image(img, fpath, nrow=4, padding=2, normalize=True)
    return imageio.imread(fpath)

