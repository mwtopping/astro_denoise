import torch
import numpy as np

import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval

from plotting_parameters import *


def process_full_image(model, image, size, device):
    image_shape = np.shape(image)

    new_img = np.zeros_like(image).astype(np.float32)
    mask = np.zeros_like(image).astype(np.float32)

    ii = 1

    Nxtiles = range(0, image_shape[0]-size, int(size/2))
    Nytiles = range(0, image_shape[1]-size, int(size/2))
    total_tiles = len(Nxtiles) * len(Nytiles)

    for ix in Nxtiles:
        for iy in Nytiles:
            print(f"\rProcessing tile number {ii}/{total_tiles}", end="")

            crop = image[ix:ix+size, iy:iy+size].astype(np.float64)

            crop -= np.nanmean(image[ix:ix+size, iy:iy+size])
            crop /= np.nanstd(image[ix:ix+size, iy:iy+size])

            cropmedian = np.nanmedian(crop)
            test = model(torch.tensor(crop, dtype=torch.float)[None, :, :].to(device))
            out = test[0].cpu().detach().numpy().astype(np.float32)
            outmedian = np.nanmedian(out)

            out = out-outmedian+cropmedian

            out *= np.nanstd(image[ix:ix+size, iy:iy+size])
            out += np.nanmean(image[ix:ix+size, iy:iy+size])

            pad = 2

            new_img[ix+pad:ix+size-pad, iy+pad:iy+size-pad] += out[pad:-pad, pad:-pad]
            mask[ix+pad:ix+size-pad, iy+pad:iy+size-pad] += np.ones_like(out[pad:-pad, pad:-pad])
            ii += 1

    new_img /= mask
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
    limits = ZScaleInterval().get_limits(image)

    # show varios versions of the image
    ax[0].imshow(image, vmin=limits[0], vmax=limits[1], cmap='Greys_r')
    ax[1].imshow(new_img, vmin=limits[0], vmax=limits[1], cmap='Greys_r')
    ax[2].imshow((0.3*image+0.7*new_img), vmin=limits[0], vmax=limits[1], cmap='Greys_r')

    # plotting parameters
    ax[0].set_ylabel("Pixel")
    for a in ax:
        a.set_xlabel("Pixel")
    ax[0].set_title("Original Image")
    ax[1].set_title("Denoised Image")
    ax[2].set_title("Final Blended Image")



    return



def test_image(model, image_filename, tile_size=64, device=torch.device("cpu")):
    model.eval()
    data = fits.getdata(image_filename)
    process_full_image(model, data, tile_size, device)
    model.train()
    plt.show()



