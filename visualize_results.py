import torch
import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval

from plotting_parameters import *


def get_mask(pad):
    xs = np.linspace(0, 1, 64)
    ys = np.linspace(0, 1, 64)

    XX, YY = np.meshgrid(xs, ys)
    ZZ = np.zeros_like(XX)
    ZZ[pad:-pad, pad:-pad] = 1

    kernel = np.ones((2*pad, 2*pad))
    kernel /= np.sum(kernel)

    newimg = cv.filter2D(ZZ, -1, kernel)
    return newimg


def process_full_image(model, image, size, device):
    image_shape = np.shape(image)

    new_img = np.zeros_like(image).astype(np.float32)
    mask = np.zeros_like(image).astype(np.float32)

    ii = 1
    pad = 2

    Nxtiles = range(0, image_shape[0]-size, int(size/2))
    Nytiles = range(0, image_shape[1]-size, int(size/2))
#    Nxtiles = range(0, image_shape[0]-size, int(size))
#    Nytiles = range(0, image_shape[1]-size, int(size))

    total_tiles = len(Nxtiles) * len(Nytiles)

    for ix in Nxtiles:
        for iy in Nytiles:
            print(f"\rProcessing tile number {ii}/{total_tiles}", end="")


            crop = image[ix:ix+size, iy:iy+size].astype(np.float64)

            #_, _, bkg = get_background(crop, 3)
            #crop -= bkg


            crop -= np.nanmean(image[ix:ix+size, iy:iy+size])
            crop /= np.nanstd(image[ix:ix+size, iy:iy+size])

            cropmedian = np.nanmedian(crop)
            # get and remove the background
            test = model(torch.tensor(crop, dtype=torch.float)[None, :, :].to(device))
            out = test[0].cpu().detach().numpy().astype(np.float32)



            outmedian = np.nanmedian(out)

            out = out-outmedian+cropmedian


            out *= np.nanstd(image[ix:ix+size, iy:iy+size])
            out += np.nanmean(image[ix:ix+size, iy:iy+size])

            #out += bkg

            #if pad == 0:
            mymask = get_mask(10)
            if True:
                new_img[ix:ix+size, iy:iy+size] += out*mymask
                #mask[ix:ix+size, iy:iy+size] += np.ones_like(out)
                mask[ix:ix+size, iy:iy+size] += mymask
            else:
                new_img[ix+pad:ix+size-pad, iy+pad:iy+size-pad] += out[pad:-pad, pad:-pad]
                mask[ix+pad:ix+size-pad, iy+pad:iy+size-pad] += np.ones_like(out[pad:-pad, pad:-pad])
            ii += 1

    new_img /= mask
    fig, ax = plt.subplots()
    ax.imshow(mask)
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 5))
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



