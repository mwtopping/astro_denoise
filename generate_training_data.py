import numpy as np

# TODO: create two images; one with narrower stars than the other
#  Save gaussian widths and reacreate at 80% or something
# - allow for different x and y sigma?
def make_img(Npoints, size):
    img = np.zeros((size, size))
    fine_img = np.zeros((size, size))
    for ii in range(Npoints):
        s = np.random.uniform(0.001, 0.05)
        x = np.random.uniform(0, 1)
        y = np.random.uniform(0, 1)
        img += gauss2d(x, 
                       y,
                      s,s,size)
        fine_img += gauss2d(x,y,0.85*s,0.85*s,size)


    return img, fine_img

def gauss2d(mux, muy, sigmax, sigmay, size):
    xx = np.linspace(0, 1, size)
    yy = np.linspace(0, 1, size)
    XX, YY = np.meshgrid(xx, yy)

    img = 1. / (2. * np.pi * sigmax * sigmay) * np.exp(-((XX - mux)**2. / (2. * sigmax**2.) + (YY - muy)**2. / (2. * sigmay**2.)))

    img = img - np.nanmedian(img)
#    img /= np.nanstd(img)


    return np.array(img)



def add_noise(img, fine_img, amount=1.0):
    img = img - np.nanmean(img)
    fine_img = fine_img - np.nanmean(fine_img)
    if np.nanstd(img) != 0:
        img /= np.nanstd(img)
        fine_img /= np.nanstd(fine_img)

    nimg = img + np.random.normal(loc=0, scale=amount, size=img.shape)

    return fine_img, nimg

