import numpy as np
import matplotlib.pyplot as plt


# function to select the edge of an image and fit a plane to the resulting boundary
def get_background(data, padding=1):
    xsize, ysize = np.shape(data)
    xs = np.linspace(0, 1, xsize)
    ys = np.linspace(0, 1, ysize)
    XX, YY = np.meshgrid(xs, ys)

    data_mask = data.copy()
    data_mask[padding:-padding, padding:-padding] = 0
    mask_inds = data_mask != 0

    xflat = XX[mask_inds].flatten()
    yflat = YY[mask_inds].flatten()
    zflat = data[mask_inds].flatten()
   
    A = np.matrix([[np.sum(xflat*xflat), np.sum(xflat*yflat), np.sum(xflat)],
                   [np.sum(xflat*yflat), np.sum(yflat*yflat), np.sum(yflat)],
                   [np.sum(xflat), np.sum(yflat), np.sum(np.ones_like(xflat))]])
    
    B = np.matrix([np.sum(xflat*zflat), np.sum(yflat*zflat), np.sum(zflat)]).T
    
    res = np.linalg.solve(A, B)
    
    background = float(res[0])*XX+float(res[1])*YY+float(res[2])

    return XX, YY, background



# for testing
if __name__ == "__main__":
    xs = np.linspace(0, 1, 64)
    ys = np.linspace(0, 1, 64)
    xinds = np.arange(64) 
    yinds = np.arange(64)
    
    inds = np.where( np.logical_and(xinds < 63, xinds > 1))
    print(inds)
    
    
    print(np.shape(xs[~1:63]))
    
    
    XX, YY = np.meshgrid(xs, ys)
    
    ZZ = XX + YY
    ZZ += np.random.normal(loc=0, scale=0.1, size=ZZ.shape)
   
    ax = plt.figure().add_subplot(projection='3d')
    
    _, _, background = get_background(ZZ, 2)
    
    ax.scatter(XX, YY, ZZ, alpha=0.5, color='black')
    ax.plot_surface(XX, YY, background)
    plt.show()
