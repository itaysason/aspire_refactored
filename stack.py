import numpy as np
from aspire.common import default_logger
from aspire.utils.common import cfft2


def contrast(instack, radius=None):
    """
        Compute the contrast of each image in a stack.
        Only pixels within the given radius around the center of each image are used for computing the contrast (to
        eliminate noise pixels at the "corners"). The contrast of an image is defined as the standard deviation of
        the pixels within radius r of the image.


        :param instack: Three dimensional array with first index corresponding to the slice number.
        :param radius: Only pixels within the given radius will be used to computing the contrast.
                       For images of size nxn the default value is floor(n/2) pixels.

        :return: An array with the contrast of each image.
        """

    if instack.ndim != 3:
        raise ValueError(f"Input stack must be KxLxL stack but given {instack.shape}")

    n = instack.shape[1]
    if n != instack.shape[2]:
        raise ValueError("Images must be square")

    if radius is None:
        radius = np.floor(n/2)

    default_logger.debug(f'radius={radius}')

    radii = cart2rad(n)
    idx = np.where(radii <= radius)

    c = np.zeros((instack.shape[0], 2), instack.dtype)

    for k in range(instack.shape[0]):
        p = instack[k, :, :]
        c[k, 0] = k     # Index of the image
        c[k, 1] = np.std(p[idx], ddof=1)    # contrast of the image

    return c


def cart2rad(n):
    """ Compute the radii corresponding to the points of a Cartesian grid of size nxn points, relative to the center
        of the grid.

    :param n: Size of the grid.
    :return:  An nxn array with the radius corresponding to each grid point.
    """

    n = np.floor(n)
    x, y = image_grid(n)
    r = np.sqrt(np.square(x) + np.square(y))
    return r


def image_grid(n):
    """ Return the coordinates of Cartesian points in an nxn grid centered around the origin.
        The origin of the grid is always in the center, for both odd and even n.
    :param n: Size of the grid.
    :return: Two (2D) arrays x and y with the x and y coordinates of each grid point, respectively.
    """
    p = (n - 1.0) / 2.0
    x, y = np.meshgrid(np.linspace(-p, p, n), np.linspace(-p, p, n))
    return x, y


def sort_by_bandpass(projs, lc=0.05, hc=0.2):
    """ Sort images by their energy in a band of frequencies. Return
        the indices of the projections sorted by their energy in a 
        band of frequencies between lc (low cutoff) and hc (high cutoff).
        Both lc and hc are between 0 and 0.5.

    XXX images must have odd side? Check.
    
    :param projs: Stack of input images. First dimension is slice number     
    :param lc: lower frequency cutoff (between 0 and 0.5)
    :param hc: high frequency cutoff (between 0 and 0.5)
    :return idx: Indices of sorted images
    :return val_sorted: Energy in the band (lc, hc) sorted from high to low. 
            vals_sorted[i] is the energy for image idx[i].
    """

    if projs.ndim != 3:
        raise ValueError(f"Input stack must be KxLxL stack but given {projs.shape}")

    # To convert a Python stack to matlab used permute(projs,[2 3 1]) or projs.T in Python
    fp = cfft2(projs)
    im_side = (projs.shape[1] - 1) / 2
    x, y = image_grid(projs.shape[1])
    idx1 = np.where(np.multiply(x, x) + np.multiply(y, y) < (im_side * lc) ** 2)
    idx2 = np.where(np.multiply(x, x) + np.multiply(y, y) > (im_side * hc) ** 2)

    vals = np.zeros((fp.shape[0], 1), np.float64)
    for k in range(vals.size):
        img = fp[k, :, :]
        img[idx2] = 0
        if np.linalg.norm(img) > 1.0e-6:
            img = np.subtract(img, np.mean(img))
            img = np.divide(img, np.linalg.norm(img))
            img[idx1] = 0
            v = np.sum(np.power(np.abs(img), 2))
        else:
            v = -1

        vals[k] = v

    idx = vals[:, 0].argsort()  # Find the indices that sort vals in ascending order
    idx = idx[:: -1]    # Flip the indices to descending order
    vals_sorted = vals[idx, :]  # Sort vals in descending order

    c = np.zeros((projs.shape[0], 2), projs.dtype)
    c[:, 0] = idx
    c[:, 1] = vals_sorted.flatten()
    return c
