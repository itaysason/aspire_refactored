import mrcfile
import numpy as np
from aspire.common import default_logger


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
        c[k,0] = k  # Index of the image
        c[k,1] = np.std(p[idx],ddof=1) # contrast of the image

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
        The origin of the grid is always in the center, for both odd and even N.
    :param n: Size of the grid.
    :return: Two (2D) arrays x and y with the x and y coordinates of each grid point, respectively.
    """
    p = (n - 1.0) / 2.0
    x, y = np.meshgrid(np.linspace(-p, p, n), np.linspace(-p, p, n))
    return x, y


if __name__ == "__main__":
    stackfile = mrcfile.open('averages_small.mrcs', mode='r')
    print(stackfile.data.shape)
