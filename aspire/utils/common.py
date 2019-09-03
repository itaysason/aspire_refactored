import numpy as np
from numpy.fft import fftshift, ifftshift
from pyfftw.interfaces.numpy_fft import fft2, ifft2
from numpy.polynomial.legendre import leggauss
import scipy.special as sp
import finufftpy


def crop(x, out_shape):
    """

    :param x: ndarray of size (N_1,...N_k)
    :param out_shape: iterable of integers of length k. The value in position i is the size we want to cut from the
        center of x in dimension i. If the value is <= 0 then the dimension is left as is
    :return: out: The center of x with size outshape.
    """
    in_shape = np.array(x.shape)
    out_shape = np.array([s if s > 0 else in_shape[i] for i, s in enumerate(out_shape)])
    start_indices = in_shape // 2 - out_shape // 2
    end_indices = start_indices + out_shape
    indexer = tuple([slice(i, j) for (i, j) in zip(start_indices, end_indices)])
    out = x[indexer]
    return out


def downsample(images, out_size, is_stack=True):
    # TODO
    return 0


def estimate_snr(images):
    """
    Estimate signal-noise-ratio for a stack of projections.

    :arg images: stack of projections (between 1 and N projections)
    """

    if len(images.shape) == 2:  # in case of a single projection
        images = images[:, :, None]

    p = images.shape[1]
    n = images.shape[2]

    radius_of_mask = p // 2 - 1

    points_inside_circle = disc(p, r=radius_of_mask, inner=True)
    num_signal_points = np.count_nonzero(points_inside_circle)
    num_noise_points = p * p - num_signal_points

    noise = np.sum(np.var(images[~points_inside_circle], axis=0)) * num_noise_points / (num_noise_points * n - 1)

    signal = np.sum(np.var(images[points_inside_circle], axis=0)) * num_signal_points / (num_signal_points * n - 1)

    signal -= noise

    snr = signal / noise

    return snr, signal, noise


def disc(n, r=None, inner=False):
    """
    Return the points inside the circle of radius=r in a square with side n. if inner is True don't return only the
    strictly inside points.
    :param n: integer, the side of the square
    :param r: The radius of the circle
    :param inner:
    :return: nd array with 0 outside of the circle and 1 inside
    """
    r = n // 2 if r is None else r
    ctr = (n + 1) / 2
    y_axis, x_axis = np.meshgrid(np.arange(1, n + 1), np.arange(1, n + 1))
    radiisq = np.square(x_axis - ctr) + np.square(y_axis - ctr)
    if inner is True:
        return radiisq < r ** 2
    return radiisq <= r ** 2


def cfft2(x, axes=(-1, -2)):
    if len(x.shape) == 2:
        return np.fft.fftshift(np.transpose(np.fft.fft2(np.transpose(np.fft.ifftshift(x)))))
    elif len(x.shape) == 3:
        y = np.fft.ifftshift(x, axes=axes)
        y = np.fft.ifft2(y, axes=axes)
        y = np.fft.fftshift(y, axes=axes)
        return y
    else:
        raise ValueError("x must be 2D or 3D")


def icfft2(x, axes=(-1, -2)):
    if len(x.shape) == 2:
        return np.fft.fftshift(np.transpose(np.fft.ifft2(np.transpose(np.fft.ifftshift(x)))))
    elif len(x.shape) == 3:
        y = np.fft.ifftshift(x, axes=axes)
        y = np.fft.ifft2(y, axes=axes)
        y = np.fft.fftshift(y, axes=axes)
        return y
    else:
        raise ValueError("x must be 2D or 3D")


def icfft(x, axis=0):
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axis), axis=axis), axis)


def fast_cfft2(x, axes=(-1, -2)):
    if len(x.shape) == 2:
        return fftshift(np.transpose(fft2(np.transpose(ifftshift(x)))))
    elif len(x.shape) == 3:
        y = ifftshift(x, axes=axes)
        y = fft2(y, axes=axes)
        y = fftshift(y, axes=axes)
        return y
    else:
        raise ValueError("x must be 2D or 3D")


def fast_icfft2(x, axes=(-1, -2)):
    if len(x.shape) == 2:
        return fftshift(np.transpose(ifft2(np.transpose(ifftshift(x)))))

    elif len(x.shape) == 3:
        y = ifftshift(x, axes=axes)
        y = ifft2(y, axes=axes)
        y = fftshift(y, axes=axes)
        return y

    else:
        raise ValueError("x must be 2D or 3D")


def lgwt(n, a, b):
    """
    Get n leggauss points in interval [a, b]

    :param n: number of points
    :param a: interval starting point
    :param b: interval end point
    :returns SamplePoints(x, w): sample points, weight
    """

    x1, w = leggauss(n)
    m = (b - a) / 2
    c = (a + b) / 2
    x = m * x1 + c
    w = m * w
    x = np.flipud(x)
    return create_struct({'x': x, 'w': w})


def cryo_pft(p, n_r, n_theta):
    """
    Compute the polar Fourier transform of projections with resolution n_r in the radial direction
    and resolution n_theta in the angular direction.
    :param p:
    :param n_r: Number of samples along each ray (in the radial direction).
    :param n_theta: Angular resolution. Number of Fourier rays computed for each projection.
    :return:
    """
    if n_theta % 2 == 1:
        raise ValueError('n_theta must be even')

    n_projs = p.shape[2]
    omega0 = 2 * np.pi / (2 * n_r - 1)
    dtheta = 2 * np.pi / n_theta

    freqs = np.zeros((2, n_r * n_theta // 2))
    for i in range(n_theta // 2):
        freqs[0, i * n_r: (i + 1) * n_r] = np.arange(n_r) * np.sin(i * dtheta)
        freqs[1, i * n_r: (i + 1) * n_r] = np.arange(n_r) * np.cos(i * dtheta)

    freqs *= omega0
    # finufftpy require it to be aligned in fortran order
    pf = np.empty((n_r * n_theta // 2, n_projs), dtype='complex128', order='F')
    finufftpy.nufft2d2many(freqs[0], freqs[1], pf, 1, 1e-15, p)
    pf = pf.reshape((n_r, n_theta // 2, n_projs), order='F')
    pf = np.concatenate((pf, pf.conj()), axis=1).copy()
    return pf, freqs


def fuzzy_mask(n, r0, origin=None):
    if isinstance(n, int):
        n = np.array([n])

    if isinstance(r0, int):
        r0 = np.array([r0])

    k = 1.782 / 2  # Are these constants?

    if origin is None:
        origin = np.floor(n / 2) + 1
        origin = origin.astype('int')
    if len(n) == 1:
        x, y = np.mgrid[1 - origin[0]:n[0] - origin[0] + 1, 1 - origin[0]:n[0] - origin[0] + 1]
    else:
        x, y = np.mgrid[1 - origin[0]:n[0] - origin[0] + 1, 1 - origin[1]:n[1] - origin[1] + 1]

    if len(r0) < 2:
        r = np.sqrt(np.square(x) + np.square(y))
    else:
        r = np.sqrt(np.square(x) + np.square(y * r0[0] / r0[1]))

    m = 0.5 * (1 - sp.erf(k * (r - r0[0])))
    return m


def fill_struct(obj=None, att_vals=None, overwrite=None):
    """
    Fill object with attributes in a dictionary.
    If a struct is not given a new object will be created and filled.
    If the given struct has a field in att_vals, the original field will stay, unless specified otherwise in overwrite.
    att_vals is a dictionary with string keys, and for each key:
    if hasattr(s, key) and key in overwrite:
        pass
    else:
        setattr(s, key, att_vals[key])
    :param obj:
    :param att_vals:
    :param overwrite
    :return:
    """
    # TODO should consider making copy option - i.e that the input won't change
    if obj is None:
        class DisposableObject:
            pass

        obj = DisposableObject()

    if att_vals is None:
        return obj

    if overwrite is None or not overwrite:
        overwrite = []
    if overwrite is True:
        overwrite = list(att_vals.keys())

    for key in att_vals.keys():
        if hasattr(obj, key) and key not in overwrite:
            continue
        else:
            setattr(obj, key, att_vals[key])

    return obj


def create_struct(att_vals=None):
    """
    Creates object
    :param att_vals:
    :return:
    """
    return fill_struct(att_vals=att_vals)
