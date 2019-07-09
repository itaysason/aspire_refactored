import aspire.utils.common as common
import numpy as np
from pyfftw.interfaces.numpy_fft import fft2, ifft2
from tqdm import tqdm
from aspire.common import *


def downsample(stack, n, mask=None, stack_in_fourier=False,  verbose=0):
    """ Use Fourier methods to change the sample interval and/or aspect ratio
        of any dimensions of the input image 'img'. If the optional argument
        stack is set to True, then the *first* dimension of 'img' is interpreted as the index of
        each image in the stack. The size argument side is an integer, the size of the
        output images.  Let the size of a stack
        of 2D images 'img' be n1 x n1 x k.  The size of the output will be side x side x k.

        If the optional mask argument is given, this is used as the
        zero-centered Fourier mask for the re-sampling. The size of mask should
        be the same as the output image size. For example for downsampling an
        n0 x n0 image with a 0.9 x nyquist filter, do the following:
        msk = fuzzymask(n,2,.45*n,.05*n)
        out = downsample(img, n, 0, msk)
        The size of the mask must be the size of output. The optional fx output
        argument is the padded or cropped, masked, FT of in, with zero
        frequency at the origin.

         :param verbose: Verbosity level (0: silent, 1: progress, 2: info, 3: debug).
    """

    default_logger.debug(f'Input stack of size {stack.shape}')
    size_in = np.square(stack.shape[1])
    size_out = np.square(n)

    mask = 1 if mask is None else mask
    num_images = stack.shape[0]
    output = np.zeros((num_images, n, n), dtype='float32')
    default_logger.debug(f'Output stack of size {output.shape}')

    images_batches = np.array_split(np.arange(num_images), 500)
    default_logger.debug(f'Using {len(images_batches)} batches of size {stack[images_batches[0]].shape}')

    pbar = tqdm(total=num_images, disable=(verbose != 1))
    for batch in images_batches:
        curr_batch = np.array(stack[batch])
        curr_batch = curr_batch if stack_in_fourier else fft2(curr_batch)
        fx = common.crop(np.fft.fftshift(curr_batch, axes=(-2, -1)), (-1, n, n)) * mask
        output[batch] = ifft2(np.fft.ifftshift(fx, axes=(-2, -1))) * (size_out / size_in)
        default_logger.info(f'Processed {batch[-1] + 1}/{num_images} images')
        pbar.update(curr_batch.shape[0])
    return output
