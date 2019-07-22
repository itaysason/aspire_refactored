import numpy as np


def global_phaseflip(stack, forceflip=False):
    """ Apply global phase flip to an image stack if needed.

    Check if all images in a stack should be globally phase flipped so that
    the molecule corresponds to brighter pixels and the background corresponds
    to darker pixels. This is done by comparing the mean in a small circle
    around the origin (supposed to correspond to the molecule) with the mean
    of the noise, and making sure that the mean of the molecule is larger.

    Examples:
        >> import mrcfile
        >> stack = mrcfile.open('stack.mrcs')
        >> stack = global_phaseflip_stack(stack)

    :param stack: stack of images to phaseflip if needed
    :param forceflip: force global phase inversion
    :return stack: stack which might be phaseflipped when needed
    :return flipped: True if phase was flipped and False otherwise
    """

    n = stack.shape[0]
    image_center = (n + 1) / 2
    coor_mat_m, coor_mat_n = np.meshgrid(np.arange(1, n + 1), np.arange(1, n + 1))
    distance_from_center = np.sqrt((coor_mat_m - image_center) ** 2 + (coor_mat_n - image_center) ** 2)

    # calculate indices of signal and noise samples assuming molecule is around the center
    signal_indices = distance_from_center < round(n / 4)
    noise_indices = distance_from_center > round(n / 2 * 0.8)

    signal_mean = np.mean(stack[signal_indices], 0)
    noise_mean = np.mean(stack[noise_indices], 0)

    signal_mean = np.mean(signal_mean)
    noise_mean = np.mean(noise_mean)

    flipped = False
    if (signal_mean < noise_mean) or forceflip:
        flipped = True
        stack *= -1

    return stack, flipped
