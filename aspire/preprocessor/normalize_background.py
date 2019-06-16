import numpy as np
import aspire.utils.common as common


def normalize_background(stack, radius=None):
    _, n, num_images = stack.shape
    circle = ~common.disc(n, radius)
    background_pixels = stack[circle]
    mean = np.mean(background_pixels, 0)
    std = np.std(background_pixels, 0, ddof=1)
    stack -= mean
    stack /= std
    return stack, mean, std
