import numpy as np
import aspire.aspire.utils.common as common


def normalize_background(stack, radius=None):
    n = stack.shape[1]
    radius = n // 2 if radius is None else radius
    circle = ~common.disc(n, radius)
    background_pixels = stack[circle]
    mean = np.mean(background_pixels, 0)
    std = np.std(background_pixels, 0, ddof=1)
    stack -= mean
    stack /= std
    return stack, mean, std
