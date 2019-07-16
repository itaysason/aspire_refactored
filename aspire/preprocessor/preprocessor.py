import numpy as np
from aspire.preprocessor.phaseflip import phaseflip_star_file
from aspire.preprocessor.downsample import downsample
from aspire.preprocessor.normalize_background import normalize_background
from aspire.preprocessor.prewhiten import prewhiten
import aspire.preprocessor.global_phaseflip
import aspire.utils.common as common
from aspire.common import *
import time


def preprocess(star_file, pixel_size=None, crop_size=-1, downsample_size=89, verbose=0):
    use_crop = crop_size > 0
    use_downsample = downsample_size > 0
    # flag to indicate not to transform back in phaseflip and to to transform in downsample
    flag = use_downsample and not use_crop
    default_logger.info('Starting phaseflip')
    stack = phaseflip_star_file(star_file, pixel_size, flag, verbose)
    s = stack.shape
    default_logger.info('Finished phaseflip.')
    if use_crop:
        default_logger.info('Start cropping')
        # TODO - out of core version
        stack = common.crop(stack, (-1, crop_size, crop_size))
        default_logger.info(f'Finished cropping from {s[1]}x{s[2]} to {crop_size}x{crop_size}')
    else:
        default_logger.info('Skip cropping')
        crop_size = s[1]
    if use_downsample > 0:
        default_logger.info('Start downsampling')
        tic = time.time()
        stack = downsample(stack, downsample_size, flag, verbose)
        toc = time.time()
        default_logger.info(f'Finished downsampling from {crop_size}x{crop_size} to {downsample_size}x{downsample_size}')
    else:
        default_logger.info('Skip downsampling')

    # Up to this point, the stacks are C aligned, now aligning to matlab (in the future it will stay C aligned)
    default_logger.debug('Changing the stack to matlab align (temporary)')
    stack = np.ascontiguousarray(stack.T)

    default_logger.info('Start normalizing background')
    stack, _, _ = normalize_background(stack, stack.shape[1] * 45 // 100)
    default_logger.info('Start prewhitening')
    stack = prewhiten(stack, verbose)
    default_logger.info('Start global phaseflip')
    stack, _ = aspire.preprocessor.global_phaseflip.global_phaseflip(stack)
    return stack


