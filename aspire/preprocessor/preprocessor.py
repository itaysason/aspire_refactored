import numpy as np
from aspire.preprocessor.phaseflip import phaseflip_star_file
from aspire.preprocessor.downsample import downsample
from aspire.preprocessor.normalize_background import normalize_background
from aspire.preprocessor.prewhiten import prewhiten
from aspire.preprocessor.global_phaseflip import global_phaseflip
import aspire.utils.common as common
import time


def preprocess(star_file, crop_size=-1, downsample_size=89):
    print('Starting phaseflip')
    tic = time.time()
    stack = phaseflip_star_file(star_file)
    toc = time.time()
    s = stack.shape
    print('Finished phaseflip in {} seconds, found {} images with resolution {}'.format(toc - tic, s[0], s[1]))
    if crop_size > 0:
        print('Start cropping')
        tic = time.time()
        stack = common.crop(stack, (-1, crop_size, crop_size))
        toc = time.time()
        print('Finished cropping in {} seconds, from {} to {}'.format(toc - tic, s[1], crop_size))
    else:
        print('Skip cropping')
        crop_size = s[1]
    if downsample_size > 0:
        print('Start downsampling')
        tic = time.time()
        stack = downsample(stack, downsample_size)
        toc = time.time()
        print('Finished downsampling in {} seconds, from {} to {}'.format(toc - tic, crop_size, downsample_size))
    else:
        print('Skip downsampling')
    # Up to this point, the stacks are C aligned, now aligning to matlab (in the future it will stay C aligned)
    stack = np.ascontiguousarray(stack.T)
    print('Start normalizing background')
    stack, _, _ = normalize_background(stack)
    print('Start prewhitening')
    stack = prewhiten(stack)
    print('Start global phaseflip')
    stack = global_phaseflip(stack)
    return stack


