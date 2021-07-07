import numpy as np
from aspire.aspire.utils.common import cryo_pft, create_struct, fuzzy_mask
from aspire.aspire.abinitio.cryo_clmatrix import cryo_clmatrix_cpu
from aspire.aspire.abinitio.cryo_estimate_mean import cryo_estimate_mean
from aspire.aspire.abinitio.cryo_syncmatrix_vote import cryo_syncmatrix_vote
from aspire.aspire.abinitio.cryo_sync_rotations import cryo_sync_rotations
from aspire.aspire.abinitio.cryo_estimate_shifts import cryo_estimate_shifts
import time


def cryo_abinitio_c1_worker(stack, algo, n_theta=360, n_r=0.5, max_shift=0.15, shift_step=1):
    resolution = stack.shape[1]
    num_projections = stack.shape[2]
    #
    n_r = int(np.ceil(n_r * resolution))
    max_shift = int(np.ceil(max_shift * resolution))

    mask_radius = resolution * 0.45
    # mask_radius is of the form xxx.5
    if mask_radius * 2 == int(mask_radius * 2):
        mask_radius = np.ceil(mask_radius)
    # mask is not of the form xxx.5
    else:
        mask_radius = int(round(mask_radius))

    # mask projections
    center = (resolution + 1) / 2
    m = fuzzy_mask(resolution, mask_radius, origin=(center, center))
    masked_projs = stack.copy()
    masked_projs = masked_projs.transpose((2, 0, 1))
    masked_projs *= m
    masked_projs = masked_projs.transpose((1, 2, 0)).copy()

    # compute polar fourier transform
    pf, _ = cryo_pft(masked_projs, n_r, n_theta)

    # find common lines from projections
    print('Finding common lines between pairs of imagges with maximum shift of {} pixels and shift step of {}'.format(
        max_shift, shift_step))
    tic = time.time()
    clstack, _, _, _, _ = cryo_clmatrix_cpu(pf, num_projections, max_shift, shift_step)
    toc = time.time()
    print('Finished in {} seconds'.format(toc - tic))

    if algo == 2:
        s = cryo_syncmatrix_vote(clstack, n_theta)
        rotations = cryo_sync_rotations(s)
    else:
        raise NotImplementedError('algo currently support only "2"!')

    est_shifts, _ = cryo_estimate_shifts(pf, rotations, max_shift, shift_step)

    # reconstruct downsampled volume with no CTF correction
    n = stack.shape[1]
    params = create_struct({'rot_matrices': rotations, 'ctf': np.ones((n, n)), 'ampl': np.ones(num_projections),
                            'ctf_idx': np.array([True] * num_projections), 'shifts': est_shifts})

    print('Estimating mean')
    tic = time.time()
    v1, _ = cryo_estimate_mean(stack, params)
    toc = time.time()
    print('Finished in {} seconds'.format(toc - tic))
    v1 = v1.real
    return v1
