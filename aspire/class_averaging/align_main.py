import numpy as np
import aspire.utils.common as common
import pyfftw
from tqdm import tqdm
from aspire.common import *


def align_main(data, angle, class_vdm, refl, spca_data, k, max_shifts, list_recon, tmpdir, use_em):
    data = data.swapaxes(0, 2)
    data = data.swapaxes(1, 2)
    data = np.ascontiguousarray(data)
    resolution = data.shape[1]

    if class_vdm.shape[1] < k:
        # raise error
        pass

    shifts = np.zeros((len(list_recon), k + 1), dtype='complex128')
    corr = np.zeros((len(list_recon), k + 1), dtype='complex128')
    norm_variance = np.zeros(len(list_recon))

    m = np.fix(resolution * 1.0 / 2)
    omega_x, omega_y = np.mgrid[-m:m + 1, -m:m + 1]
    omega_x = -2 * np.pi * omega_x / resolution
    omega_y = -2 * np.pi * omega_y / resolution
    omega_x = omega_x.flatten('F')
    omega_y = omega_y.flatten('F')
    a = np.arange(-max_shifts, max_shifts + 1)
    num = len(a)
    a1 = np.tile(a, num)
    a2 = np.repeat(a, num)
    shifts_list = np.column_stack((a1, a2))

    phase = np.exp(1j * (np.outer(a1, omega_x) + np.outer(a2, omega_y)))
    conj_phase = np.conj(phase)

    angle = np.round(-angle).astype('int')
    angle[angle < 0] += 360

    angle[angle == 360] = 0
    m = []
    for i in range(1, 360):
        m.append(fast_rotate_precomp(resolution, resolution, i))

    n = resolution // 2
    r = spca_data.r
    coeff = spca_data.coeff
    eig_im = spca_data.eig_im
    freqs = spca_data.freqs

    mean_im = np.dot(spca_data.fn0, spca_data.mean)
    output = np.zeros(data.shape)

    # pre allocating stuff
    images = np.zeros((k + 1, resolution, resolution), dtype='float64')
    images2 = np.zeros((k + 1, resolution, resolution), dtype='complex128')
    tmp_alloc = np.zeros((resolution, resolution), dtype='complex128')
    tmp_alloc2 = np.zeros((resolution, resolution), dtype='complex128')
    pf_images = np.zeros((resolution * resolution, k + 1), dtype='complex128')
    pf2 = np.zeros(conj_phase.shape, dtype='complex128')
    c = np.zeros((conj_phase.shape[0], k + 1), dtype='complex128')
    var = np.zeros(resolution * resolution, dtype='float64')
    mean = np.zeros(resolution * resolution, dtype='complex128')
    pf_images_shift = np.zeros((resolution * resolution, k + 1), dtype='complex128')
    tmps, plans = get_fast_rotate_vars(resolution)

    angle_j = np.zeros((k + 1), dtype='int')
    refl_j = np.ones((k + 1), dtype='int')
    index = np.zeros((k + 1), dtype='int')
    import time
    rotate_time = 0
    mult_time = 0
    cfft_time = 0
    multiply_time = 0
    dot_time = 0
    rest_time = 0

    pbar = tqdm(total=len(list_recon), disable=(default_logger.getEffectiveLevel() != logging.INFO),
                desc="Computing averages", leave=True)
    for j in range(len(list_recon)):
        default_logger.debug('Averaging image {} out of {}'.format(j, len(list_recon)))
        angle_j[1:] = angle[list_recon[j], :k]
        refl_j[1:] = refl[list_recon[j], :k]
        index[1:] = class_vdm[list_recon[j], :k]
        index[0] = list_recon[j]

        for i in range(k + 1):
            if refl_j[i] == 2:
                images[i] = np.flipud(data[index[i]])
            else:
                images[i] = data[index[i]]

        tic0 = time.time()
        # 2610 sec for 9000 images, 1021 for matlab
        for i in range(k + 1):
            if angle_j[i] != 0:
                fast_rotate_image(images[i], angle_j[i], tmps, plans, m[angle_j[i] - 1])
        tic1 = time.time()

        # 190 sec for 9000 images, 103 for matlab
        # "Denoise" current image by projecting it on the eigen-images. All neighbors of the current image
        # are aligned against this denoised image.
        tmp = np.dot(eig_im[:, freqs == 0], coeff[freqs == 0, list_recon[j]]) + 2 * np.real(
            np.dot(eig_im[:, freqs != 0], coeff[freqs != 0, list_recon[j]])) + mean_im
        tic2 = time.time()

        # 1170 sec for 9000 images, 375 for matlab
        tmp_alloc[n - r:n + r, n - r:n + r] = np.reshape(tmp, (2 * r, 2 * r), 'F')

        pf1 = common.cfft2(tmp_alloc).flatten('F')
        for i in range(k + 1):
            images2[i] = common.cfft2(images[i])
        tic3 = time.time()

        # 651 sec for 9000 images, 261 for matlab
        # Compute all shifts of image j
        pf_images[:] = images2.reshape((k + 1, resolution * resolution), order='F').T
        np.multiply(conj_phase, np.conj(pf1), out=pf2)
        tic4 = time.time()

        # 313 sec for 9000 images, 233 for matlab
        # Compute correlations between image j and all its shifts and its neighboring images
        np.dot(pf2, pf_images, out=c)
        tic5 = time.time()

        # 307 sec for 9000 images, 100 for matlab
        # Estimate shifts to re-align neighbors with image j.
        ind = np.lexsort((np.angle(c), np.abs(c)), axis=0)[-1]
        ind_for_c = ind, np.arange(len(ind))
        corr[j] = c[ind_for_c]

        # Re-align neighbors with image j.
        np.multiply(pf_images, conj_phase[ind].T, out=pf_images_shift)
        np.var(pf_images_shift, 1, ddof=1, out=var)
        norm_variance[j] = np.linalg.norm(var)
        np.mean(pf_images_shift, axis=1, out=mean)
        tmp_alloc2[:] = np.reshape(mean, (resolution, resolution), 'F')

        output[j] = np.real(common.icfft2(tmp_alloc2))
        shifts[j] = -shifts_list[ind, 0] - 1j * shifts_list[ind, 1]
        tic6 = time.time()

        rotate_time += tic1 - tic0
        mult_time += tic2 - tic1
        cfft_time += tic3 - tic2
        multiply_time += tic4 - tic3
        dot_time += tic5 - tic4
        rest_time += tic6 - tic5

        pbar.update(1)

    pbar.close()

    default_logger.debug(f'Timeings:')
    default_logger.debug(f'\tRotating images (rotate_time) {rotate_time:.2f} sec')
    default_logger.debug(f'\tDenoising current image (mult_time) {mult_time:.2f} sec')
    default_logger.debug(f'\tFFT neighbors (cfft_time) {cfft_time:.2f} sec')
    default_logger.debug(f'\tComputing shifts of image j (multiply_time) {multiply_time:.2f} sec')
    default_logger.debug(f'\tComputing correlations (dot_time) {dot_time}.2f sec')
    default_logger.debug(f'\tCompute average image (rest_time) {rest_time}.2f sec')
    output = output.swapaxes(1, 2)
    output = output.swapaxes(0, 2)
    output = np.ascontiguousarray(output)

    return shifts, corr, output, norm_variance


def fast_rotate_precomp(szx, szy, phi):
    phi, mult90 = adjust_rotate(phi)

    phi = np.pi * phi / 180
    phi = -phi

    if szy % 2:
        cy = (szy + 1) // 2
        sy = 0
    else:
        cy = szy // 2 + 1
        sy = 0.5

    if szx % 2:
        cx = (szx + 1) // 2
        sx = 0
    else:
        cx = szx // 2 + 1
        sx = 0.5

    my = np.zeros((szy, szx), dtype='complex128')
    r = np.arange(cy)
    r_t = np.arange(szy, cy, -1) - 1
    u = (1 - np.cos(phi)) / np.sin(phi + np.finfo(float).eps)
    alpha1 = 2 * np.pi * 1j * r / szy
    for x in range(szx):
        ux = u * (x + 1 - cx + sx)
        my[r, x] = np.exp(alpha1 * ux)
        my[r_t, x] = np.conj(my[1: cy - 2 * sy, x])

    my = my.T

    mx = np.zeros((szx, szy), dtype='complex128')
    r = np.arange(cx)
    r_t = np.arange(szx, cx, -1) - 1
    u = -np.sin(phi)
    alpha2 = 2 * np.pi * 1j * r / szx
    for y in range(szy):
        uy = u * (y + 1 - cy + sy)
        mx[r, y] = np.exp(alpha2 * uy)
        mx[r_t, y] = np.conj(mx[1: cx - 2 * sx, y])

    # because I am using real fft I take only part of mx and my
    return common.create_struct({'phi': phi, 'mx': mx[:szx // 2 + 1].copy(), 'my': my[:, :szy // 2 + 1].copy(),
                                 'mult90': mult90})


def adjust_rotate(phi):
    phi = phi % 360
    mult90 = 0
    phi2 = phi

    if 45 <= phi < 90:
        mult90 = 1
        phi2 = -(90 - phi)
    elif 90 <= phi < 135:
        mult90 = 1
        phi2 = phi - 90
    elif 135 <= phi < 180:
        mult90 = 2
        phi2 = -(180 - phi)
    elif 180 <= phi < 225:
        mult90 = 2
        phi2 = phi - 180
    elif 215 <= phi < 270:
        mult90 = 3
        phi2 = -(270 - phi)
    elif 270 <= phi < 315:
        mult90 = 3
        phi2 = phi - 270
    elif 315 <= phi < 360:
        mult90 = 0
        phi2 = phi - 360
    return phi2, mult90


def get_fast_rotate_vars(resolution):
    tmp1 = np.empty((resolution, resolution // 2 + 1), dtype='complex128')
    tmp2 = np.empty((resolution // 2 + 1, resolution), dtype='complex128')
    tmps = tmp1, tmp2
    tmp01 = pyfftw.empty_aligned(tmp1.shape, tmp1.dtype)
    tmp02 = pyfftw.empty_aligned(tmp2.shape, tmp1.dtype)
    tmp03 = pyfftw.empty_aligned((resolution, resolution), 'float64')
    flags = ('FFTW_MEASURE', 'FFTW_UNALIGNED')
    plans = [pyfftw.FFTW(tmp03, tmp01, flags=flags),
             pyfftw.FFTW(tmp01, tmp03, direction='FFTW_BACKWARD', flags=flags),
             pyfftw.FFTW(tmp03, tmp02, axes=(0,), flags=flags),
             pyfftw.FFTW(tmp02, tmp03, axes=(0,), direction='FFTW_BACKWARD', flags=flags)]

    return tmps, plans


def fast_rotate_image(image, phi, tmps=None, plans=None, m=None):
    """
    Could make it faster without the flag 'FFTW_UNALIGNED' if I could make
    :param image:
    :param phi:
    :param tmps:
    :param plans:
    :param m:
    :return:
    """
    szx, szy = image.shape

    if m is None:
        m = fast_rotate_precomp(szx, szy, phi)

    mx = m.mx
    my = m.my

    image[:] = np.rot90(image, m.mult90)

    if tmps is None:
        const_size0 = image.shape[0] // 2 + 1
        const_size1 = image.shape[1] // 2 + 1
        tmp1 = np.empty((len(image), const_size1), dtype='complex128')
        tmp2 = np.empty((const_size0, len(image)), dtype='complex128')
    else:
        tmp1 = tmps[0]
        tmp2 = tmps[1]
    if plans is None:
        tmp01 = pyfftw.empty_aligned(tmp1.shape, tmp1.dtype)
        tmp02 = pyfftw.empty_aligned(tmp2.shape, tmp1.dtype)
        tmp03 = pyfftw.empty_aligned(image.shape, image.dtype)
        plans = [pyfftw.FFTW(tmp03, tmp01), pyfftw.FFTW(tmp01, tmp03, direction='FFTW_BACKWARD'),
                 pyfftw.FFTW(tmp03, tmp02, axes=(0,)), pyfftw.FFTW(tmp02, tmp03, axes=(0,), direction='FFTW_BACKWARD')]

    # first pass
    plan = plans[0]
    plan(image, tmp1)
    tmp1 *= my
    plan = plans[1]
    plan(tmp1, image)

    # second pass
    plan = plans[2]
    plan(image, tmp2)
    tmp2 *= mx
    plan = plans[3]
    plan(tmp2, image)

    # first pass
    plan = plans[0]
    plan(image, tmp1)
    tmp1 *= my
    plan = plans[1]
    plan(tmp1, image)
