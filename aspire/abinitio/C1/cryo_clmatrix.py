import numpy as np
import scipy.sparse as sps
import time
from aspire.common import *
from tqdm import tqdm


def cryo_clmatrix_cpu_ref(pf, nk=None, verbose=1, max_shift=15, shift_step=1, map_filter_radius=0, ref_clmatrix=0, ref_shifts_2d=0):
    n_projs = pf.shape[2]
    n_shifts = int(np.ceil(2 * max_shift / shift_step + 1))
    n_theta = pf.shape[1]
    if n_theta % 2 == 1:
        raise ValueError('n_theta must be even')
    n_theta = n_theta // 2

    pf = np.concatenate((np.flip(pf[1:, n_theta:], 0), pf[:, :n_theta]), 0).copy()

    found_ref_clmatrix = 0
    if not np.isscalar(ref_clmatrix):
        found_ref_clmatrix = 1

    found_ref_shifts = 0
    if not np.isscalar(ref_shifts_2d):
        found_ref_shifts = 1

    verbose_plot_shifts = 0
    verbose_detailed_debugging = 0
    verbose_progress = 0

    # Allocate variables
    clstack = np.zeros((n_projs, n_projs)) - 1
    corrstack = np.zeros((n_projs, n_projs))
    clstack_mask = np.zeros((n_projs, n_projs))
    refcorr = np.zeros((n_projs, n_projs))
    thetha_diff = np.zeros((n_projs, n_projs))

    # Allocate variables used for shift estimation
    shifts_1d = np.zeros((n_projs, n_projs))
    ref_shifts_1d = np.zeros((n_projs, n_projs))
    shifts_estimation_error = np.zeros((n_projs, n_projs))
    shift_i = np.zeros(4 * n_projs * nk)
    shift_j = np.zeros(4 * n_projs * nk)
    shift_eq = np.zeros(4 * n_projs * nk)
    shift_equations_map = np.zeros((n_projs, n_projs))
    shift_equation_idx = 0
    shift_b = np.zeros(n_projs * (n_projs - 1) // 2)
    dtheta = np.pi / n_theta

    # Debugging handles and variables - not implemented
    pass

    # search for common lines between pairs of projections
    r_max = int((pf.shape[0] - 1) / 2)
    rk = np.arange(-r_max, r_max + 1)
    h = np.sqrt(np.abs(rk)) * np.exp(-np.square(rk) / (2 * np.square(r_max / 4)))

    pf3 = np.empty(pf.shape, dtype=pf.dtype)
    np.einsum('ijk, i -> ijk', pf, h, out=pf3)
    pf3[r_max - 1:r_max + 2] = 0
    pf3 /= np.linalg.norm(pf3, axis=0)

    rk2 = rk[:r_max]
    for i in range(n_projs):
        n2 = min(n_projs - i, nk)

        if n_projs - i - 1 == 0:
            subset_k2 = []
        else:
            subset_k2 = np.sort(np.random.choice(n_projs - i - 1, n2 - 1, replace=False) + i + 1)

        proj1 = pf3[:, :, i]
        p1 = proj1[:r_max].T
        p1_flipped = np.conj(p1)

        if np.linalg.norm(proj1[r_max]) > 1e-13:
            raise ValueError('DC component of projection is not zero.')

        for j in subset_k2:
            proj2 = pf3[:, :, j]
            p2 = proj2[:r_max]

            if np.linalg.norm(proj2[r_max]) > 1e-13:
                raise ValueError('DC component of projection is not zero.')

            if verbose_plot_shifts and found_ref_clmatrix:
                raise NotImplementedError

            tic = time.time()
            for shift in range(-max_shift, n_shifts, shift_step):
                shift_phases = np.exp(-2 * np.pi * 1j * rk2 * shift / (2 * r_max + 1))
                p1_shifted = shift_phases * p1
                p1_shifted_flipped = shift_phases * p1_flipped
                c1 = 2 * np.real(np.dot(p1_shifted.conj(), p2))
                c2 = 2 * np.real(np.dot(p1_shifted_flipped.conj(), p2))
                c = np.concatenate((c1, c2), 1)

                if map_filter_radius > 0:
                    raise NotImplementedError
                    # c = cryo_average_clmap(c, map_filter_radius)

                sidx = c.argmax()
                cl1, cl2 = np.unravel_index(sidx, c.shape)
                sval = c[cl1, cl2]
                improved_correlation = 0

                if sval > corrstack[i, j]:
                    clstack[i, j] = cl1
                    clstack[j, i] = cl2
                    corrstack[i, j] = sval
                    shifts_1d[i, j] = shift
                    improved_correlation = 1

                if verbose_detailed_debugging and found_ref_clmatrix and found_ref_shifts:
                    raise NotImplementedError

                if verbose_plot_shifts and improved_correlation:
                    raise NotImplementedError

                if verbose_detailed_debugging:
                    raise NotImplementedError

                if verbose_detailed_debugging:
                    raise NotImplementedError

            toc = time.time()
            # Create a shift equation for the projections pair (i, j).
            idx = np.arange(4 * shift_equation_idx, 4 * shift_equation_idx + 4)
            shift_alpha = clstack[i, j] * dtheta
            shift_beta = clstack[j, i] * dtheta
            shift_i[idx] = shift_equation_idx
            shift_j[idx] = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
            shift_b[shift_equation_idx] = shifts_1d[i, j]

            # Compute the coefficients of the current equation.
            if shift_beta < np.pi:
                shift_eq[idx] = [np.sin(shift_alpha), np.cos(shift_alpha), -np.sin(shift_beta), -np.cos(shift_beta)]
            else:
                shift_beta -= np.pi
                shift_eq[idx] = [-np.sin(shift_alpha), -np.cos(shift_alpha), -np.sin(shift_beta), -np.cos(shift_beta)]

            shift_equations_map[i, j] = shift_equation_idx
            print(i, j, shift_equation_idx, corrstack[i, j], toc - tic)
            shift_equation_idx += 1

            if verbose_progress:
                raise NotImplementedError

    if verbose_detailed_debugging and found_ref_clmatrix:
        raise NotImplementedError

    tmp = np.where(corrstack != 0)
    corrstack[tmp] = 1 - corrstack[tmp]
    l = 4 * shift_equation_idx
    shift_eq[l: l + shift_equation_idx] = shift_b
    shift_i[l: l + shift_equation_idx] = np.arange(shift_equation_idx)
    shift_j[l: l + shift_equation_idx] = 2 * n_projs
    tmp = np.where(shift_eq != 0)[0]
    shift_eq = shift_eq[tmp]
    shift_i = shift_i[tmp]
    shift_j = shift_j[tmp]
    l += shift_equation_idx
    shift_equations = sps.csr_matrix((shift_eq, (shift_i, shift_j)), shape=(shift_equation_idx, 2 * n_projs + 1))

    if verbose_detailed_debugging:
        raise NotImplementedError

    return clstack, corrstack, shift_equations, shift_equations_map, clstack_mask


def cryo_clmatrix_cpu(pf, nk=None, max_shift=15, shift_step=1):
    n_projs = pf.shape[2]
    n_shifts = int(np.ceil(2 * max_shift / shift_step + 1))
    n_theta = pf.shape[1]
    if nk is None:
        nk = n_projs
    if n_theta % 2 == 1:
        raise ValueError('n_theta must be even')
    n_theta = n_theta // 2

    pf = np.concatenate((np.flip(pf[1:, n_theta:], 0), pf[:, :n_theta]), 0).copy()

    # Allocate variables
    clstack = np.zeros((n_projs, n_projs)) - 1
    corrstack = np.zeros((n_projs, n_projs))
    clstack_mask = np.zeros((n_projs, n_projs))

    # Allocate variables used for shift estimation
    shifts_1d = np.zeros((n_projs, n_projs))
    shift_i = np.zeros(4 * n_projs * nk)
    shift_j = np.zeros(4 * n_projs * nk)
    shift_eq = np.zeros(4 * n_projs * nk)
    shift_equations_map = np.zeros((n_projs, n_projs))
    shift_equation_idx = 0
    shift_b = np.zeros(n_projs * (n_projs - 1) // 2)
    dtheta = np.pi / n_theta

    # search for common lines between pairs of projections
    r_max = int((pf.shape[0] - 1) / 2)
    rk = np.arange(-r_max, r_max + 1)
    h = np.sqrt(np.abs(rk)) * np.exp(-np.square(rk) / (2 * np.square(r_max / 4)))

    pf3 = np.empty(pf.shape, dtype=pf.dtype)
    np.einsum('ijk, i -> ijk', pf, h, out=pf3)
    pf3[r_max - 1:r_max + 2] = 0
    pf3 /= np.linalg.norm(pf3, axis=0)
    pf3 = pf3[:r_max]

    pf3 = pf3.transpose((2, 0, 1)).copy()
    pf3_transposed_real = np.real(pf3.transpose((0, 2, 1))).copy()
    pf3_transposed_imag = np.imag(pf3.transpose((0, 2, 1))).copy()
    rk2 = rk[:r_max]

    all_shift_phases = np.zeros((n_shifts, r_max), 'complex128')
    shifts = np.array([-max_shift + i * shift_step for i in range(n_shifts)], dtype='int')
    for i in range(n_shifts):
        shift = shifts[i]
        all_shift_phases[i] = np.exp(-2 * np.pi * 1j * rk2 * shift / (2 * r_max + 1))

    stack_p2_shifted_flipped = np.zeros((r_max, n_shifts * n_theta), pf3.dtype)
    pbar=tqdm(total=int(n_projs*(n_projs-1)/2), disable=(default_logger.getEffectiveLevel() != logging.INFO), desc="Processed image pairs", leave=True)
    default_logger.debug(f"Compute correlation between pairs of images")
    for j in range(n_projs - 1, 0, -1):
        p2_flipped = np.conj(pf3[j])
        for k in range(n_shifts):
            curr_p2_shifted_flipped = (all_shift_phases[k] * p2_flipped.T).T
            stack_p2_shifted_flipped[:, k * n_theta:(k + 1) * n_theta] = curr_p2_shifted_flipped

        stack_p2_shifted_flipped_real = np.real(stack_p2_shifted_flipped)
        stack_p2_shifted_flipped_imag = np.imag(stack_p2_shifted_flipped)
        for i in range(j):
            p1_real = pf3_transposed_real[i]
            p1_imag = pf3_transposed_imag[i]

            tic = time.time()

            part1_stack = np.dot(p1_real, stack_p2_shifted_flipped_real)
            part2_stack = np.dot(p1_imag, stack_p2_shifted_flipped_imag)

            c_max = part1_stack + np.abs(part2_stack)
            sidx = np.argmax(c_max)
            cl1, shift_idx, cl2 = np.unravel_index(sidx, (n_theta, n_shifts, n_theta))
            tmp_idx = shift_idx * n_theta + cl2
            sval = c_max[cl1, tmp_idx]
            if part2_stack[cl1, tmp_idx] > 0:
                cl2_addition = n_theta
            else:
                cl2_addition = 0
            clstack[i, j] = cl1
            clstack[j, i] = cl2 + cl2_addition
            corrstack[i, j] = 2 * sval
            shifts_1d[i, j] = -max_shift + shift_idx * shift_step

            toc = time.time()
            default_logger.debug(f"Pair ({i},{j}): cl=({clstack[i, j]},{clstack[j, i]}), "
                                 f"corr={corrstack[i, j]:4.3f}, time={toc-tic:5.4f} sec")
            pbar.update(1)

    pbar.close()

    for i in range(n_projs):
        for j in range(i + 1, n_projs):

            idx = np.arange(4 * shift_equation_idx, 4 * shift_equation_idx + 4)
            shift_alpha = clstack[i, j] * dtheta
            shift_beta = clstack[j, i] * dtheta
            shift_i[idx] = shift_equation_idx
            shift_j[idx] = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
            shift_b[shift_equation_idx] = shifts_1d[i, j]

            # Compute the coefficients of the current equation.
            if shift_beta < np.pi:
                shift_eq[idx] = [np.sin(shift_alpha), np.cos(shift_alpha), -np.sin(shift_beta), -np.cos(shift_beta)]
            else:
                shift_beta -= np.pi
                shift_eq[idx] = [-np.sin(shift_alpha), -np.cos(shift_alpha), -np.sin(shift_beta), -np.cos(shift_beta)]

            shift_equations_map[i, j] = shift_equation_idx
            shift_equation_idx += 1

    tmp = np.where(corrstack != 0)
    corrstack[tmp] = 1 - corrstack[tmp]
    l = 4 * shift_equation_idx
    shift_eq[l: l + shift_equation_idx] = shift_b
    shift_i[l: l + shift_equation_idx] = np.arange(shift_equation_idx)
    shift_j[l: l + shift_equation_idx] = 2 * n_projs
    tmp = np.where(shift_eq != 0)[0]
    shift_eq = shift_eq[tmp]
    shift_i = shift_i[tmp]
    shift_j = shift_j[tmp]
    l += shift_equation_idx
    shift_equations = sps.csr_matrix((shift_eq, (shift_i, shift_j)), shape=(shift_equation_idx, 2 * n_projs + 1))

    return clstack, corrstack, shift_equations, shift_equations_map, clstack_mask


def cryo_clmatrix_cpu_pystyle(npf, max_shift, shift_step):
    """
    A wrapper function for cryo_clmatrix_cpu which expects and returns arrays in python style
    (i.e. image index is the first)
    :param npf: an m-by-n_theta-by-nr array of Fourier transformed projection images
    :param max_shift: the maximum 2d shift to apply
    :param shift_step:
    :return:
    """
    n_images = len(npf)
    max_shift_1d = np.ceil(2 * np.sqrt(2) * max_shift)
    npf = np.transpose(npf, axes=(2, 1, 0))
    return cryo_clmatrix_cpu(npf, n_images, max_shift_1d, shift_step)
    # return cryo_clmatrix_cpu(npf, n_images, param, max_shift_1d, shift_step)
