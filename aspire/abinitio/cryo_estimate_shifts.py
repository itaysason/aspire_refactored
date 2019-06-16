import numpy as np
import scipy.sparse as sps


def cryo_estimate_shifts(pf, rotations, max_shift, shift_step=1, memory_factor=10000, shifts_2d_ref=None, verbose=0):
    if memory_factor < 0 or (memory_factor > 1 and memory_factor < 100):
        raise ValueError('subsamplingfactor must be between 0 and 1 or larger than 100')

    n_theta = pf.shape[1] // 2
    n_projs = pf.shape[2]
    pf = np.concatenate((np.flip(pf[1:, n_theta:], 0), pf[:, :n_theta]), 0).copy()

    n_equations_total = int(np.ceil(n_projs * (n_projs - 1) / 2))
    memory_total = n_equations_total * 2 * n_projs * 8

    if memory_factor <= 1:
        n_equations = int(np.ceil(n_projs * (n_projs - 1) * memory_factor / 2))
    else:
        subsampling_factor = (memory_factor * 10 ** 6) / memory_total
        if subsampling_factor < 1:
            n_equations = int(np.ceil(n_projs * (n_projs - 1) * subsampling_factor / 2))
        else:
            n_equations = n_equations_total

    if n_equations < n_projs:
        Warning('Too few equations. Increase memory_factor. Setting n_equations to n_projs')
        n_equations = n_projs

    if n_equations < 2 * n_projs:
        Warning('Number of equations is small. Consider increase memory_factor.')

    shift_i = np.zeros(4 * n_equations + n_equations)
    shift_j = np.zeros(4 * n_equations + n_equations)
    shift_eq = np.zeros(4 * n_equations + n_equations)
    shift_b = np.zeros(n_equations)

    n_shifts = int(np.ceil(2 * max_shift / shift_step + 1))
    r_max = (pf.shape[0] - 1) // 2
    rk = np.arange(-r_max, r_max + 1)
    rk2 = rk[:r_max]
    shift_phases = np.exp(
        np.outer(-2 * np.pi * 1j * rk2 / (2 * r_max + 1), np.arange(-max_shift, -max_shift + n_shifts * shift_step)))

    h = np.sqrt(np.abs(rk)) * np.exp(-np.square(rk) / (2 * (r_max / 4) ** 2))

    d_theta = np.pi / n_theta

    idx_i = []
    idx_j = []
    for i in range(n_projs):
        tmp_j = range(i + 1, n_projs)
        idx_i.extend([i] * len(tmp_j))
        idx_j.extend(tmp_j)
    idx_i = np.array(idx_i, dtype='int')
    idx_j = np.array(idx_j, dtype='int')
    rp = np.random.choice(np.arange(len(idx_j)), size=n_equations, replace=False)

    # might be able to vectorize this
    for shift_eq_idx in range(n_equations):
        i = idx_i[rp[shift_eq_idx]]
        j = idx_j[rp[shift_eq_idx]]

        r_i = rotations[:, :, i]
        r_j = rotations[:, :, j]
        c_ij, c_ji = common_line_r(r_i.T, r_j.T, 2 * n_theta)

        if c_ij >= n_theta:
            c_ij -= n_theta
            c_ji -= n_theta
        if c_ji < 0:
            c_ji += 2 * n_theta

        c_ij = int(c_ij)
        c_ji = int(c_ji)
        is_pf_j_flipped = 0
        if c_ji < n_theta:
            pf_j = pf[:, c_ji, j].copy()
        else:
            pf_j = pf[:, c_ji - n_theta, j].copy()
            is_pf_j_flipped = 1
        pf_i = pf[:, c_ij, i].copy()

        pf_i *= h
        pf_i[r_max - 1:r_max + 2] = 0
        pf_i /= np.linalg.norm(pf_i)
        pf_i = pf_i[:r_max]

        pf_j *= h
        pf_j[r_max - 1:r_max + 2] = 0
        pf_j /= np.linalg.norm(pf_j)
        pf_j = pf_j[:r_max]

        pf_i_flipped = np.conj(pf_i)
        pf_i_stack = np.einsum('i, ij -> ij', pf_i, shift_phases)
        pf_i_flipped_stack = np.einsum('i, ij -> ij', pf_i_flipped, shift_phases)

        c1 = 2 * np.real(np.dot(np.conj(pf_i_stack.T), pf_j))
        c2 = 2 * np.real(np.dot(np.conj(pf_i_flipped_stack.T), pf_j))

        sidx1 = np.argmax(c1)
        sidx2 = np.argmax(c2)

        if c1[sidx1] > c2[sidx2]:
            dx = -max_shift + sidx1 * shift_step
        else:
            dx = -max_shift + sidx2 * shift_step

        idx = np.arange(4 * shift_eq_idx, 4 * shift_eq_idx + 4)
        shift_alpha = c_ij * d_theta
        shift_beta = c_ji * d_theta
        shift_i[idx] = shift_eq_idx
        shift_j[idx] = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
        shift_b[shift_eq_idx] = dx

        # check with compare cl
        if not is_pf_j_flipped:
            shift_eq[idx] = [np.sin(shift_alpha), np.cos(shift_alpha), -np.sin(shift_beta), -np.cos(shift_beta)]
        else:
            shift_beta -= np.pi
            shift_eq[idx] = [-np.sin(shift_alpha), -np.cos(shift_alpha), -np.sin(shift_beta), -np.cos(shift_beta)]

    t = 4 * n_equations
    shift_eq[t: t + n_equations] = shift_b
    shift_i[t: t + n_equations] = np.arange(n_equations)
    shift_j[t: t + n_equations] = 2 * n_projs
    tmp = np.where(shift_eq != 0)[0]
    shift_eq = shift_eq[tmp]
    shift_i = shift_i[tmp]
    shift_j = shift_j[tmp]
    shift_equations = sps.csr_matrix((shift_eq, (shift_i, shift_j)), shape=(n_equations, 2 * n_projs + 1))

    est_shifts = np.linalg.lstsq(shift_equations[:, :-1].todense(), shift_b)[0]
    est_shifts = est_shifts.reshape((2, n_projs), order='F')

    if shifts_2d_ref is not None:
        raise NotImplementedError

    if verbose != 0:
        raise NotImplementedError

    return est_shifts, shift_equations


def common_line_r(r1, r2, l):
    ut = np.dot(r2, r1.T)
    alpha_ij = np.arctan2(ut[2, 0], -ut[2, 1]) + np.pi
    alpha_ji = np.arctan2(-ut[0, 2], ut[1, 2]) + np.pi

    l_ij = alpha_ij * l / (2 * np.pi)
    l_ji = alpha_ji * l / (2 * np.pi)

    l_ij = np.mod(np.round(l_ij), l)
    l_ji = np.mod(np.round(l_ji), l)
    return l_ij, l_ji
