import numpy as np
import aspire.utils.common as common
import scipy.sparse as sps
import scipy.linalg as scl


def initial_classification_fd_update(spca_data, n_nbor, is_rand=False):
    # unpacking spca_data
    print('starting initial_classification')
    coeff = spca_data.coeff.copy()
    freqs = spca_data.freqs.copy()
    eigval = spca_data.eigval.copy()

    n_im = coeff.shape[1]
    coeff[freqs == 0] /= np.sqrt(2)
    # could possibly do it faster
    for i in range(n_im):
        coeff[:, i] /= np.linalg.norm(coeff[:, i])

    coeff[freqs == 0] *= np.sqrt(2)
    print('starting bispec_2drot_large')
    coeff_b, coeff_b_r, _ = bispec_2drot_large(coeff, freqs, eigval)

    concat_coeff = np.concatenate((coeff_b, coeff_b_r), axis=1)

    # TODO find a better implementation to NN, use transpose coeff_b might be faster
    if n_im <= 10000:
        # could use einsum
        corr = np.real(np.dot(np.conjugate(coeff_b[:, :n_im]).T, concat_coeff))
        range_arr = np.arange(n_im)
        corr = corr - sps.csr_matrix((np.ones(n_im), (range_arr, range_arr)),
                                     shape=(n_im, 2 * n_im))
        classes = np.argsort(-corr, axis=1)
        classes = classes[:, :n_nbor]

    else:
        if not is_rand:
            batch_size = 2000
            num_batches = int(np.ceil(1.0 * n_im / batch_size))
            classes = np.zeros((n_im, n_nbor), dtype='int')
            for i in range(num_batches):
                start = i * batch_size
                finish = min((i + 1) * batch_size, n_im)
                corr = np.real(np.dot(np.conjugate(coeff_b[:, start: finish]).T, concat_coeff))
                classes[start: finish] = np.argsort(-corr, axis=1)[:, 1: n_nbor + 1]
                print('processed {}/{} images'.format(finish, n_im))
        else:
            # TODO implement random nn
            batch_size = 2000
            num_batches = int(np.ceil(n_im / batch_size))
            classes = np.zeros((n_im, n_nbor), dtype='int')
            for i in range(num_batches):
                start = i * batch_size
                finish = min((i + 1) * batch_size, n_im)
                corr = np.real(np.dot(np.conjugate(coeff_b[:, start: finish]).T, concat_coeff))
                classes[start: finish] = np.argsort(-corr, axis=1)[:, 1: n_nbor + 1]
                print('processed {}/{} images'.format(finish, n_im))

    classes = np.array(classes)
    max_freq = np.max(freqs)
    cell_coeff = []
    for i in range(max_freq + 1):
        cell_coeff.append(
            np.concatenate((coeff[freqs == i], np.conjugate(coeff[freqs == i])), axis=1))

    # maybe pairs should also be transposed
    pairs = np.stack((classes.flatten('F'), np.tile(np.arange(n_im), n_nbor)), axis=1)
    print('starting rot_align')
    corr, rot = rot_align_fast(max_freq, cell_coeff, pairs)

    rot = rot.reshape((n_im, n_nbor), order='F')
    classes = classes.reshape((n_im, n_nbor), order='F')  # this should already be in that shape
    corr = corr.reshape((n_im, n_nbor), order='F')
    id_corr = np.argsort(-corr, axis=1)
    for i in range(n_im):
        corr[i] = corr[i, id_corr[i]]
        classes[i] = classes[i, id_corr[i]]
        rot[i] = rot[i, id_corr[i]]

    class_refl = np.ceil((classes + 1.0) / n_im).astype('int')
    classes[classes >= n_im] = classes[classes >= n_im] - n_im
    rot[class_refl == 2] = np.mod(rot[class_refl == 2] + 180, 360)
    return classes, class_refl, rot, corr, 0


def bispec_2drot_large(coeff, freqs, eigval):
    alpha = 1 / 3
    freqs_not_zero = freqs != 0
    coeff_norm = np.log(np.power(np.absolute(coeff[freqs_not_zero]), alpha))
    check = coeff_norm == np.float('-inf')
    # should assert if there is a -inf in the coeff norm
    if True in check:
        return 0

    phase = coeff[freqs_not_zero] / np.absolute(coeff[freqs_not_zero])
    phase = np.arctan2(np.imag(phase), np.real(phase))
    eigval = eigval[freqs_not_zero]
    o1, o2 = bispec_operator_1(freqs[freqs_not_zero])

    n = 50000
    m = np.exp(o1 * np.log(np.power(eigval, alpha)))
    p_m = m / m.sum()
    x = np.random.rand(len(m))
    m_id = np.where(x < n * p_m)[0]
    o1 = o1[m_id]
    o2 = o2[m_id]
    m = np.exp(o1 * coeff_norm + 1j * o2 * phase)

    # svd of the reduced bispectrum
    u, s, v = pca_y(m, 300)
    coeff_b = np.einsum('i, ij -> ij', s, np.conj(v.T))
    coeff_b_r = np.conj(u.T).dot(np.conjugate(m))

    coeff_b = coeff_b / np.linalg.norm(coeff_b, axis=0)
    coeff_b_r = coeff_b_r / np.linalg.norm(coeff_b_r, axis=0)
    return coeff_b, coeff_b_r, 0


def bispec_operator_1(freqs):
    max_freq = np.max(freqs)
    count = 0
    for i in range(2, max_freq):
        for j in range(1, min(i, max_freq - i + 1)):
            k = i + j
            id1 = np.where(freqs == i)[0]
            id2 = np.where(freqs == j)[0]
            id3 = np.where(freqs == k)[0]
            nd1 = len(id1)
            nd2 = len(id2)
            nd3 = len(id3)
            count += nd1 * nd2 * nd3

    full_list = np.zeros((count, 3), dtype='int')
    count = 0
    for i in range(2, max_freq):
        for j in range(1, min(i, max_freq - i + 1)):
            k = i + j
            id1 = np.where(freqs == i)[0]
            id2 = np.where(freqs == j)[0]
            id3 = np.where(freqs == k)[0]
            nd1 = len(id1)
            nd2 = len(id2)
            nd3 = len(id3)
            nd = nd1 * nd2 * nd3
            if nd != 0:
                tmp1 = np.tile(id1, nd2)
                tmp2 = np.repeat(id2, nd1)
                tmp = np.stack((tmp1, tmp2), axis=1)
                tmp1 = np.tile(tmp, (nd3, 1))
                tmp2 = np.repeat(id3, nd1 * nd2)
                full_list[count: count + nd, :2] = tmp1
                full_list[count: count + nd, 2] = tmp2
                count += nd

    val = np.ones(full_list.shape)
    val[:, 2] = -1
    n_col = count
    full_list = full_list.flatten('F')
    val = val.flatten('F')
    col = np.tile(np.arange(n_col), 3)
    o1 = sps.csr_matrix((np.ones(len(full_list)), (col, full_list)), shape=(n_col, len(freqs)))
    o2 = sps.csr_matrix((val, (col, full_list)), shape=(n_col, len(freqs)))
    return o1, o2


def rot_align(m, coeff, pairs):
    """
    Reference function that behaves like matlab's function.
    :param m:
    :param coeff:
    :param pairs:
    :return:
    """
    n_theta = 360
    p = pairs.shape[0]
    c = np.zeros((m + 1, p), dtype='complex128')
    m_list = np.arange(1, m + 1)

    max_iter = 100
    precision = 1e-10

    # Find initial points for Newton Raphson
    for i in range(m + 1):
        c[i] = np.einsum('ij, ij -> j', np.conj(coeff[i][:, pairs[:, 0]]), coeff[i][:, pairs[:, 1]])

    c2 = np.flipud(np.conj(c[1:]))
    b = (2 * m + 1) * np.real(common.icfft(np.concatenate((c2, c), axis=0)))
    rot = np.argmax(b, axis=0)
    rot = (rot - m) * n_theta / (2 * m + 1)

    # creating f' and f'' function
    m_list_ang_1j = 1j * m_list * np.pi / 180
    c_for_f_prime_1 = m_list_ang_1j * c[1:].T
    c_for_f_prime_2 = np.square(m_list_ang_1j) * c[1:].T

    def f_prime(x):
        return np.sum(np.real(c_for_f_prime_1 * np.exp(np.outer(x, m_list_ang_1j))), 1)

    def f_prime2(x):
        return np.sum(np.real(c_for_f_prime_2 * np.exp(np.outer(x, m_list_ang_1j))), 1)

    # Finding brackets, x1<x2 such that sign(f(x1)) != sign(f(x2)) and rot = (x1 + x2) / 2
    step_size = 0.5
    x1 = rot.copy()
    x2 = rot.copy()
    bad_indices = np.full(p, True)
    while np.any(bad_indices):
        x1[bad_indices] -= step_size
        x2[bad_indices] += step_size
        f_x1 = f_prime(x1)
        f_x2 = f_prime(x2)
        bad_indices = f_x1 * f_x2 > 0

    # Setting x1, x2 into x_low, x_high such that f(x_low)<f(x_high).
    x_low = x1.copy()
    x_high = x2.copy()
    f_x_low = f_prime(x_low)
    f_x_high = f_prime(x_high)
    x_high_is_low = f_x_high < f_x_low
    tmp = x_low.copy()
    tmp[x_high_is_low] = x_high[x_high_is_low]
    x_high[x_high_is_low] = x_low[x_high_is_low]
    x_low = tmp

    # Handling f(x) = 0 case
    f_x_low = f_prime(x_low)
    f_x_low_0 = f_x_low == 0
    x_high[f_x_low_0] = x_low[f_x_low_0]
    f_x_high = f_prime(x_high)
    f_x_high_0 = f_x_high == 0
    x_low[f_x_high_0] = x_high[f_x_high_0]

    rts = (x_low + x_high) / 2
    dx = np.abs(x_low - x_high)
    dx_old = dx.copy()
    f = f_prime(rts)
    df = f_prime2(rts)
    for _ in range(max_iter):
        bisect_indices = np.bitwise_or(((rts - x_high) * df - f) * ((rts - x_low) * df - f) > 0,
                                       np.abs(2 * f) > np.abs(dx_old * df))
        newton_indices = ~bisect_indices
        dx_old = dx.copy()

        # Handling out of range indices with Bisect step
        dx[bisect_indices] = (x_high[bisect_indices] - x_low[bisect_indices]) / 2
        rts[bisect_indices] = x_low[bisect_indices] + dx[bisect_indices]

        # Handling the rest with newton step
        dx[newton_indices] = f[newton_indices] / df[newton_indices]
        rts[newton_indices] -= dx[newton_indices]

        # Stop criteria
        if np.all(np.abs(dx) < precision):
            break

        # Else update parameters
        f = f_prime(rts)
        df = f_prime2(rts)
        f_negative = f < 0
        x_low[f_negative] = rts[f_negative]
        x_high[~f_negative] = rts[~f_negative]

        # Changing low and high of converged points
        converged = np.abs(dx) < precision
        x_low[converged] = rts[converged]
        x_high[converged] = rts[converged]

        print(np.sum(np.abs(dx) < precision))

    rot = rts
    m_list = np.arange(m + 1)
    m_list_ang = m_list * np.pi / 180
    c *= np.exp(1j * np.outer(m_list_ang, rot))
    corr = (np.real(c[0]) + 2 * np.sum(np.real(c[1:]), axis=0)) / 2

    return corr, rot


def rot_align_fast(m, coeff, pairs):
    """
    Optimized rot_align.
    :param m:
    :param coeff:
    :param pairs:
    :return:
    """
    n_theta = 360
    p = pairs.shape[0]
    c = np.zeros((m + 1, p), dtype='complex128')
    m_list = np.arange(1, m + 1)

    max_iter = 100
    precision = 1e-10

    # Find initial points for Newton Raphson
    for i in range(m + 1):
        c[i] = np.einsum('ij, ij -> j', np.conj(coeff[i][:, pairs[:, 0]]), coeff[i][:, pairs[:, 1]])

    c2 = np.flipud(np.conj(c[1:]))
    b = (2 * m + 1) * np.real(common.icfft(np.concatenate((c2, c), axis=0)))
    rot = np.argmax(b, axis=0)
    rot = (rot - m) * n_theta / (2 * m + 1)

    # creating f' and f'' function
    m_list_ang_1j = 1j * m_list * np.pi / 180
    c_for_f_prime_1 = np.ascontiguousarray(m_list_ang_1j * c[1:].T)
    c_for_f_prime_2 = m_list_ang_1j * c_for_f_prime_1

    def f_prime(x):
        return np.sum(np.real(c_for_f_prime_1 * np.exp(np.outer(x, m_list_ang_1j))), 1)

    # a faster implementation for f_prime and f_prime2 together.
    exp_outer_product_allocation = np.empty((p, len(m_list_ang_1j)), dtype='complex128')
    f_allocation = np.empty(p, dtype='complex128')
    df_allocation = np.empty(p, dtype='complex128')

    def f_prime_prime2(x):
        np.exp(np.outer(x, m_list_ang_1j, exp_outer_product_allocation), out=exp_outer_product_allocation)
        np.einsum('ij, ij -> i', c_for_f_prime_1, exp_outer_product_allocation, out=f_allocation)
        np.einsum('ij, ij -> i', c_for_f_prime_2, exp_outer_product_allocation, out=df_allocation)
        return f_allocation.real, df_allocation.real

    # Finding brackets, x1<x2 such that sign(f(x1)) != sign(f(x2)) and rot = (x1 + x2) / 2
    step_size = 0.5
    x1 = rot.copy()
    x2 = rot.copy()
    g_x1 = c_for_f_prime_1 * np.exp(np.outer(rot, m_list_ang_1j))
    g_x2 = g_x1.copy()
    step_size_vec = np.full(rot.shape, step_size)
    g_step_size = np.exp(np.outer(step_size_vec, m_list_ang_1j))

    bad_indices = np.full(p, True)
    while np.any(bad_indices):
        x1[bad_indices] -= step_size
        x2[bad_indices] += step_size
        g_x1 /= g_step_size
        g_x2 *= g_step_size
        f_x1 = np.sum(np.real(g_x1), 1)
        f_x2 = np.sum(np.real(g_x2), 1)
        bad_indices[f_x1 * f_x2 < 0] = 0

    # This is equivalent to this. I used the fact that the function is similar on similar inputs (only shifted).
    # step_size = 0.5
    # x1 = rot.copy()
    # x2 = rot.copy()
    # bad_indices = np.full(p, True)
    # while np.any(bad_indices):
    #     x1[bad_indices] -= step_size
    #     x2[bad_indices] += step_size
    #     f_x1 = f_prime(x1)
    #     f_x2 = f_prime(x2)
    #     bad_indices = f_x1 * f_x2 > 0

    # Setting x1, x2 into x_low, x_high such that f(x_low)<f(x_high).
    x_low = x1.copy()
    x_high = x2.copy()
    f_x_low, _ = f_prime_prime2(x_low)
    f_x_high, _ = f_prime_prime2(x_high)
    x_high_is_low = f_x_high < f_x_low
    tmp = x_low.copy()
    tmp[x_high_is_low] = x_high[x_high_is_low]
    x_high[x_high_is_low] = x_low[x_high_is_low]
    x_low = tmp

    # handling f(x) = 0 case
    f_x_low, _ = f_prime_prime2(x_low)
    f_x_low_0 = f_x_low == 0
    x_high[f_x_low_0] = x_low[f_x_low_0]
    f_x_high, _ = f_prime(x_high)
    f_x_high_0 = f_x_high == 0
    x_low[f_x_high_0] = x_high[f_x_high_0]

    rts = (x_low + x_high) / 2
    dx = np.abs(x_low - x_high)
    dx_old = dx.copy()
    f, df = f_prime_prime2(rts)
    for _ in range(max_iter):
        bisect_indices = np.bitwise_or(((rts - x_high) * df - f) * ((rts - x_low) * df - f) > 0,
                                       np.abs(2 * f) > np.abs(dx_old * df))
        newton_indices = ~bisect_indices
        dx_old = dx.copy()

        # Handling out of range indices with Bisect step
        dx[bisect_indices] = (x_high[bisect_indices] - x_low[bisect_indices]) / 2
        rts[bisect_indices] = x_low[bisect_indices] + dx[bisect_indices]

        # Handling the rest with newton step
        dx[newton_indices] = f[newton_indices] / df[newton_indices]
        rts[newton_indices] -= dx[newton_indices]

        # Stop criteria
        if np.all(np.abs(dx) < precision):
            print(np.sum(np.abs(dx) < precision))
            break

        # Else update parameters
        f, df = f_prime_prime2(rts)
        f_negative = f < 0
        x_low[f_negative] = rts[f_negative]
        x_high[~f_negative] = rts[~f_negative]

        # Changing low and high of converged points
        converged = np.abs(dx) < precision
        x_low[converged] = rts[converged]
        x_high[converged] = rts[converged]

        print(np.sum(np.abs(dx) < precision))

    rot = rts
    m_list = np.arange(m + 1)
    m_list_ang = m_list * np.pi / 180
    c *= np.exp(1j * np.outer(m_list_ang, rot))
    corr = (np.real(c[0]) + 2 * np.sum(np.real(c[1:]), axis=0)) / 2

    return corr, rot


def pca_y(x, k, num_iters=2):
    m, n = x.shape
    x_conj_transpose = np.ascontiguousarray(np.conj(x.T))

    def operator(mat):
        return x.dot(mat)

    def operator_transpose(mat):
        return x_conj_transpose.dot(mat)

    flag = False
    if m < n:
        flag = True
        operator_transpose, operator = operator, operator_transpose
        m, n = n, m

    ones = np.ones((n, k + 2))
    if x.dtype == np.dtype('complex'):
        h = operator((2 * np.random.random((k + 2, n)).T - ones) + 1j * (2 * np.random.random((k + 2, n)).T - ones))
    else:
        h = operator(2 * np.random.random((k + 2, n)).T - ones)

    f = [h]
    for i in range(num_iters):
        h = operator_transpose(h)
        h = operator(h)
        f.append(h)

    f = np.concatenate(f, axis=1)

    # f has e-16 error, q has e-13
    q, _, _ = scl.qr(f, mode='economic', pivoting=True)
    b = np.conj(operator_transpose(q)).T
    u, s, v = np.linalg.svd(b, full_matrices=False)
    # numpy returns vh, so I am changing it to v so it can match matlab
    v = np.conj(v.T)
    u = np.dot(q, u)
    u = u[:, :k]
    v = v[:, :k]
    s = s[:k]

    if flag:
        u, v = v, u

    return u, s, v
