import numpy as np
import scipy.special as sp


def cryo_syncmatrix_vote(clmatrix, l, rots_ref=None, is_perturbed=0):
    """
    3e-16 error from matlab
    :param clmatrix:
    :param l:
    :param rots_ref:
    :param is_perturbed:
    :return:
    """
    sz = clmatrix.shape
    if len(sz) != 2:
        raise ValueError('clmatrix must be a square matrix')
    if sz[0] != sz[1]:
        raise ValueError('clmatrix must be a square matrix')

    k = sz[0]
    s = np.eye(2 * k)
    a = cryo_syncmatrix_ij_vote(clmatrix, 78 // 2, 92 // 2, np.arange(k), l, rots_ref, is_perturbed)
    for i in range(k - 1):
        stmp = np.zeros((2, 2, k))
        for j in range(i + 1, k):
            stmp[:, :, j] = cryo_syncmatrix_ij_vote(clmatrix, i, j, np.arange(k), l, rots_ref, is_perturbed)

        for j in range(i + 1, k):
            r22 = stmp[:, :, j]
            s[2 * i:2 * (i + 1), 2 * j:2 * (j + 1)] = r22
            s[2 * j:2 * (j + 1), 2 * i:2 * (i + 1)] = r22.T
    return s


def cryo_syncmatrix_ij_vote(clmatrix, i, j, k, l, rots_ref=None, is_perturbed=None):
    tol = 1e-12
    ref = 0 if rots_ref is None else 1

    good_k, _, _ = cryo_vote_ij(clmatrix, l, i, j, k, rots_ref, is_perturbed)

    rs, good_rotations = rotration_eulerangle_vec(clmatrix, i, j, good_k, l)

    if rots_ref is not None:
        reflection_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        raise NotImplementedError

    if len(good_rotations) > 0:
        rk = np.mean(rs, 2)
        tmp_r = rs[:2, :2]
        diff = tmp_r - rk[:2, :2, np.newaxis]
        err = np.linalg.norm(diff) / np.linalg.norm(tmp_r)
        if err > tol:
            pass
    else:
        rk = np.zeros((3, 3))
        if rots_ref is not None:
            raise NotImplementedError

    r22 = rk[:2, :2]
    return r22


def rotration_eulerangle_vec(cl, i, j, good_k, n_theta):
    r = np.zeros((3, 3, len(good_k)))
    if i == j:
        return 0, 0

    tol = 1e-12

    idx1 = cl[good_k, j] - cl[good_k, i]
    idx2 = cl[j, good_k] - cl[j, i]
    idx3 = cl[i, good_k] - cl[i, j]

    a = np.cos(2 * np.pi * idx1 / n_theta)
    b = np.cos(2 * np.pi * idx2 / n_theta)
    c = np.cos(2 * np.pi * idx3 / n_theta)

    cond = 1 + 2 * a * b * c - (np.square(a) + np.square(b) + np.square(c))
    too_small_idx = np.where(cond <= 1.0e-5)[0]
    good_idx = np.where(cond > 1.0e-5)[0]

    a = a[good_idx]
    b = b[good_idx]
    c = c[good_idx]
    idx2 = idx2[good_idx]
    idx3 = idx3[good_idx]
    c_alpha = (a - b * c) / np.sqrt(1 - np.square(b)) / np.sqrt(1 - np.square(c))

    ind1 = np.logical_or(idx3 > n_theta / 2 + tol, np.logical_and(idx3 < -tol, idx3 > -n_theta / 2))
    ind2 = np.logical_or(idx2 > n_theta / 2 + tol, np.logical_and(idx2 < -tol, idx2 > -n_theta / 2))
    c_alpha[np.logical_xor(ind1, ind2)] = -c_alpha[np.logical_xor(ind1, ind2)]

    aa = cl[i, j] * 2 * np.pi / n_theta
    bb = cl[j, i] * 2 * np.pi / n_theta
    alpha = np.arccos(c_alpha)

    ang1 = np.pi - bb
    ang2 = alpha
    ang3 = aa - np.pi
    sa = np.sin(ang1)
    ca = np.cos(ang1)
    sb = np.sin(ang2)
    cb = np.cos(ang2)
    sc = np.sin(ang3)
    cc = np.cos(ang3)

    r[0, 0, good_idx] = cc * ca - sc * cb * sa
    r[0, 1, good_idx] = -cc * sa - sc * cb * ca
    r[0, 2, good_idx] = sc * sb
    r[1, 0, good_idx] = sc * ca + cc * cb * sa
    r[1, 1, good_idx] = -sa * sc + cc * cb * ca
    r[1, 2, good_idx] = -cc * sb
    r[2, 0, good_idx] = sb * sa
    r[2, 1, good_idx] = sb * ca
    r[2, 2, good_idx] = cb

    if len(too_small_idx) > 0:
        r[:, :, too_small_idx] = 0

    return r, good_idx


def cryo_vote_ij(clmatrix, l, i, j, k, rots_ref, is_perturbed):
    ntics = 60
    x = np.linspace(0, 180, ntics, True)
    phis = np.zeros((len(k), 2))
    rejected = np.zeros(len(k))
    idx = 0
    rej_idx = 0
    if i != j and clmatrix[i, j] != -1:
        l_idx12 = clmatrix[i, j]
        l_idx21 = clmatrix[j, i]
        k = k[np.logical_and(np.logical_and(k != i, clmatrix[i, k] != -1), clmatrix[j, k] != -1)]

        l_idx13 = clmatrix[i, k]
        l_idx31 = clmatrix[k, i]
        l_idx23 = clmatrix[j, k]
        l_idx32 = clmatrix[k, j]

        theta1 = (l_idx13 - l_idx12) * 2 * np.pi / l
        theta2 = (l_idx21 - l_idx23) * 2 * np.pi / l
        theta3 = (l_idx32 - l_idx31) * 2 * np.pi / l

        c1 = np.cos(theta1)
        c2 = np.cos(theta2)
        c3 = np.cos(theta3)

        cond = 1 + 2 * c1 * c2 * c3 - (np.square(c1) + np.square(c2) + np.square(c3))

        good_idx = np.where(cond > 1e-5)[0]
        bad_idx = np.where(cond <= 1e-5)[0]

        cos_phi2 = (c3[good_idx] - c1[good_idx] * c2[good_idx]) / (np.sin(theta1[good_idx]) * np.sin(theta2[good_idx]))
        check_idx = np.where(np.abs(cos_phi2) > 1)[0]
        if np.any(np.abs(cos_phi2) - 1 > 1e-12):
            Warning('GCAR:numericalProblem')
        elif len(check_idx) == 0:
            cos_phi2[check_idx] = np.sign(cos_phi2[check_idx])

        phis[:idx + len(good_idx), 0] = cos_phi2
        phis[:idx + len(good_idx), 1] = k[good_idx]
        idx += len(good_idx)

        rejected[: rej_idx + len(bad_idx)] = k[bad_idx]
        rej_idx += len(bad_idx)

    phis = phis[:idx]
    rejected = rejected[:rej_idx]

    good_k = []
    peakh = -1
    alpha = []

    if idx > 0:
        angles = np.arccos(phis[:, 0]) * 180 / np.pi
        sigma = 3.0

        tmp = np.add.outer(np.square(angles), np.square(x))
        h = np.sum(np.exp((np.multiply.outer(2 * angles, x) - tmp) / (2 * sigma ** 2)), 0)
        peak_idx = h.argmax()
        peakh = h[peak_idx]
        idx = np.where(np.abs(angles - x[peak_idx]) < 360 / ntics)[0]
        good_k = phis[idx, 1]
        alpha = phis[idx, 0]

        if rots_ref is not None:
            raise NotImplementedError
    return good_k.astype('int'), peakh, alpha

def cryo_syncmatrix_vote_3n(clmatrix, n_theta, rots_ref=None, is_perturbed=0):
    """
    3e-16 error from matlab
    :param clmatrix:
    :param n_theta:
    :param rots_ref:
    :param is_perturbed:
    :return:
    """
    sz = clmatrix.shape
    if len(sz) != 2:
        raise ValueError('clmatrix must be a square matrix')
    if sz[0] != sz[1]:
        raise ValueError('clmatrix must be a square matrix')

    n = sz[0]
    # s = np.eye(3 * n)

    Rijs = np.zeros((sp.comb(n, 2).astype(int), 3, 3))
    counter = 0
    for i in range(n):
        stmp = np.zeros((3, 3, n))
        for j in range(i + 1, n):
            stmp[:, :, j] = cryo_syncmatrix_ij_vote_3n(clmatrix, i, j, np.arange(n), n_theta, rots_ref, is_perturbed)

        for j in range(i + 1, n):
            Rijs[counter] = stmp[:, :, j]
            counter += 1
            # s[3*i:3*(i+1), 3*j:3*(j+1)] = Rij
            # s[3*j:3*(j+1), 3*i:3*(i+1)] = Rij.T
    return Rijs
    # return s


def cryo_syncmatrix_ij_vote_3n(clmatrix, i, j, k, l, rots_ref=None, is_perturbed=None):
    tol = 1e-12
    ref = 0 if rots_ref is None else 1

    good_k, _, _ = cryo_vote_ij(clmatrix, l, i, j, k, rots_ref, is_perturbed)

    rs, good_rotations = rotratio_eulerangle_vec(clmatrix, i, j, good_k, l)

    if rots_ref is not None:
        reflection_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        raise NotImplementedError

    if len(good_rotations) > 0:
        rk = np.mean(rs, 2)
    else:
        rk = np.zeros((3, 3))
        if rots_ref is not None:
            raise NotImplementedError
    return rk

def rotratio_eulerangle_vec(cl, i, j, good_k, n_theta):
    r = np.zeros((3, 3, len(good_k)))
    if i == j:
        return 0, 0

    tol = 1e-12

    idx1 = cl[good_k, j] - cl[good_k, i]
    idx2 = cl[j, good_k] - cl[j, i]
    idx3 = cl[i, good_k] - cl[i, j]

    a = np.cos(2 * np.pi * idx1 / n_theta)
    b = np.cos(2 * np.pi * idx2 / n_theta)
    c = np.cos(2 * np.pi * idx3 / n_theta)

    cond = 1 + 2 * a * b * c - (np.square(a) + np.square(b) + np.square(c))
    too_small_idx = np.where(cond <= 1.0e-5)[0]
    good_idx = np.where(cond > 1.0e-5)[0]

    a = a[good_idx]
    b = b[good_idx]
    c = c[good_idx]
    idx2 = idx2[good_idx]
    idx3 = idx3[good_idx]
    c_alpha = (a - b * c) / np.sqrt(1 - np.square(b)) / np.sqrt(1 - np.square(c))

    ind1 = np.logical_or(idx3 > n_theta / 2 + tol, np.logical_and(idx3 < -tol, idx3 > -n_theta / 2))
    ind2 = np.logical_or(idx2 > n_theta / 2 + tol, np.logical_and(idx2 < -tol, idx2 > -n_theta / 2))
    c_alpha[np.logical_xor(ind1, ind2)] = -c_alpha[np.logical_xor(ind1, ind2)]

    aa = cl[i, j] * 2 * np.pi / n_theta
    bb = cl[j, i] * 2 * np.pi / n_theta
    alpha = np.arccos(c_alpha)

    ang1 = np.pi - bb
    ang2 = alpha
    ang3 = aa - np.pi
    sa = np.sin(ang1)
    ca = np.cos(ang1)
    sb = np.sin(ang2)
    cb = np.cos(ang2)
    sc = np.sin(ang3)
    cc = np.cos(ang3)

    r[0, 0, good_idx] = cc * ca - sc * cb * sa
    r[0, 1, good_idx] = -cc * sa - sc * cb * ca
    r[0, 2, good_idx] = sc * sb
    r[1, 0, good_idx] = sc * ca + cc * cb * sa
    r[1, 1, good_idx] = -sa * sc + cc * cb * ca
    r[1, 2, good_idx] = -cc * sb
    r[2, 0, good_idx] = sb * sa
    r[2, 1, good_idx] = sb * ca
    r[2, 2, good_idx] = cb

    if len(too_small_idx) > 0:
        r[:, :, too_small_idx] = 0

    return r, good_idx
