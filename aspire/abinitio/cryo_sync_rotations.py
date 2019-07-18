import numpy as np
import scipy.sparse.linalg as spsl
from aspire.common import default_logger

def cryo_sync_rotations(s, rots_ref=None, verbose=0):
    """
    3e-14 err from matlab
    :param s:
    :param rots_ref:
    :param verbose:
    :return:
    """
    tol = 1e-14
    ref = 0 if rots_ref is None else 1

    sz = s.shape
    if len(sz) != 2:
        raise ValueError('clmatrix must be a square matrix')
    if sz[0] != sz[1]:
        raise ValueError('clmatrix must be a square matrix')
    if sz[0] % 2 == 1:
        raise ValueError('clmatrix must be a square matrix of size 2Kx2K')

    k = sz[0] // 2

    d, v = spsl.eigs(s, 10)
    d = np.real(d)
    sort_idx = np.argsort(-d)

    default_logger.info("Top eigenvalues of synchronization matrix:")
    default_logger.info(f"{d[sort_idx]}")

    v = np.real(v[:, sort_idx[:3]])
    v1 = v[:2*k:2].T.copy()
    v2 = v[1:2*k:2].T.copy()

    equations = np.zeros((3*k, 9))
    for i in range(3):
        for j in range(3):
            equations[0::3, 3*i+j] = v1[i] * v1[j]
            equations[1::3, 3*i+j] = v2[i] * v2[j]
            equations[2::3, 3*i+j] = v1[i] * v2[j]
    truncated_equations = equations[:, [0, 1, 2, 4, 5, 8]]

    b = np.ones(3 * k)
    b[2::3] = 0

    ata_vec = np.linalg.lstsq(truncated_equations, b)[0]
    ata = np.zeros((3, 3))
    ata[0, 0] = ata_vec[0]
    ata[0, 1] = ata_vec[1]
    ata[0, 2] = ata_vec[2]
    ata[1, 0] = ata_vec[1]
    ata[1, 1] = ata_vec[3]
    ata[1, 2] = ata_vec[4]
    ata[2, 0] = ata_vec[2]
    ata[2, 1] = ata_vec[4]
    ata[2, 2] = ata_vec[5]

    # numpy returns lower, matlab upper so I use transpose
    a = np.linalg.cholesky(ata).T

    r1 = np.dot(a, v1)
    r2 = np.dot(a, v2)
    r3 = np.cross(r1, r2, axis=0)

    rotations = np.empty((k, 3, 3))
    rotations[:, :, 0] = r1.T
    rotations[:, :, 1] = r2.T
    rotations[:, :, 2] = r3.T
    u, _, v = np.linalg.svd(rotations)
    np.einsum('ijk, ikl -> ijl', u, v, out=rotations)
    rotations = rotations.transpose((1, 2, 0)).copy()

    return rotations
