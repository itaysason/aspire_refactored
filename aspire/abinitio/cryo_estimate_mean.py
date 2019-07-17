import numpy as np
from aspire.utils.common import fill_struct, cfft2, icfft2
from pyfftw.interfaces import numpy_fft
import finufftpy


class DiracBasis:
    # for now doesn't work with mask, probably there is no need for it too.
    def __init__(self, sz, mask=None):
        if mask is None:
            mask = np.ones(sz)

        self.type = 0  # TODO define a constant for this

        self.sz = sz
        self.mask = mask
        self.count = mask.size
        self.sz_prod = np.count_nonzero(sz)

    def evaluate(self, x):
        if x.shape[0] != self.count:
            raise ValueError('First dimension of input must be of size basis.count')

        return x.reshape(self.sz, order='F').copy()

    def expand(self, x):
        if len(x.shape) < len(self.sz) or x.shape[:len(self.sz)] != self.sz:
            raise ValueError('First {} dimension of input must be of size basis.count'.format(len(self.sz)))

        return x.flatten('F').copy()

    def evaluate_t(self, x):
        return self.expand(x)

    def expand_t(self, x):
        return self.expand(x)


def cryo_estimate_mean(im, params, basis=None, mean_est_opt=None):
    """
    1e-5 error from matlab
    :param im:
    :param params:
    :param basis:
    :param mean_est_opt:
    :return:
    """
    resolution = im.shape[1]

    if basis is None:
        basis = DiracBasis((resolution, resolution, resolution))

    mean_est_opt = fill_struct(mean_est_opt, {'precision': 'float64', 'preconditioner': 'circulant'})

    kernel_f = cryo_mean_kernel_f(resolution, params, mean_est_opt)

    precond_kernel_f = []
    if mean_est_opt.preconditioner == 'circulant':
        precond_kernel_f = 1 / circularize_kernel_f(kernel_f)
    elif mean_est_opt.preconditioner != 'none':
        raise ValueError('Invalid preconditioner type')

    def identity(x):
        return x

    mean_est_opt.preconditioner = identity
    im_bp = cryo_mean_backproject(im, params, mean_est_opt)

    mean_est, cg_info = cryo_conj_grad_mean(kernel_f, im_bp, basis, precond_kernel_f, mean_est_opt)
    return mean_est, cg_info


def cryo_conj_grad_mean(kernel_f, im_bp, basis, precond_kernel_f=None, mean_est_opt=None):
    mean_est_opt = fill_struct(mean_est_opt)
    resolution = im_bp.shape[0]
    if len(im_bp.shape) != 3 or im_bp.shape[1] != resolution or im_bp.shape[2] != resolution:
        raise ValueError('im_bp must be as array of size LxLxL')

    def fun(vol_basis):
        return apply_mean_kernel(vol_basis, kernel_f, basis)

    if precond_kernel_f is not None:
        def precond_fun(vol_basis):
            return apply_mean_kernel(vol_basis, precond_kernel_f, basis)

        mean_est_opt.preconditioner = precond_fun

    im_bp_basis = basis.evaluate_t(im_bp)
    mean_est_basis, _, cg_info = conj_grad(fun, im_bp_basis, mean_est_opt)
    mean_est = basis.evaluate(mean_est_basis)
    return mean_est, cg_info


def conj_grad(a_fun, b, cg_opt=None, init=None):

    def identity(input_x):
        return input_x

    cg_opt = fill_struct(cg_opt, {'max_iter': 50, 'verbose': 0, 'iter_callback': [], 'preconditioner': identity,
                                  'rel_tolerance': 1e-15, 'store_iterates': False})
    init = fill_struct(init, {'x': None, 'p': None})
    if init.x is None:
        x = np.zeros(b.shape)
    else:
        x = init.x

    b_norm = np.linalg.norm(b)
    r = b.copy()
    s = cg_opt.preconditioner(r)

    if np.any(x != 0):
        if cg_opt.verbose:
            print('[CG] Calculating initial residual')
        a_x = a_fun(x)
        r = r-a_x
        s = cg_opt.preconditioner(r)
    else:
        a_x = np.zeros(x.shape)

    obj = np.real(np.sum(x.conj() * a_x, 0) - 2 * np.real(np.sum(np.conj(b * x), 0)))

    if init.p is None:
        p = s
    else:
        p = init.p

    info = fill_struct(att_vals={'iter': [0], 'res': [np.linalg.norm(r)], 'obj': [obj]})
    if cg_opt.store_iterates:
        info = fill_struct(info, att_vals={'x': [x], 'r': [r], 'p': [p]})

    if cg_opt.verbose:
        print('[CG] Initialized. Residual: {}. Objective: {}'.format(np.linalg.norm(info.res[0]), np.sum(info.obj[0])))

    if b_norm == 0:
        print('b_norm == 0')
        return

    for i in range(1, cg_opt.max_iter):
        if cg_opt.verbose:
            print('[CG] Applying matrix & preconditioner')

        a_p = a_fun(p)
        old_gamma = np.real(np.sum(s.conj() * r))

        alpha = old_gamma / np.real(np.sum(p.conj() * a_p))
        x += alpha * p
        a_x += alpha * a_p

        r -= alpha * a_p
        s = cg_opt.preconditioner(r)
        new_gamma = np.real(np.sum(r.conj() * s))
        beta = new_gamma / old_gamma
        p *= beta
        p += s

        obj = np.real(np.sum(x.conj() * a_x, 0) - 2 * np.real(np.sum(np.conj(b * x), 0)))
        res = np.linalg.norm(r)
        info.iter.append(i)
        info.res.append(res)
        info.obj.append(obj)
        if cg_opt.store_iterates:
            info.x.append(x)
            info.r.append(r)
            info.p.append(p)

        if cg_opt.verbose:
            print('[CG] Initialized. Residual: {}. Objective: {}'.format(np.linalg.norm(info.res[0]), np.sum(info.obj[0])))

        if np.all(res < b_norm * cg_opt.rel_tolerance):
            break

    # if i == cg_opt.max_iter - 1:
    #     raise Warning('Conjugate gradient reached maximum number of iterations!')
    return x, obj, info


def apply_mean_kernel(vol_basis, kernel_f, basis):
    vol = basis.evaluate(vol_basis)
    vol = cryo_conv_vol(vol, kernel_f)
    vol_basis = basis.evaluate_t(vol)
    return vol_basis


def cryo_conv_vol(x, kernel_f):
    n = x.shape[0]
    n_ker = kernel_f.shape[0]

    if np.any(np.array(x.shape) != n):
        raise ValueError('Volume in `x` must be cubic')

    if np.any(np.array(kernel_f.shape) != n_ker):
        raise ValueError('Convolution kernel in `kernel_f` must be cubic')

    is_singleton = len(x.shape) == 3

    shifted_kernel_f = np.fft.ifftshift(np.fft.ifftshift(np.fft.ifftshift(kernel_f, 0), 1), 2)

    if is_singleton:
        x = numpy_fft.fftn(x, [n_ker] * 3)
    else:
        x = numpy_fft.fft(x, n=n_ker, axis=0)
        x = numpy_fft.fft(x, n=n_ker, axis=1)
        x = numpy_fft.fft(x, n=n_ker, axis=2)

    x *= shifted_kernel_f

    if is_singleton:
        x = numpy_fft.ifftn(x)
        x = x[:n, :n, :n]
    else:
        x = numpy_fft.ifft(x, axis=0)
        x = numpy_fft.ifft(x, axis=1)
        x = numpy_fft.ifft(x, axis=2)

    x = x.real
    return x


def cryo_mean_backproject(im, params, mean_est_opt=None):
    """
    1e-7 error from matlab
    :param im:
    :param params:
    :param mean_est_opt:
    :return:
    """
    mean_est_opt = fill_struct(mean_est_opt, {'precision': 'float64', 'half_pixel': False, 'batch_size': 0})
    if im.shape[0] != im.shape[1] or im.shape[0] == 1 or len(im.shape) != 3:
        raise ValueError('im must be 3 dimensional LxLxn where L > 1')

    resolution = im.shape[1]
    n = im.shape[2]

    if mean_est_opt.batch_size != 0:
        batch_size = mean_est_opt.batch_size
        mean_est_opt.batch_size = 0

        batch_ct = np.ceil(n / batch_size)
        im_bp = np.zeros([2 * resolution] * 3, dtype=mean_est_opt.precision)

        for batch in range(batch_ct):
            start = batch_size * batch
            end = min((batch_size + 1) * batch, n)

            batch_params = subset_params(params, np.arange(start, end))
            batch_im = im[:, :, start:end]
            batch_im_bp = cryo_mean_kernel_f(batch_im, batch_params, mean_est_opt)
            im_bp += (end - start) / n * batch_im_bp

        return im_bp

    if mean_est_opt.precision == 'float32' or mean_est_opt.precision == 'single':
        im = im.astype('float32')

    filter_f = np.einsum('ij, k -> ijk', params.ctf, np.ones(np.count_nonzero(params.ctf_idx)))
    im = im * params.ampl
    im = im_translate(im, -params.shifts)
    im = im_filter(im, filter_f)
    im = im_backproject(im, params.rot_matrices, mean_est_opt.half_pixel)
    im /= n
    return im


def im_backproject(im, rot_matrices, half_pixel=False):
    # TODO - test for even resolution
    resolution = im.shape[1]
    n = im.shape[2]
    if im.shape[0] != resolution:
        raise ValueError('im must be squared - LxLxN')
    if rot_matrices.shape[2] != n:
        raise ValueError('The number of rotation matrices must match the number of images')

    pts_rot = rotated_grids(resolution, rot_matrices, half_pixel)
    pts_rot = pts_rot.reshape((3, -1), order='F')

    if resolution % 2 == 0 and half_pixel:
        grid = np.arange(-resolution / 2, resolution / 2)
        y, x = np.meshgrid(grid, grid)
        phase_shift = 2 * np.pi * (x + y) / (2 * resolution)
        im = np.einsum('ijk, ij -> ijk', im, np.exp(1j * phase_shift))

    im = im.transpose((1, 0, 2))
    im_f = cfft2(im, axes=(0, 1)) / resolution ** 2

    if resolution % 2 == 0:
        if half_pixel:
            grid = np.arange(-resolution / 2, resolution / 2)
            y, x = np.meshgrid(grid, grid)
            phase_shift = 2 * np.pi * (x + y) / (2 * resolution)
            phase_shift += - np.reshape(pts_rot.sum(0), (resolution, resolution, n)) / 2
            im_f = np.einsum('ijk, ij -> ijk', im_f, np.exp(1j * phase_shift))
        else:
            im_f[0] = 0
            im_f[:, 0] = 0

    im_f = im_f.flatten('F')
    vol = anufft3(im_f, pts_rot, [resolution] * 3)
    vol = vol.real
    return vol


def im_filter(im, filter_f):
    n_im = im.shape[2]
    n_filter = filter_f.shape[2]

    if n_filter != 1 and n_filter != n_im:
        raise ValueError('The number of filters must be 1 or match the number of images')
    if im.shape[:2] != filter_f.shape[:2]:
        raise ValueError('The size of the images and filters must match')

    im_f = cfft2(im, axes=(0, 1))
    im_filtered_f = im_f * filter_f
    im_filtered = np.real(icfft2(im_filtered_f, axes=(0, 1)))
    return im_filtered


def im_translate(im, shifts):
    n_im = im.shape[2]
    n_shifts = shifts.shape[1]

    if shifts.shape[0] != 2:
        raise ValueError('Input `shifts` must be of size 2-by-n')

    if n_shifts != 1 and n_shifts != n_im:
        raise ValueError('The number of shifts must be 1 or match the number of images')

    if im.shape[0] != im.shape[1]:
        raise ValueError('Images must be square')

    resolution = im.shape[1]
    grid = np.fft.ifftshift(np.ceil(np.arange(-resolution / 2, resolution / 2)))
    om_y, om_x = np.meshgrid(grid, grid)
    phase_shifts = np.einsum('ij, k -> ijk', om_x, shifts[0]) + np.einsum('ij, k -> ijk', om_y, shifts[1])
    phase_shifts /= resolution

    mult_f = np.exp(-2 * np.pi * 1j * phase_shifts)
    im_f = np.fft.fft2(im, axes=(0, 1))
    im_translated_f = im_f * mult_f
    im_translated = np.real(np.fft.ifft2(im_translated_f, axes=(0, 1)))
    return im_translated


def circularize_kernel_f(kernel_f):
    kernel = mdim_fftshift(np.fft.ifftn(mdim_ifftshift(kernel_f)))

    for dim in range(len(kernel_f.shape)):
        kernel = circularize_kernel_1d(kernel, dim)

    kernel = mdim_fftshift(np.fft.fftn(mdim_ifftshift(kernel)))
    return kernel


def circularize_kernel_1d(kernel, dim):
    sz = kernel.shape
    if dim >= len(sz):
        raise ValueError('dim exceeds kernal dimensions')

    n = sz[dim] // 2

    mult = np.arange(n) / n
    if dim == 0:
        kernel_circ = np.einsum('i, ijk -> ijk', mult, kernel[:n])
    elif dim == 1:
        kernel_circ = np.einsum('j, ijk -> ijk', mult, kernel[:, :n])
    else:
        kernel_circ = np.einsum('k, ijk -> ijk', mult, kernel[:, :, :n])

    mult = np.arange(n, 0, -1) / n
    if dim == 0:
        kernel_circ += np.einsum('i, ijk -> ijk', mult, kernel[n:])
    elif dim == 1:
        kernel_circ += np.einsum('j, ijk -> ijk', mult, kernel[:, n:])
    else:
        kernel_circ += np.einsum('k, ijk -> ijk', mult, kernel[:, :, n:])

    kernel_circ = np.fft.fftshift(kernel_circ, dim)
    return kernel_circ


def rotated_grids(resolution, rot_matrices, half_pixel=False):
    mesh2d = mesh_2d(resolution)

    if resolution % 2 == 0 and half_pixel:
        mesh2d.x += 1 / resolution
        mesh2d.y += 1 / resolution

    num_pts = resolution ** 2
    num_rots = rot_matrices.shape[2]

    pts = np.pi * np.stack((mesh2d.x.flatten('F'), mesh2d.y.flatten('F'), np.zeros(num_pts)))
    pts_rot = np.einsum('ilk, lj -> ijk', rot_matrices, pts)

    pts_rot = pts_rot.reshape((3, resolution, resolution, num_rots), order='F')
    return pts_rot


def cryo_mean_kernel_f(resolution, params, mean_est_opt=None):
    """
    8e-14 error from matlab
    :param resolution:
    :param params:
    :param mean_est_opt:
    :return:
    """
    mean_est_opt = fill_struct(mean_est_opt, {'precision': 'float64', 'half_pixel': False, 'batch_size': 0})
    n = params.rot_matrices.shape[2]

    # TODO debug, might be a problem with the first 2 lines
    if mean_est_opt.batch_size != 0:
        batch_size = mean_est_opt.batch_size
        mean_est_opt.batch_size = 0

        batch_ct = np.ceil(n / batch_size)
        mean_kernel_f = np.zeros([2 * resolution] * 3, dtype=mean_est_opt.precision)

        for batch in range(batch_ct):
            start = batch_size * batch
            end = min((batch_size + 1) * batch, n)

            batch_params = subset_params(params, np.arange(start, end))
            batch_kernel_f = cryo_mean_kernel_f(resolution, batch_params, mean_est_opt)
            mean_kernel_f += (end - start) / n * batch_kernel_f

        return mean_kernel_f

    pts_rot = rotated_grids(resolution, params.rot_matrices, mean_est_opt.half_pixel)
    filt = np.einsum('ij, k -> ijk', np.square(params.ctf), np.square(params.ampl), dtype=mean_est_opt.precision)

    if resolution % 2 == 0 and not mean_est_opt.half_pixel:
        # is it necessary?
        pts_rot = pts_rot[:, 1:, 1:]
        filt = filt[1:, 1:]

    # Reshape inputs into appropriate sizes and apply adjoint NUFFT
    pts_rot = pts_rot.reshape((3, -1), order='F')
    filt = filt.flatten('F')
    mean_kernel = anufft3(filt, pts_rot, [2 * resolution] * 3)
    mean_kernel /= n * resolution ** 2

    # Ensure symmetric kernel
    mean_kernel[0] = 0
    mean_kernel[:, 0] = 0
    mean_kernel[:, :, 0] = 0

    mean_kernel = mean_kernel.copy()
    # Take the Fourier transform since this is what we want to use when convolving
    mean_kernel = np.fft.ifftshift(mean_kernel)
    mean_kernel = np.fft.fftn(mean_kernel)
    mean_kernel = np.fft.fftshift(mean_kernel)
    mean_kernel = np.real(mean_kernel)
    return mean_kernel


def anufft3(vol_f, fourier_pts, sz):
    if len(sz) != 3:
        raise ValueError('sz must be 3')
    if len(fourier_pts.shape) != 2:
        raise ValueError('fourier_pts must be 2D with shape 3x_')
    if fourier_pts.shape[0] != 3:
        raise ValueError('fourier_pts must be 2D with shape 3x_')
    if not fourier_pts.flags.c_contiguous:
        fourier_pts = fourier_pts.copy()
    if not vol_f.flags.c_contiguous:
        vol_f = vol_f.copy()

    x = fourier_pts[0]
    y = fourier_pts[1]
    z = fourier_pts[2]
    isign = 1
    eps = 1e-15
    ms, mt, mu = sz
    f = np.empty(sz, dtype='complex128', order='F')
    finufftpy.nufft3d1(x, y, z, vol_f, isign, eps, ms, mt, mu, f)
    return f.copy()


def mdim_ifftshift(x, dims=None):
    if dims is None:
        dims = np.arange(len(x.shape))

    x = np.fft.ifftshift(x, dims)
    return x


def mdim_fftshift(x, dims=None):
    if dims is None:
        dims = np.arange(len(x.shape))

    x = np.fft.fftshift(x, dims)
    return x


def subset_params(params, ind):
    batch_params = fill_struct()

    params.rot_matrices = params.rot_matrices[:, :, ind]
    params.ctf_idx = params.ctf_idx[:, ind]
    params.ampl = params.ampl[:, ind]
    params.shifts = params.shifts[:, ind]

    return batch_params


def mesh_2d(resolution, inclusive=False):
    if inclusive:
        cons = (resolution - 1) / 2
        grid = np.arange(-cons, cons + 1) / cons
    else:
        cons = resolution / 2
        grid = np.ceil(np.arange(-cons, cons)) / cons

    mesh = fill_struct()
    mesh.y, mesh.x = np.meshgrid(grid, grid)  # reversed from matlab
    mesh.phi, mesh.r, _ = cart2pol(mesh.x, mesh.y)
    return mesh


def cart2pol(x, y, z=None):
    th = np.arctan2(y, x)
    r = np.hypot(x, y)
    return th, r, z
