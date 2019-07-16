import click
import numpy as np
from aspire.common import *
import  aspire.utils.common
from aspire.preprocessor.phaseflip import phaseflip_star_file
from aspire.preprocessor.downsample import downsample
from aspire.preprocessor.normalize_background import normalize_background
from aspire.preprocessor.global_phaseflip import global_phaseflip
from aspire.preprocessor.prewhiten import prewhiten

from aspire.preprocessor.preprocessor import preprocess
from aspire.class_averaging.class_averaging import class_averaging
from aspire.abinitio.cryo_abinitio_c1_worker import cryo_abinitio_c1_worker
from aspire.utils.read_write import *
import stack

np.random.seed(1137)

# CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group()
@click.option('-v', '--verbosity', type=click.IntRange(0, 3), default=0, help='Verbosity level (0: silent, 1: progress, 2: debug).',show_default=True)
@click.option('--logfile', default=None, type=click.Path(exists=False), help='Filename of log file for log messages.')
@click.pass_context
def cli(ctx, verbosity, logfile):
    """\b
    \033[1;33m ASPIRE - Algorithms for Single Particle Reconstruction \033[0m
    \b
    To view the help message of a command, simply type:
    $ python aspire.py <cmd> -h
    \b
    To view the full docs, please visit
    https://aspire-python.readthedocs.io
    """

    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below
    ctx.ensure_object(dict)

    ctx.obj['VERBOSITY'] = verbosity
    ctx.obj['LOGFILE'] = logfile

    configure_logger(default_logger, logfile, verbosity)
    return


@cli.group('stack', short_help='Process 2D stack')
@click.pass_context
#   Bug notice:
#   Putting required=True on the above two options breaks the help on sub-commands.
#   See https://github.com/pallets/click/issues/814.
#   For now, check manually that files were given.
def stack_cmds(ctx):
    """\b
        Two-dimensional processing of a stack of images.
    """

    return


@stack_cmds.command('phaseflip', short_help='Apply phase-flipping')
@click.option('--star-file',type=click.Path(exists=True), required=True, help='STAR file with CTF data.')
@click.option('-o', '--output', type=click.Path(exists=False),  help='Output stack name.')
@click.option('--pixA', 'pixA', type=float, default=None, help=',Pixel size in Angstroms (if missing from STAR file).',
              show_default=True)
@click.pass_context
def phaseflip_cmd(ctx, star_file, output, pixA):
    """\b
        Apply phase-flipping to a stack of images.
    """

    default_logger.debug('Starting phaseflip')
    stack=phaseflip_star_file(star_file, pixA, return_in_fourier=False, verbose=ctx.obj['VERBOSITY'])
    stack = np.ascontiguousarray(stack.T)
    default_logger.info(f'Saving MRC file {output}')
    write_mrc(output, stack)
    default_logger.debug('Done phaseflip')

    return


@stack_cmds.command('crop', short_help='Crop images')
@click.option('-i', '--input', type=click.Path(exists=True),  help='Input stack name.')
@click.option('-o', '--output', type=click.Path(exists=False),  help='Output stack name.')
@click.option('--new-size', type=int, required=True, default=89, help='Size of cropped images (in pixels).',
              show_default=True)
def crop_cmd(input, output, new_size):
    """\b
        Crop a stack of images to a given size.
    """

    default_logger.debug('Starting crop')
    default_logger.debug(f'Reading MRC file {input}')
    stack=read_mrc(input)
    stack = np.ascontiguousarray(stack.T)
    default_logger.debug(f'stack of size {stack.shape}')
    default_logger.debug(f'Cropping from size {stack.shape[1]}x{stack.shape[2]} to size {new_size}x{new_size}')
    stack =  aspire.utils.common.crop(stack, (-1, new_size, new_size))
    stack = np.ascontiguousarray(stack.T)
    default_logger.info(f'Saving MRC file {output}')
    write_mrc(output, stack)
    default_logger.debug('Done crop')

    return


@stack_cmds.command('downsample', short_help='Downsample images')
@click.option('-i', '--input', type=click.Path(exists=True),  help='Input stack name.')
@click.option('-o', '--output', type=click.Path(exists=False),  help='Output stack name.')
@click.option('--new-size', type=int, required=True, default=89, help='Size of downsampled images (in pixels).',
              show_default=True)
@click.pass_context
def downsample_cmd(ctx, input, output, new_size):
    """\b
        Downsample a stack of images to a given size.
    """

    default_logger.debug('Starting downsample')
    default_logger.debug(f'Reading MRC file {input}')
    stack=read_mrc(input)
    stack = np.ascontiguousarray(stack.T)
    default_logger.debug(f'stack of size {stack.shape}')
    default_logger.debug(f'Downsampling from size {stack.shape[1]}x{stack.shape[2]} to size {new_size}x{new_size}')
    stack = aspire.preprocessor.downsample.downsample(stack, new_size, stack_in_fourier=False,  verbose=ctx.obj['VERBOSITY'])
    stack = np.ascontiguousarray(stack.T)
    default_logger.info(f'Saving MRC file {output}')
    write_mrc(output, stack)
    default_logger.debug('Done downsample')

    return


@stack_cmds.command('normalize', short_help='Normalize bacgkround')
@click.option('-i', '--input', type=click.Path(exists=True),  help='Input stack name.')
@click.option('-o', '--output', type=click.Path(exists=False),  help='Output stack name.')
@click.option('--radius', type=int, required=True, help='Radius of particle (in pixels).')
@click.pass_context
def normalize_cmd(ctx, input, output, radius):
    """\b
        Normalize the background of each image to mean 0 and std 1.
        The mean and std of each image in the stack are estimated using pixels
        outside radius r (in pixels). Each image in the stack is corrected
        separately.
    """

    default_logger.debug('Starting normalize')
    default_logger.debug(f'Reading MRC file {input}')
    stack=read_mrc(input)

    # No need to transpose the stack as in the other functions, as the last dimension already
    # corresponds to the slice number.

    default_logger.debug(f'stack of size {stack.shape}')
    default_logger.debug(f'Normalizing background using radius of {radius} pixels')
    stack, _, _ = aspire.preprocessor.normalize_background.normalize_background(stack, radius)

    default_logger.info(f'Saving MRC file {output}')
    write_mrc(output, stack)
    default_logger.debug('Done normalize')

    return


@stack_cmds.command('estimate-snr', short_help='Estimate SNR (signal-to-noise ratio) of a stack.')
@click.option('-i', '--input', type=click.Path(exists=True),  help='Input stack name.')
@click.pass_context
def estimate_snr_cmd(ctx, input):
    """\b
        Estimate SNR (signal-to-noise ratio) of a stack of images. For images of size pxp,
        all pixels outside radius p/2 are considered noise pixels, and all pixels inside
        radius p/2 are considered signal pixels.
    """

    default_logger.debug('Starting estimate SNR')
    default_logger.debug(f'Reading MRC file {input}')
    stack=read_mrc(input)

    default_logger.debug(f'stack of size {stack.shape}')
    snr, sig_power, noise_power = aspire.utils.common.estimate_snr(stack)
    default_logger.debug('Done estimate SNR')
    click.echo(f'SNR={snr}, signal_power={sig_power}, noise_power={noise_power}')

    return snr



@stack_cmds.command('prewhiten', short_help='Prewhiten images')
@click.option('-i', '--input', type=click.Path(exists=True),  help='Input stack name.')
@click.option('-o', '--output', type=click.Path(exists=False),  help='Output stack name.')
@click.pass_context
def prewhiten_cmd(ctx, input, output):
    """\b
        Prewhiten noise in the images. The 2D isotropic power spectrum of the noise is
        estimated from pixels outside the largest circle bounded in each image.
    """
    default_logger.debug('Starting prewhiten')
    default_logger.debug(f'Reading MRC file {input}')
    stack=read_mrc(input)

    default_logger.debug(f'stack of size {stack.shape}')
    stack = aspire.preprocessor.prewhiten.prewhiten(stack, verbose=ctx.obj['VERBOSITY'])

    default_logger.info(f'Saving MRC file {output}')
    write_mrc(output, stack)
    default_logger.debug('Done prewhiten')
    return


@stack_cmds.command('globalflip', short_help='Invert contrast of images.')
@click.option('-i', '--input', type=click.Path(exists=True),  help='Input stack name.')
@click.option('-o', '--output', type=click.Path(exists=False),  help='Output stack name.')
@click.option('--force', is_flag=True, default=False, help='Force contrast flipping.', show_default=True)
@click.pass_context
def globalflip_cmd(ctx, input, output, force):
    """\b
        Check if all images in a stack should be globally phase flipped so that
        the molecule corresponds brighter pixels and the background corresponds
        to darker pixels. This is done by comparing the mean in a small circle
        around the origin (supposed to correspond to the molecule) with the mean
        of the noise, and making sure that the mean of the molecule is larger.
    """

    default_logger.debug('Starting global flipping')
    default_logger.debug(f'Reading MRC file {input}')
    stack=read_mrc(input)

    # No need to transpose the stack as in the other functions, as the last dimension already
    # corresponds to the slice number.

    default_logger.debug(f'stack of size {stack.shape}')
    stack, flipped = aspire.preprocessor.global_phaseflip.global_phaseflip(stack, force)

    default_logger.info(f'stack flipped = {flipped}')
    default_logger.info(f'Saving MRC file {output}')
    write_mrc(output, stack)
    default_logger.debug('Done global flipping')

    return


@stack_cmds.command('preprocess', short_help='Standard preprocessing of a stack of images')
@click.option('--star-file',type=click.Path(exists=True), required=True, help='STAR file with CTF data.')
@click.option('-o', '--output', type=click.Path(exists=False),  help='Output stack name.')
@click.option('--pixA', 'pixA', type=float, default=None, help=',Pixel size in Angstroms (if missing from STAR file).',
              show_default=True)
@click.option('--crop', type=int, default=-1, help='Size of cropped images (in pixels, negative to skip).',
              show_default=True)
@click.option('--downsample', type=int, default=89, help='Size of downsampled images (in pixels, negative to skip).',
              show_default=True)

@click.pass_context
def preprocess_cmd(ctx, star_file, output, pixA, crop, downsample):
    """ \b
         Preprocess a stack by applying phaseflipping, cropping, downsampling, prewhitening,
         and global phase adjustment (in that order).
    """

    stack = preprocess(star_file, pixA, crop, downsample, verbose=ctx.obj['VERBOSITY'])
    default_logger.info(f'Saving MRC file {output}')
    write_mrc(output, stack)

    return


@stack_cmds.command('info', short_help='Information on MRC stacl.')
@click.option('-i', '--input', type=click.Path(exists=True),  help='Input stack name.')
def info_cmd(input):
    """\b
        Print information on MRC stack.
    """

    print_mrc_info(input)
    return

@stack_cmds.command('sort', short_help='Sort a stack of projections.')
@click.option('-i', '--input', type=click.Path(exists=True),  help='Input stack name.')
@click.option('-o', '--output', type=click.Path(exists=False),  help='Sorted index list (text file).')
@click.option('--method', type=click.Choice(['contrast', 'bandpass']))
def sort_stack_cmd(input, output, method):
    """\b
            Sort a stack of projections.
            \b
            Sort a stack of projections (by either contrast of energy in a bandpass) and
        """

    stack_data = read_mrc(input) # XXX Change all "stack" above to "projections". Change function names to _cmd
    stack_data=stack_data.T # XXX Remove once we change all stack to python convention
    if method == "contrast":
        c = stack.contrast(stack_data)
    else:
        raise NotImplementedError("bandpass sorting not implemented yet")
    idx = c[:,1].argsort()
    idx = idx[::-1]
    c = c[idx,:]
    np.savetxt(output, c, fmt='%07d    %e')

@stack_cmds.command('select', short_help='Select a subset of the projections.')
@click.option('-i', '--input', type=click.Path(exists=True),  help='Input stack name.')
@click.option('-o', '--output', type=click.Path(exists=False),  help='output stack name.')
@click.option('--index-file', type=click.Path(exists=True), help='file with indices of images to select (zero-based)')
@click.option('--max-images', type=int, help='Maximal number of images to select')
def select_cmd(input, output, index_file, max_images):
    """\b
            Select a subset of projections
            \b
            Select a subset of projections from the input stack and write them to the output stack.
            The indices of the images to select should be provided as the first column of index-file.
            Other columns in the file are ignored. The indexing of the images is zero based, and are
            written to the output stack in the order in which they are given in the index-file.
            Use # to insert comments to the index file.

        """

    # Read indices of images to select
    default_logger.debug(f"Read index file {index_file}")
    index_info=np.loadtxt(index_file)
    if index_info.ndim > 1:
        default_logger.debug(f"Index file has {index_info.shape[0]} rows {index_info.shape[1]} columns")
        index_info = index_info[:, 0]
    else:
        default_logger.debug(f"Index file has {index_info.shape[0]} rows")
        index_info = index_info.reshape(index_info.shape[0],1)

    if max_images is None:
        max_images = index_info.shape[0]+1

    n=np.minimum(index_info.shape[0], max_images)
    idx=index_info[:n]
    idx=idx.astype(np.int64)

    # Read input stack
    default_logger.info(f"Reading MRC file {input}")
    stack_data = read_mrc(input)  # XXX Change all "stack" above to "projections". Change function names to _cmd
    stack_data = stack_data.T
    default_logger.info(f"Selecting images {n} from MRC file")
    stack_data = stack_data[idx, :, :]  # XXX Remove once we change all stack to python convention

    # Write output stack
    default_logger.info(f"Writing MRC file {output}")
    write_mrc(output,stack_data.T) # XXX Remove T once we switch to python indexing



@cli.command('denoise', short_help='Denoise a stack of images')
@click.option('-i', '--input', type=click.Path(exists=True),  help='Input stack name.')
@click.option('-o', '--output', default=None,
              type=click.Path(exists=False), help='output stack name.')
@click.option("--num_nbor", default=100,
              help=("Number of nearest neighbors to find for each "
                    "image during initial classification. (default=100)"))
@click.option("--nn_avg", default=50, help="Number of images to average into each class. (default=50)")
@click.option("--max_shift", default=15, help="Max shift of projections from the center. (default=15)")
@click.option("--subset-select", default=5000, help="Number of images to pick for abinitio. (default=5000)")
@click.option("--subset-name", default=None, help="Name of file for top images. (default averages_subset.mrcs)")
@click.pass_context
def classify_cmd(ctx, input, output, num_nbor, nn_avg, max_shift, subset_select, subset_name):
    """ \b
        Denoise a stack of images.
        \b
        This command accepts a stack file and denoises each image in the stack with its nn_avg most similar images
        (after rotational and translational alignment).
        classification averaging algorithm.
        \b
        The function returns two stacks:
            1) A stack of the same dimensions as the input stack where each image is averaged with its nn_avg
               most similar images
            2) A subset of the of full stack XXX. Remove this variable.
    """

    ordered_output = 'ordered_averages.mrcs' if subset_name is None else subset_name

    default_logger.info(f'Reading {input}')
    stack = read_mrc(input)
    averages, ordered_averages = class_averaging(stack, num_nbor, nn_avg, max_shift, subset_select,
                                                 verbose=ctx.obj['VERBOSITY'])
    write_mrc(output, averages)
    # write_mrc(ordered_output, ordered_averages)


@cli.command('abinitio', short_help='Common lines-based abinitio reconstruction')
@click.option('-i', '--input', type=click.Path(exists=True),  help='Input stack name.')
@click.option('-o', '--output', default=None,
              type=click.Path(exists=False), help='Filename for reconstructed map.')
@click.option('--symmetry', type=str,  default='C1', help='Symmetry group.')
@click.pass_context
def abinitio_cmd(ctx,input,output,symmetry):
    """ \b
    Ab-initio reconstruction from stack of images.
    \b
    
    Estimate an ab-initio map from the given stack of images. 
    Currently supported symmetry groups are Cn, n>=1.
    """

if __name__ == "__main__":
    cli()
