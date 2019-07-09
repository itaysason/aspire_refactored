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
from aspire.utils.read_write import write_mrc, read_mrc


np.random.seed(1137)

# CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group()
@click.option('-v', '--verbosity', type=click.IntRange(0, 3), default=0, help='Verbosity level (0: silent, 1: progress, 2: info, 3: debug).',show_default=True)
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
def stack(ctx):
    """\b
        Two-dimensional processing of a stack of images.
    """

    return


@stack.command('phaseflip', short_help='Apply phase-flipping')
@click.option('--star-file',type=click.Path(exists=True), required=True, help='STAR file with CTF data.')
@click.option('-o', '--output', type=click.Path(exists=False),  help='Output stack name.')
@click.option('--pixA', 'pixA', type=float, default=None, help=',Pixel size in Angstroms (if missing from STAR file).')
@click.pass_context
def phaseflip(ctx, star_file, output, pixA):
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


@stack.command('crop', short_help='Crop images')
@click.option('-i', '--input', type=click.Path(exists=True),  help='Input stack name.')
@click.option('-o', '--output', type=click.Path(exists=False),  help='Output stack name.')
@click.option('--new-size', type=int, required=True, default=89, help='Size of cropped images (in pixels).')
def crop(input, output, new_size):
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


@stack.command('downsample', short_help='Downsample images')
@click.option('-i', '--input', type=click.Path(exists=True),  help='Input stack name.')
@click.option('-o', '--output', type=click.Path(exists=False),  help='Output stack name.')
@click.option('--new-size', type=int, required=True, default=89, help='Size of downsampled images (in pixels).')
@click.pass_context
def downsample(ctx, input, output, new_size):
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


@stack.command('normalize', short_help='Normalize bacgkround')
@click.option('-i', '--input', type=click.Path(exists=True),  help='Input stack name.')
@click.option('-o', '--output', type=click.Path(exists=False),  help='Output stack name.')
@click.option('--radius', type=int, required=True, help='Radius of particle (in pixels).')
@click.pass_context
def normalize(ctx, input, output, radius):
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


@stack.command('estimate-snr', short_help='Estimate SNR (signal-to-noise ratio) of a stack.')
@click.option('-i', '--input', type=click.Path(exists=True),  help='Input stack name.')
@click.pass_context
def estimate_snr(ctx, input):
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



@stack.command('prewhiten', short_help='Prewhiten images')
@click.option('-i', '--input', type=click.Path(exists=True),  help='Input stack name.')
@click.option('-o', '--output', type=click.Path(exists=False),  help='Output stack name.')
@click.pass_context
def prewhiten(ctx, input, output):
    """\b
        Prewhiten noise in the images. The 2D isotropic power spectrum of the noise is
        estimated from pixels outside the largest circle bounded in each image.
    """
    default_logger.debug('Starting prewhiten')
    default_logger.debug(f'Reading MRC file {input}')
    stack=read_mrc(input)

    default_logger.debug(f'stack of size {stack.shape}')
    stack = aspire.preprocessor.prewhiten.prewhiten(stack)

    default_logger.info(f'Saving MRC file {output}')
    write_mrc(output, stack)
    default_logger.debug('Done prewhiten')
    return


@stack.command('globalflip', short_help='Invert contrast of images.')
@click.option('-i', '--input', type=click.Path(exists=True),  help='Input stack name.')
@click.option('-o', '--output', type=click.Path(exists=False),  help='Output stack name.')
@click.option('--force', is_flag=True, default=False, help='Force contrast flipping.')
@click.pass_context
def globalflip(ctx, input, output, force):
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


@stack.command('preprocess', short_help='Standard preprocessing of a stack of images')
@click.option('--pixA', 'pixA', type=float, help=',Pixel size in Angstroms (if missing from STAR file).')
@click.option('--crop', type=int, default=-1, help='Size of cropped images (in pixels, negative to skip).')
@click.option('--downsample_', type=int, default=89)
def preprocess_cmd(pixA, crop, downsample):
    """ \b
         Preprocess a stack by applying phaseflipping, cropping, downsampling, prewhitening,
         and global phase adjustment.
    """
    return

if __name__ == "__main__":
    cli()
