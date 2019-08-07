import numpy as np
import aspire.utils.common as common
from aspire.class_averaging.compute_spca import compute_spca
from aspire.class_averaging.initial_classification import initial_classification_fd_update
from aspire.class_averaging.align_main import align_main
from aspire.class_averaging.cryo_select_subset import cryo_select_subset
from aspire.common import *


def class_averaging(stack, num_nbor=100, nn_avg=50, max_shift=15, n_images_to_pick=5000):
    default_logger.info('Step 1/4: Estimating SNR of images images')
    snr, signal, noise = common.estimate_snr(stack)
    default_logger.info(f'Signal power =  {signal:.4e}')
    default_logger.info(f'Noise power = {noise:.4e}')
    default_logger.info(f'Estimated SNR = 1/{int(round(1.0/snr)):d} (more precisely {snr:.4e})')

    # spca data
    default_logger.info('Step 2/4: Computing steerable PCA')
    spca_data = compute_spca(stack, noise)
    default_logger.info('Step 2/4: Finished computing steerable PCA')

    # initial classification fd update
    default_logger.info('Step 3/4: Initial classification (Finding nearest neighbors for each projection)')
    classes, class_refl, rot, corr, _ = initial_classification_fd_update(spca_data, num_nbor)
    default_logger.info('Step 3/4: Finished initial classification')

    # VDM
    # class_vdm, class_vdm_refl, angle = cls.vdm(classes, np.ones(classes.shape), rot,
    #                                            class_refl, 50, False, 50)

    # align main
    list_recon = np.arange(classes.shape[0])
    use_em = True
    default_logger.info('Step 4/4: Averaging images with their {} nearest neighbors with maximum shift of {} pixels'.format(nn_avg, max_shift))
    shifts, corr, averages, norm_variance = align_main(stack, rot, classes, class_refl, spca_data, nn_avg, max_shift,
                                                       list_recon, 'my_tmpdir', use_em)
    default_logger.info('Step 4/4: Finished averaging')

    # Picking images for abinitio. I think it should be in abinitio or in a completely separate function
    # indices = cryo_smart_select_subset(classes, size_output, contrast_priority, to_image)
    print('Finding {} images with highest contrast'.format(n_images_to_pick))
    indices = cryo_select_subset(averages, classes, n_images_to_pick)
    return averages, indices
