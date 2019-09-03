import numpy as np
import aspire.aspire.utils.common as common
from aspire.aspire.class_averaging.compute_spca import compute_spca
from aspire.aspire.class_averaging.initial_classification import initial_classification_fd_update
from aspire.aspire.class_averaging.align_main import align_main
from aspire.aspire.class_averaging.cryo_select_subset import cryo_select_subset
import time


def class_averaging(stack, num_nbor=100, nn_avg=50, max_shift=15, n_images_to_pick=5000):
    print('Computing the signal and noise of the images')
    snr, signal, noise = common.estimate_snr(stack)
    print('Signal is {}, noise is {}, snr is {}'.format(signal, noise, snr))

    # spca data
    print('Computing steerable PCA')
    tic = time.time()
    spca_data = compute_spca(stack, noise)
    toc = time.time()
    print('Finished computing steerable PCA in {} seconds'.format(toc - tic))

    # initial classification fd update
    print('Finding {} nearest neighbors')
    tic = time.time()
    classes, class_refl, rot, corr, _ = initial_classification_fd_update(spca_data, num_nbor)
    toc = time.time()
    print('Finished classification in {} seconds'.format(toc - tic))

    # VDM
    # class_vdm, class_vdm_refl, angle = cls.vdm(classes, np.ones(classes.shape), rot,
    #                                            class_refl, 50, False, 50)

    # align main
    list_recon = np.arange(classes.shape[0])
    use_em = True
    print('Averaging images with their {} nearest neighbors with maximum shift of {}'.format(nn_avg, max_shift))
    tic = time.time()
    shifts, corr, averages, norm_variance = align_main(stack, rot, classes, class_refl, spca_data, nn_avg, max_shift,
                                                       list_recon, 'my_tmpdir', use_em)
    toc = time.time()
    print('Finished averaging in {} seconds'.format(toc - tic))

    # Picking images for abinitio. I think it should be in abinitio or in a completely separate function
    # indices = cryo_smart_select_subset(classes, size_output, contrast_priority, to_image)
    print('Finding top {} images by contrast'.format(n_images_to_pick))
    ordered_averages = cryo_select_subset(averages, classes, n_images_to_pick)
    return averages, ordered_averages
