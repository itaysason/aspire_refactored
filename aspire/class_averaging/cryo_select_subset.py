import numpy as np
import aspire.utils.common as common


def cryo_select_subset(images, classes, size_output, to_image=None, n_skip=None):
    contrast = cryo_image_contrast(images)
    priority = np.argsort(-contrast)
    num_images, num_neighbors = classes.shape

    if to_image is None:
        to_image = num_images

    if n_skip is None:
        n_skip = min(to_image // size_output, num_neighbors)
    # else:
    #     if n_skip > min(to_image // size_output, num_neighbors):
    #         n_skip = min(to_image // size_output, num_neighbors)

    mask = np.zeros(num_images, dtype='int')
    selected = []
    curr_image_idx = 0

    while len(selected) <= size_output and curr_image_idx < to_image:
        while curr_image_idx < to_image and mask[priority[curr_image_idx]] == 1:
            curr_image_idx += 1
        if curr_image_idx < to_image:
            selected.append(priority[curr_image_idx])
            mask[classes[priority[curr_image_idx], :n_skip]] = 1
            curr_image_idx += 1
    indices = np.array(selected, dtype='int')[:min(size_output, len(selected))]
    return images[:, :, indices]


def cryo_smart_select_subset(classes, size_output, priority=None, to_image=None):
    """
    Not tested
    :param classes:
    :param size_output:
    :param priority:
    :param to_image:
    :return:
    """
    num_images = classes.shape[0]
    num_neighbors = classes.shape[1]
    if to_image is None:
        to_image = num_images

    if priority is None:
        priority = np.arange(num_images)

    n_skip = min(to_image // size_output, num_neighbors)
    for i in range(num_neighbors, n_skip - 1, -1):
        selected = cryo_select_subset(classes, size_output, priority, to_image, i)
        if len(selected) == size_output:
            return selected
    return cryo_select_subset(classes, size_output, priority, to_image)


def cryo_image_contrast(projs, r=None):
    # Redundency with stack.contrast (which I verified carefully)
    n = projs.shape[0]
    if r is None:
        r = n // 2

    indices = common.disc(n, r)
    contrast = np.std(projs[indices], axis=0, ddof=1)

    return contrast
