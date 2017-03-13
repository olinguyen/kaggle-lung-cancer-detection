from __future__ import division

import numpy as np
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
from skimage import morphology


def segment_lungs(img, nodule_mask=None):
    mean = np.mean(img)
    std = np.std(img)
    img = (img - mean) / std
    img = img.astype(np.float64)
    middle = img[100:400, 100:400]
    mean = np.mean(middle)
    img_max = np.max(img)
    img_min = np.min(img)

    img[img == img_max] = mean
    img[img == img_min] = mean

    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img < threshold, 1.0, 0.0)

    eroded = morphology.erosion(thresh_img, np.ones([4, 4]))
    dilation = morphology.dilation(eroded, np.ones([10, 10]))

    labels = measure.label(dilation)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2] - B[0] < 475 and B[3] - B[1] < 475 and B[0] > 40 and B[2] < 472:
            good_labels.append(prop.label)
    mask = np.ndarray([512, 512], dtype=np.int8)
    mask[:] = 0

    for N in good_labels:
        mask += np.where(labels == N, 1, 0)

    mask = morphology.dilation(mask, np.ones([10, 10]))

    img = mask * img

    new_mean = np.mean(img[mask > 0])
    new_std = np.std(img[mask > 0])

    old_min = np.min(img)
    img[img == old_min] = new_mean - 1.2 * new_std
    img = (img - new_mean) / new_std

    labels = measure.label(mask)
    regions = measure.regionprops(labels)

    min_row = 512
    max_row = 0
    min_col = 512
    max_col = 0
    for prop in regions:
        B = prop.bbox
        if min_row > B[0]:
            min_row = B[0]
        if min_col > B[1]:
            min_col = B[1]
        if max_row < B[2]:
            max_row = B[2]
        if max_col < B[3]:
            max_col = B[3]
    width = max_col - min_col
    height = max_row - min_row
    if width > height:
        max_row = min_row + width
    else:
        max_col = min_col + height

    img = img[min_row:max_row, min_col:max_col]
    if max_row - min_row < 5 or max_col - min_col < 5:
        return ()
    else:
        mean = np.mean(img)
        img -= mean
        img_min = np.min(img)
        img_max = np.max(img)
        img /= (img_max - img_min)
        new_img = resize(img, [512, 512])
        if isinstance(nodule_mask, np.ndarray):
            nodule_mask = nodule_mask.astype(np.float64)
            new_nodule_mask = resize(nodule_mask[min_row:max_row, min_col:max_col], [512, 512])
        else:
            new_nodule_mask = 0

        return new_img, new_nodule_mask
