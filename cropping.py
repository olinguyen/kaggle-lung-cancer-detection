from __future__ import division
from __future__ import print_function
import pickle
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dicom
from keras.models import load_model, Model
from keras import backend as K
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from segmentation import segment_lungs
from skimage import measure
from score import false_positive
from keras.layers import Dense, Convolution2D, MaxPooling2D, UpSampling2D, Flatten, Dropout, Reshape
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.cluster import k_means
from keras.regularizers import l1
from sklearn.svm import SVC

databowl = 'databowl/'


def get_patch_coord(centroid, patch_size):
    if centroid[0] < patch_size / 2:
        r_min = 0
        r_max = patch_size
    elif centroid[0] > 512 - patch_size / 2:
        r_min = 512 - patch_size
        r_max = 512
    else:
        r_min = centroid[0] - patch_size / 2
        r_max = centroid[0] + patch_size / 2

    if centroid[1] < patch_size / 2:
        c_min = 0
        c_max = patch_size
    elif centroid[1] > 512 - patch_size / 2:
        c_min = 512 - patch_size
        c_max = 512
    else:
        c_min = centroid[1] - patch_size / 2
        c_max = centroid[1] + patch_size / 2

    return int(r_min), int(r_max), int(c_min), int(c_max)


def read_imgs(patient):
    img_files = glob(patient + '/*')
    slices = [dicom.read_file(img_file) for img_file in img_files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    imgs = np.stack([s.pixel_array for s in slices]).astype(np.float64)
    new_imgs = np.zeros(imgs.shape, dtype=np.float32)
    count = 0
    for img in imgs[int(0.12*imgs.shape[0]):int(0.97*imgs.shape[0])]:
        segmented = segment_lungs(img)
        if len(segmented) == 0:
            continue
        new_imgs[count] = (segmented[0] - np.mean(segmented[0])) / np.std(segmented[0])
        count += 1

    return new_imgs[:count]


def get_filtered_nodules(imgs, unet):
    nodules = []
    masks = unet.predict(imgs[:, np.newaxis, :, :], batch_size=4).astype(int)
    for i in range(masks.shape[0] - 1):
        mask = masks[i, 0]
        next_mask = masks[i + 1, 0]
        blobs = map(lambda x: np.array(x.centroid), measure.regionprops(measure.label(mask)))
        next_blobs = map(lambda x: np.array(x.centroid), measure.regionprops(measure.label(next_mask)))
        for blob in blobs:
            if not false_positive(next_blobs, blob, 15):
                r_min, r_max, c_min, c_max = get_patch_coord(blob, 50)
                nodules.append(imgs[i, r_min:r_max, c_min:c_max])

    return np.array(nodules, dtype=np.float32)


def get_all_nodules(imgs, unet):
    nodules = []
    masks = unet.predict(imgs[:, np.newaxis, :, :], batch_size=4).astype(int)
    for idx, mask in enumerate(masks):
        blobs = map(lambda x: np.array(x.centroid), measure.regionprops(measure.label(mask[0])))
        for blob in blobs:
            r_min, r_max, c_min, c_max = get_patch_coord(blob, 50)
            nodules.append(imgs[idx, r_min:r_max, c_min:c_max])

    return np.array(nodules, dtype=np.float32)


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)


def dice_coef_np(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)

    return (2. * intersection + 1) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1)


def dice_coef_loss(y_true, y_pred):

    return -1*dice_coef(y_true, y_pred)


K.set_image_dim_ordering('th')


unet = load_model(databowl + 'luna/segmented_lungs_unet1.h5', custom_objects={'dice_coef_loss': dice_coef_loss})

df = pd.read_csv(databowl + 'stage1_labels.csv')
ids = df['id'].tolist()
patients = glob(databowl + 'stage1/*')
tr_patients = [patient for patient in patients if patient.split('/')[-1] in ids]
np.random.shuffle(tr_patients)
tr_labels = [df.cancer[df.id == patient.split('/')[-1]].values[0] for patient in tr_patients]
ts_patients = [patient for patient in patients if patient.split('/')[-1] not in ids]

tr_nodules = []
for idx, patient in enumerate(tr_patients):
    imgs = read_imgs(patient)
    tr_nodules.append(get_filtered_nodules(imgs, unet))



