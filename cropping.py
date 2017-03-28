from __future__ import division
from __future__ import print_function
import pickle
from glob import glob
import numpy as np
import pandas as pd
import dicom
from keras import backend as K
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from segmentation import segment_lungs
from skimage import measure
from score import false_positive
from sklearn.cluster import k_means
from sklearn.svm import SVC
from keras.models import load_model
from scipy.ndimage.measurements import label
import cv2

import time

databowl = '/media/data/kaggle/'
kaggle_datafolder = '/media/data/kaggle/'
kaggle_metadata = './data/kaggle/'
K.set_image_dim_ordering('th')


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


def get_masks(imgs, unet):
    masks = unet.predict(imgs[:, np.newaxis, :, :], batch_size=4).astype(int)
    return masks


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


def crop_nodule(img, bbox):
    padding = 5
    y_start = np.clip(bbox[0][1] - padding, 0, 512)
    y_end = np.clip(bbox[1][1] + padding, 0, 512)
    x_start = np.clip(bbox[0][0] - padding, 0, 512)
    x_end = np.clip(bbox[1][0] + padding, 0, 512)
    cropped = img[y_start:y_end, x_start:x_end]
    cropped = cv2.resize(cropped, (50, 50))
    return cropped


def draw_labeled_bboxes(img, labels):
    copied = np.copy(img)
    bboxes = []
    # Iterate through all detected nodules
    for nodule_number in range(1, labels[1]+1):
        # Find pixels with each nodule_number label value
        nonzero = (labels[0] == nodule_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        width = np.max(nonzerox) - np.min(nonzerox)
        height = np.max(nonzerox) - np.max(nonzeroy)

        if width > 5 and height > 5:
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            bboxes.append(bbox)
            # Draw the box on the image
            #cv2.rectangle(img, bbox[0], bbox[1], (10, 10, 10), 2)
            #copied = cv2.addWeighted(copied, 1.0, img, 1.0, 0.)

    # Return the image
    return copied, bboxes


def test_nodules():
    unet = load_model(databowl + 'segmented_lungs_unet1.h5', custom_objects={'dice_coef_loss': dice_coef_loss})

    df = pd.read_csv('./data/kaggle/stage1_labels.csv')
    test_df = pd.read_csv('./data/kaggle/stage1_sample_submission.csv')

    tr_nodules = []

    for idx, patient in enumerate(test_df['id']):
        print(idx, patient)
        imgs = read_imgs('/media/data/kaggle/stage1/' + patient)
        tr_nodules.append((get_filtered_nodules(imgs, unet)), patient)

    np.save('./test_nodules.npy', np.array(tr_nodules))


def train_masks():
    unet = load_model(databowl + 'segmented_lungs_unet1.h5', custom_objects={'dice_coef_loss': dice_coef_loss})
    train_df = pd.read_csv('./data/kaggle/stage1_labels.csv')
    num_samples = len(train_df)
    batch_size = 50
    batch = []
    print("Number of training samples:", num_samples)
    train_df.head()
    for i in range(28):
        nodules = []
        batch_start = i * batch_size
        batch_end = batch_start + batch_size
        ts = time.time()
        for idx, patient in train_df[batch_start:batch_end].iterrows():
            if idx % 5 == 0:
                print(i, idx, patient)
            slices = read_imgs('/media/data/kaggle/stage1/' + patient['id'])
            masks = get_masks(slices, unet)
            nodules.append((masks, patient['id'], patient['cancer']))
            del masks
            del slices
        np.save('/media/data/kaggle/masks/train_masks%d.npy' % i, np.array(nodules))
        te = time.time()
        print("Batch runtime:", te - ts)


def crop_nodules_heatmap():
    unet = load_model(databowl + 'segmented_lungs_unet1.h5', custom_objects={'dice_coef_loss': dice_coef_loss})
    train_df = pd.read_csv('./data/kaggle/stage1_labels.csv')
    num_samples = len(train_df)
    batch_size = 200
    batch = []
    print("Number of training samples:", num_samples)
    for i in range(7):
        nodules = []
        batch_start = i * batch_size
        batch_end = batch_start + batch_size
        ts = time.time()
        for idx, patient in train_df[batch_start:batch_end].iterrows():
            if idx % 50 == 0:
                print(i, idx, patient)
            patient_nodules = []
            # Get all slice masks from patient
            slices = read_imgs('/media/data/kaggle/stage1/' + patient['id'])

            # get predicted masks
            predicted = get_masks(slices, unet)

            # Create heatmap from all slices
            threshold = 2.0
            heatmap = np.sum(predicted, axis=0)[0]

            # threshold to keep hottest regions
            thresh_heatmap = np.copy(heatmap)
            thresh_heatmap[thresh_heatmap < threshold] = 0
            xy = thresh_heatmap.nonzero()
            thresh_heatmap[xy[0], xy[1]] = 1.

            # get bounding boxes on hottest nodule regions
            labels = label(thresh_heatmap)
            img_bbox, bboxes = draw_labeled_bboxes(np.copy(thresh_heatmap), labels)

            padding = 5
            # for each slice, keep only if dice coefficient > threshold
            for idx, predicted_slice in enumerate(predicted):
                for bbox in bboxes:
                    # isolate nodules
                    tmp = np.zeros((512, 512))
                    y_start = np.clip(bbox[0][1] - padding, 0, 512)
                    y_end = np.clip(bbox[1][1] + padding, 0, 512)
                    x_start = np.clip(bbox[0][0] - padding, 0, 512)
                    x_end = np.clip(bbox[1][0] + padding, 0, 512)
                    tmp[y_start:y_end, x_start:x_end] = 1

                    single_nodule_mask = np.logical_and(thresh_heatmap, tmp)

                    # Check if nodule covers area
                    dice_coefficient = dice_coef_np(single_nodule_mask, predicted_slice[0])
                    if dice_coefficient >= 0.40:
                        cropped_nodule = crop_nodule(slices[idx], bbox)
                        patient_nodules.append(cropped_nodule)

            nodules.append((np.array(patient_nodules), patient['id'], patient['cancer']))
            #print("Number of nodules detected for this patient",len(patient_nodules))

        np.save('/media/data/kaggle/masks/cropped_heatmap_nodules_heat2_dice40%d.npy' % i, np.array(nodules))
        te = time.time()
        print("Batch runtime:", te - ts)


def crop_test_nodules():
    unet = load_model(databowl + 'segmented_lungs_unet1.h5', custom_objects={'dice_coef_loss': dice_coef_loss})
    test_df = pd.read_csv('./data/kaggle/stage1_sample_submission.csv')
    num_samples = len(test_df)
    print("Number of testing samples:", num_samples)
    ts = time.time()
    nodules = []
    for idx, patient in test_df.iterrows():
        if idx % 25 == 0:
            print(idx, patient, len(nodules))
        patient_nodules = []
        # Get all slice masks from patient
        slices = read_imgs('/media/data/kaggle/stage1/' + patient['id'])

        # get predicted masks
        predicted = get_masks(slices, unet)

        # Create heatmap from all slices
        threshold = 2.0
        heatmap = np.sum(predicted, axis=0)[0]

        # threshold to keep hottest regions
        thresh_heatmap = np.copy(heatmap)
        thresh_heatmap[thresh_heatmap < threshold] = 0
        xy = thresh_heatmap.nonzero()
        thresh_heatmap[xy[0], xy[1]] = 1.

        # get bounding boxes on hottest nodule regions
        labels = label(thresh_heatmap)
        img_bbox, bboxes = draw_labeled_bboxes(np.copy(thresh_heatmap), labels)

        padding = 5
        # for each slice, keep only if dice coefficient > threshold
        for idx, predicted_slice in enumerate(predicted):
            for bbox in bboxes:
                # isolate nodules
                tmp = np.zeros((512, 512))
                y_start = np.clip(bbox[0][1] - padding, 0, 512)
                y_end = np.clip(bbox[1][1] + padding, 0, 512)
                x_start = np.clip(bbox[0][0] - padding, 0, 512)
                x_end = np.clip(bbox[1][0] + padding, 0, 512)
                tmp[y_start:y_end, x_start:x_end] = 1

                single_nodule_mask = np.logical_and(thresh_heatmap, tmp)

                # Check if nodule covers area
                dice_coefficient = dice_coef_np(single_nodule_mask, predicted_slice[0])
                if dice_coefficient >= 0.40:
                    cropped_nodule = crop_nodule(slices[idx], bbox)
                    patient_nodules.append(cropped_nodule)

        nodules.append((np.array(patient_nodules), patient['id']))
        #print("Number of nodules detected for this patient",len(patient_nodules))

    np.save(kaggle_datafolder + 'masks/test_cropped_heatmap_nodules_heat2_dice40.npy', np.array(nodules))
    te = time.time()
    print("Batch runtime:", te - ts)


if __name__ == "__main__":
    crop_test_nodules()
