import numpy as np
from skimage import measure


def true_positive(act_blob, pred_blobs, max_dist):
    """checks whether 1 ground truth nodule was predicted correctly"""

    for pred_blob in pred_blobs:
        if np.sqrt(np.abs(np.dot(act_blob - pred_blob, act_blob - pred_blob))) < max_dist:
            return 1
    return 0


def false_positive(act_blobs, pred_blob, max_dist):
    """checks whether 1 predicted nodule is a false positive"""

    for act_blob in act_blobs:
        if np.sqrt(np.abs(np.dot(act_blob - pred_blob, act_blob - pred_blob))) < max_dist:
            return 0
    return 1


def score_slice(act, pred, max_dist):
    """function that takes ground truth mask and predicted mask for one slice and
    outputs the numbers of true positives, false positives, and false negatives.
    max_dist is the maximum distance in pixels allowed between the centroid of ground
    truth nodule and that of predicted nodule in order to consider the predicted
    nodule a true positive.
    """

    act_blobs = map(lambda x: np.array(x.centroid), measure.regionprops(measure.label(act)))
    pred_blobs = map(lambda x: np.array(x.centroid), measure.regionprops(measure.label(pred)))
    true_positives = 0
    false_positives = 0
    for act_blob in act_blobs:
        true_positives += true_positive(act_blob, pred_blobs, max_dist)
    for pred_blob in pred_blobs:
        false_positives += false_positive(act_blobs, pred_blob, max_dist)
    false_negatives = len(act_blobs) - true_positives

    return np.array([true_positives, false_positives, false_negatives])
