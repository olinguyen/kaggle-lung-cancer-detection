# Data Science Bowl 2017: Lung Cancer Detection

## Overview

This is our submission to Kaggle's Data Science Bowl 2017 on lung cancer detection.

## Objective

Using the data set of high-resolution CT lung scans, develop an algorithm that will classify if lesions in the lungs are cancerous or not. More specifically, the Kaggle competition task is to create an automated method capable of determining whether or not a patient will be diagnosed with lung cancer within one year of the date the CT scan was taken.

## File Descriptions

### Kaggle dataset

Each patient id has an associated directory of DICOM files. The patient id is found in the DICOM header and is identical to the patient name. The exact number of images will differ from case to case, varying according in the number of slices. Images were compressed as .7z files due to the large size of the dataset.

* **stage1.7z** - contains all images for the first stage of the competition, including both the training and test set. This is file is also hosted on BitTorrent.

* **stage1_labels.csv** - contains the cancer ground truth for the stage 1 training set images

* **stage1_sample_submission.csv** - shows the submission format for stage 1. You should also use this file to determine which patients belong to the leaderboard set of stage 1.

* **sample_images.7z** - a smaller subset set of the full dataset, provided for people who wish to preview the images before downloading the large file.

* **data_password.txt** - contains the decryption key for the image files

### Luna Dataset

For this challenge, we use the publicly available LIDC/IDRI database. We excluded scans with a slice thickness greater than 2.5 mm. In total, 888 CT scans are included. The LIDC/IDRI database also contains annotations which were collected during a two-phase annotation process using 4 experienced radiologists. Each radiologist marked lesions they identified as non-nodule, nodule < 3 mm, and nodules >= 3 mm.

* **subset0.zip** to **subset9.zip**: 10 zip files which contain all CT images
* **annotations.csv**: csv file that contains the annotations used as reference standard for the 'nodule detection' track
* **candidates.csv**: csv file that contains the candidate locations for the ‘false positive reduction’ track
* **sampleSubmission.csv**: an example of a submission file in the correct format
Additional data includes:

* **evaluation script**: the evaluation script that is used in the LUNA16 framework
* **lung segmentation**: a directory that contains the lung segmentation for CT images computed using automatic algorithms
* **candidates_V2.csv**: csv file that contains the candidate locations for the extended ‘false positive reduction’ track
* **additional_annotations.csv**: csv file that contain additional nodule annotations from our observer study. The file will be available soon
