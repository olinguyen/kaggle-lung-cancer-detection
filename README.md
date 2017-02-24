# Data Science Bowl 2017: Lung Cancer Detection

## Overview

This is our submission to Kaggle's Data Science Bowl 2017 on lung cancer detection.

## Objective

Using the data set of high-resolution CT lung scans, develop an algorithm that will classify if lesions in the lungs are cancerous or not. More specifically, the Kaggle competition task is to create an automated method capable of determining whether or not a patient will be diagnosed with lung cancer within one year of the date the CT scan was taken.

## File Descriptions

Each patient id has an associated directory of DICOM files. The patient id is found in the DICOM header and is identical to the patient name. The exact number of images will differ from case to case, varying according in the number of slices. Images were compressed as .7z files due to the large size of the dataset.

* **stage1.7z** - contains all images for the first stage of the competition, including both the training and test set. This is file is also hosted on BitTorrent.

* **stage1_labels.csv** - contains the cancer ground truth for the stage 1 training set images

* **stage1_sample_submission.csv** - shows the submission format for stage 1. You should also use this file to determine which patients belong to the leaderboard set of stage 1.

* **sample_images.7z** - a smaller subset set of the full dataset, provided for people who wish to preview the images before downloading the large file.

* **data_password.txt** - contains the decryption key for the image files
