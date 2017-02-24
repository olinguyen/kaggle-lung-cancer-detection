#!/bin/bash
TARGET_DIR="./data"
FILE=$TARGET_DIR/$SAMPLE_IMGS

KAGGLE=https://www.kaggle.com/c/data-science-bowl-2017/download
ARCHIVE=stage1_labels.csv.zip
DATA_PWD=data_password.txt.zip
SAMPLE_IMGS=sample_images.7z

if [ -z "$FILE" ] ; then
  echo Unpacking archive...
  tar xf $TARGET_DIR/$SAMPLE_IMGS -C $TARGET_DIR
  unzip $TARGET_DIR/$ARCHIVE -d $TARGET_DIR
  unzip $TARGET_DIR/$DATA_PWD -d $TARGET_DIR
else
  echo Download links
  echo $KAGGLE/$ARCHIVE
  echo $KAGGLE/$DATA_PWD
  echo $KAGGLE/$SAMPLE_IMGS
fi



