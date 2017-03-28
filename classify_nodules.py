import os
import pandas as pd
import numpy as np
import os.path as path
import glob
import pickle

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K


kaggle_datafolder = '/media/data/kaggle/'
kaggle_metadata = './data/kaggle/'
K.set_image_dim_ordering('tf')


def load_pickles():
    nodule_files = glob.glob(kaggle_datafolder + '*.pickle')
    with open(nodule_files[0], "rb") as f:
        data = pickle.load(f, encoding='latin1')

    for nodule_file in nodule_files[1:]:
        with open(nodule_file, "rb") as f:
            tmp = pickle.load(f, encoding='latin1')
            data = np.hstack((data, tmp))
            
    data = np.swapaxes(data, 0, 1)    
    print("Number of patients:", data.shape)
    print("Patient shape:", data[0][0].shape)
    return data


def load_cropped_nodules():
    nodule_files = glob.glob(kaggle_datafolder + 'masks/*cropped_heatmap_nodules*')
    nodule_files.sort()
    data = np.load(nodule_files[0])
    for nodule_file in nodule_files[1:]:
        tmp = np.load(nodule_file)
        data = np.vstack((data, tmp))
	    
    print("Number of patients:", data.shape)
    print("Patient shape:", data[0][0].shape)
    return data


'''
Trains a classifier to classify nodes as cancerous/non-cancerous
'''
def train_cancer_classifier():
    data = load_cropped_nodules()

    # Prepare data for training
    nodules = data[0][0]
    labels = np.ones(data[0][0].shape[0]) * data[0][2]
    for idx, patient in enumerate(data[1:]):
        if patient[0].any():
            labels = np.concatenate((labels, np.ones(patient[0].shape[0]) * patient[2]))
            nodules = np.vstack((nodules, patient[0]))
	    
    print('Nodules shape:', nodules.shape)
    print('Labels shape:', labels.shape)

    num_classes = 2
    num_samples = nodules.shape[0]
    img_rows = nodules.shape[1]
    img_cols = nodules.shape[2]

    X_train, X_test, y_train, y_test = train_test_split(nodules, labels, test_size=0.33, random_state=42)

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)        
    print('Num train samples:', len(y_train))
    print('Num test samples:', len(y_test))
    
    epochs = 50
    batch_size = 128
    model = get_conv2d_model()
    print(X_train.shape)
    print(y_train.shape)
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs,
                      verbose=2, validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # save model
    model.save_weights('./data/kaggle/convnet_nodules.h5')    

def get_conv2d_model():
    model = Sequential()
    num_classes = 2
    input_shape = (50, 50, 1)
    model.add(Conv2D(32, 3, 3,
		     activation='relu',
		     input_shape=input_shape))

    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    return model


def classify_test_nodules():
    img_rows, img_cols = (50, 50)
    test_df = pd.read_csv('./data/kaggle/stage1_sample_submission.csv')
    test_nodules = np.load("/media/data/kaggle/masks/test_cropped_heatmap_nodules_heat2_dice40.npy")
    model = get_conv2d_model()
    model.load_weights('./data/kaggle/convnet_nodules.h5')
        
    for idx, patient in test_df[:1].iterrows():
        test_nodules[idx][0] = test_nodules[idx][0].reshape((-1, img_rows, img_cols, 1))

        if test_nodules[idx][0].shape[0] != 0:
            prob = np.mean(model.predict(test_nodules[idx][0]), axis=0)[1]
            test_df.loc[idx, 'cancer'] = prob
        else:
            test_df.loc[idx, 'cancer'] = 0.5        

    test_df.head()
    test_df.to_csv('./submission.csv')


if __name__ == "__main__":
    #train_cancer_classifier()
    classify_test_nodules()
