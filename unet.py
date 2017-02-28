from __future__ import division
from __future__ import print_function

import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras import backend as K


luna = '/home/a6elkhat/Documents/databowl/luna/'


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
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((1,512, 512))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss)

    return model


K.set_image_dim_ordering('th')

imgs = np.load(luna + 'imgs.npy').astype(np.float32)
masks = np.load(luna + 'masks.npy').astype(np.int)

masks_new = np.zeros(masks.shape)
imgs_new = np.zeros(imgs.shape)
count = 0
for idx, img in enumerate(imgs):
    if np.sum(masks[idx]) == 0:
        continue
    imgs_new[count] = (img - np.mean(img))/np.std(img)
    masks_new[count] = masks[idx]
    count += 1

imgs = imgs_new[:count]
masks = masks_new[:count]

del imgs_new, masks_new

spl = int(np.round(imgs.shape[0]*0.7))
tr_imgs = imgs[:spl, np.newaxis, :, :]
tr_masks = masks[:spl, np.newaxis, :, :]
ts_imgs = imgs[spl:, np.newaxis, :, :]
ts_masks = masks[spl:, np.newaxis, :, :]
del imgs, masks

# unet = get_unet()
unet = load_model(luna + 'unet9.h5', custom_objects={'dice_coef_loss': dice_coef_loss})
for i in range(10, 20):
    unet.fit(tr_imgs, tr_masks, batch_size=4, nb_epoch=10, verbose=2)
    unet.save(luna + 'unet%d.h5' % i)

    dc = []
    masks_h = unet.predict(tr_imgs, batch_size=4)
    for idx in range(masks_h.shape[0]):
        dc.append(dice_coef_np(tr_masks[idx, 0, :, :], masks_h[idx, 0, :, :]))
    print('train score, epoch:', (i+1)*10, '--', np.mean(dc))

    dc = []
    masks_h = unet.predict(ts_imgs, batch_size=4)
    for idx in range(masks_h.shape[0]):
        dc.append(dice_coef_np(ts_masks[idx, 0, :, :], masks_h[idx, 0, :, :]))
    print('test score, epoch:', (i+1)*10, '--', np.mean(dc))
