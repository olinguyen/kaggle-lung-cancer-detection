from glob import glob

from tqdm import tqdm
import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage.draw import circle


luna = 'data/luna2016/'
file_list = glob(luna + 'subset*/*.mhd')


def get_filename(case):
    for f in file_list:
        if case in f:
            return f


def get_nodules(img_file, biggest=False):
    mini_df = df_node[df_node['file'] == img_file]
    if len(mini_df) == 0:
        return []

    if biggest:
        idx = np.argsort(mini_df['diameter_mm'].values)[-1:]
    else:
        idx = range(len(mini_df))

    x = mini_df['coordX'].values[idx]
    y = mini_df['coordY'].values[idx]
    z = mini_df['coordZ'].values[idx]
    diam = mini_df['diameter_mm'].values[idx]
    return list(zip(x, y, z, diam))


def get_arrays(img_file, biggest=False):
    nodules = get_nodules(img_file, biggest)
    itk_img = sitk.ReadImage(img_file)
    img = sitk.GetArrayFromImage(itk_img)  # zyx
    origin = np.array(itk_img.GetOrigin())  # xyz
    spacing = np.array(itk_img.GetSpacing())  # xyz
    masks = []
    large_masks = []
    imgs = []
    for nodule in nodules:
        center = np.array(nodule[:3])  # xyz
        v_center = np.rint((center-origin)/spacing).astype(int)  # xyz
        v_diam = int(np.round(nodule[3]/spacing[0]))
        mask = np.zeros((img.shape[1], img.shape[2]), dtype=np.int8)  # yx
        rr, cc = circle(v_center[1], v_center[0], 2*(v_diam/2), shape=mask.shape)
        mask[rr, cc] = 1
        masks.append(mask)
        mask = np.zeros((img.shape[1], img.shape[2]), dtype=np.int8)  # yx
        rr, cc = circle(v_center[1], v_center[0], 4*(v_diam/2), shape=mask.shape)
        mask[rr, cc] = 1
        large_masks.append(mask)
        imgs.append(img[v_center[2], :, :])

    return masks, large_masks, imgs


df_node = pd.read_csv(luna + 'annotations.csv')
df_node['file'] = df_node['seriesuid'].apply(get_filename)
df_node = df_node.dropna()

imgs = np.zeros((2000, 512, 512), dtype=np.int16)
masks = np.zeros((2000, 512, 512), dtype=np.int8)
large_masks = np.zeros((2000, 512, 512), dtype=np.int8)
i = 0
for img_file in tqdm(file_list):
    arrays = get_arrays(img_file)
    if len(arrays[0]) == 0:
        continue
    nb = len(arrays[0])
    masks[i:i+nb, :, :] = arrays[0]
    large_masks[i:i+nb, :, :] = arrays[1]
    imgs[i:i+nb, :, :] = arrays[2]
    i += nb

imgs = imgs[:i, :, :]
masks = masks[:i, :, :]
large_masks = large_masks[:i, :, :]

np.random.seed(23)
idx = np.random.permutation(imgs.shape[0])
imgs = imgs[idx, :, :]
masks = masks[idx, :, :]
large_masks = large_masks[idx, :, :]

np.save(luna + 'imgs.npy', imgs)
np.save(luna + 'masks.npy', masks)
np.save(luna + 'large_masks.npy', large_masks)
