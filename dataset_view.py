import scipy.io
import numpy as np
#check SBDD data

mat = scipy.io.loadmat('./data/sbdd/dataset/cls/2011_002357.mat')
label = mat['GTcls'][0]['Segmentation'][0].astype(np.uint8)
# label = label[np.newaxis, ...]
print(label.shape)

from PIL import Image
pil_img = Image.open('./data/sbdd/dataset/cls/2011_002357.png')
img = np.asarray(pil_img)
print(img.shape)
#verify label from .mat and .png is same
print(np.array_equal(label, img, ))

#get bdd100k label
#color_labels: NOT index color!!
#labels
#label information: https://github.com/ucbdrive/bdd100k/blob/master/bdd100k/label.py
# pil_img = Image.open('./data/bdd100k/seg/color_labels/train/a21d5510-651ee905_train_color.png')
pil_img = Image.open('./data/bdd100k/seg/labels/train/a21d5510-651ee905_train_id.png')
img = np.asarray(pil_img)
uniq = np.unique(img)
print(uniq)
#car 13 road 0 person 11 signal 6
OTHER = 4
img = np.where((img != 0) & (img != 6) & (img != 11) & (img != 13), OTHER, img)
#road 0 person 1 signal 2 car 3 other 4 
img = np.where(img == 11, 1, img)
img = np.where(img == 6, 2, img)
img = np.where(img == 13, 3, img)
uniq = np.unique(img)
print(img.dtype)
print(uniq)
print(img.shape)

