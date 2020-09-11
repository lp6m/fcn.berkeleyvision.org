
import numpy as np
from PIL import Image
import cv2
# pil_img = Image.open('./data/bdd100k/seg/labels/train/b2e73518-547798ef_train_id.png')
# img = np.asarray(pil_img)
# y, x = img.shape
# print(y, x)
# #car 13 road 0 person 11 signal 6
# OTHER = 4
# img = np.where((img != 0) & (img != 6) & (img != 11) & (img != 13), OTHER, img)
# #road 0 person 1 signal 2 car 3 other 4 
# img = np.where(img == 11, 1, img)
# img = np.where(img == 6, 2, img)
# img = np.where(img == 13, 3, img)
# # import scipy.io
# # mat = scipy.io.loadmat('./data/sbdd/dataset/cls/2011_003185.mat')
# # img = mat['GTcls'][0]['Segmentation'][0].astype(np.uint8)
# # y, x = img.shape
# col_map = [[0, 0, 0], [224, 224, 192], [192, 128, 128], [128, 192, 0], [64, 0, 128], [128, 128, 0], [0, 0, 128], [192, 0, 0], [192, 128, 0], [64, 128, 128], [64, 0, 0], [128, 0, 128], [128, 128, 128], [192, 0, 128], [0, 128, 0], [0, 192, 0], [128, 0, 0], [64, 128, 0], [0, 64, 128], [0, 128, 128], [0, 64, 0], [128, 64, 0]]
# result = np.zeros((y, x, 3))
# for yy in range(y):
#     for xx in range(x):
#         result[yy][xx] = col_map[img[yy][xx]]
# # result = result.clip(0, 255)
# result = cv2.resize(result, (224, 224), interpolation=cv2.INTER_NEAREST)

# cv2.imwrite("dataset.png", result)
im =  Image.open('./data/bdd100k/seg/images/train/b2e73518-547798ef.jpg')
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]#BGR
in_ = cv2.resize(in_, (224, 224))
cv2.imwrite("dataset.png", in_)