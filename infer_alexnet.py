import numpy as np
from PIL import Image

import caffe
import cv2    #这里引入CV2

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
im = Image.open("./data/bdd100k/seg/images/train/b2be7200-b6f7fe0a.jpg")
# im = Image.open('/home/FCN_via_keras/dataset/VOCdevkit/VOC2012/JPEGImages/2012_003382.jpg')#此处填写需要测试的图片的路径
# im = Image.open('/home/seg_train_images/train_2234.jpg')
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
origin = np.copy(in_)
in_ -= np.array((125.92085,125.92085,125.92085))#测试图片的均值BGR传入
in_ = in_.transpose((2,0,1))
# print(in_.shape)
# load net
# net = caffe.Net('/home/fcn.berkeleyvision.org/voc-fcn-alexnet/deploy.prototxt', '/home/fcn.berkeleyvision.org/voc-fcn-alexnet/fcn-alexnet-pascal.caffemodel', caffe.TEST)
net = caffe.Net('/home/fcn.berkeleyvision.org/voc-my-fcn-alexnet/deploy.prototxt', './voc-my-fcn-alexnet/snapshot/train_iter_32000.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data   #这里提示我们NCHW分别是
#N:即batch_size
#C:即通道数，channels
#H:即每一个通道的高，height
#W：即每一个通道的宽，width
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
out = net.blobs['score'].data[0].argmax(axis=0)
categories = set()
y, x = out.shape
for yy in range(y):
    for xx in range(x):
        if out[yy][xx] not in categories:
            categories.add(out[yy][xx])
            print(out[yy][xx])
# print(out.shape)

#添加最后这两行将生成的图片保存下来
col_map = [[0, 0, 0], [224, 224, 192], [192, 128, 128], [128, 192, 0], [64, 0, 128], [128, 128, 0], [0, 0, 128], [192, 0, 0], [192, 128, 0], [64, 128, 128], [64, 0, 0], [128, 0, 128], [128, 128, 128], [192, 0, 128], [0, 128, 0], [0, 192, 0], [128, 0, 0], [64, 128, 0], [0, 64, 128], [0, 128, 128], [0, 64, 0], [128, 64, 0]]
# result = np.expand_dims(np.array((-255.) * (out-1.)).astype(np.float32), axis = 2)   
result = np.zeros((y, x, 3))
# result[:, :] = col_map[out[:, :]]
for yy in range(y):
    for xx in range(x):
        result[yy][xx] = col_map[out[yy][xx]]
print(result.shape)
print(origin.shape)
# result = origin
# result = ((0.6 * origin) + (0.4 * result)).astype("uint8")
# dst = cv2.addWeighted(result, 0.5, origin, 0.5, 120)
result = result.clip(0, 255)

cv2.imwrite("result308.png", result)