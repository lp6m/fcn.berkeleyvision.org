import numpy as np
import cv2
f = open("./seg_train_dat/train_2242_flip.dat",mode='rb')
topo = np.fromfile(f, dtype='uint8',sep='').reshape(227, 227, 1)
res = np.zeros((227, 227, 3), dtype='uint8')
for y in range(227):
    for x in range(227):
        idx = topo[y][x]
        if idx == 3:
            res[y][x] = [255, 0, 0]
        elif idx == 0:
            res[y][x] = [142, 47, 68]
        elif idx == 1:
            res[y][x] = [0, 0, 255]
        elif idx == 2:
            res[y][x] = [0, 255, 255]
        else:
            res[y][x] = [0, 0, 0]
cv2.imwrite("res.png", res)
print(topo.shape)