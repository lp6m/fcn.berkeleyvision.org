import glob
import os
import cv2

paths = glob.glob("./seg_train_images/*.jpg")
print(len(list(paths)))
for path in paths:
    img = cv2.imread(path)
    img = cv2.flip(img, 1)
    dstpath = "./seg_train_images/" + os.path.basename(path).replace('.jpg', '') + "_flip.jpg"
    print(dstpath)
    cv2.imwrite(dstpath, img)
    
paths = glob.glob("./seg_train_images/*.jpg")
print(len(list(paths)))