import os
import glob

#train
paths = sorted(list(glob.glob("./seg_train_dat/*.dat")))
print(len(paths))
basenames = [os.path.basename(path).replace('.dat', '') + "\n" for path in paths]
f = open("./train.txt", "w")
f.writelines(basenames)
f.close()

#val
paths = sorted(list(glob.glob("./seg_test_images/*.jpg")))
print(len(paths))
basenames = [os.path.basename(path).replace('.jpg', '') + "\n" for path in paths]
f = open("./val.txt", "w")
f.writelines(basenames)
f.close()


