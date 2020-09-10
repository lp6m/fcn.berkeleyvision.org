import glob
import os

for mode in ['train', 'val']:
    imagelist = glob.glob('./images/{}/*.jpg'.format(mode))
    labellist = glob.glob('./labels/{}/*.png'.format(mode))
    imagelist = [os.path.basename(i).replace('.jpg', '') for i in imagelist]
    labellist = [os.path.basename(l).replace('_train_id.png', '') for l in labellist]
    print(len(imagelist), len(labellist))
    imagelist = set(imagelist)
    labellist = set(labellist)
    res = set()
    for label in labellist:
        if label not in imagelist:
            print(f'image {label} was not found')
        else:
            res.add(label)
    print(f'{mode}: {len(res)}')
    with open(mode + '.txt', 'w') as f:
        for basename in res:
            f.write(basename + '\n')
    