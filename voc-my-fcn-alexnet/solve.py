import caffe
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import surgery
import score

import numpy as np
import sys

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

weights = '../ilsvrc-nets/fcn-alexnet-pascal.caffemodel'

# init
# caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('./solver.prototxt')
# solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt('../data/bdd100k/seg/val.txt', dtype=str)

for _ in range(25):
    solver.step(4000)
    score.seg_tests(solver, False, val, layer='score')
