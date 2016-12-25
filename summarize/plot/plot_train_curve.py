from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

lines = pickle.load(open('lines32.pkl', 'rb'))
font = {'size' : 28}
matplotlib.rc('font', **font)

xa = lines[0][0]
ya = - np.array(lines[0][1])

xb = lines[1][0]
yb = - np.array(lines[1][1])

xc = lines[2][0]
yc = - np.array(lines[2][1])

plt.ylim(0.3, 0.7)
plt.xlim(0, 2500000)
line1 = plt.plot(xa, ya, color='dimgrey', label='baseline', lw=2)
line2 = plt.plot(xb, yb, color='red', label='w/o feats', lw=2)
line3 = plt.plot(xc, yc, color='blue', label='w/ feats', lw=2)

plt.title('Negative log-likelihood of exposure model with and without features')
plt.ylabel('negative log-likelihood')
plt.xlabel('training mini-batches')
plt.legend()
plt.show()





