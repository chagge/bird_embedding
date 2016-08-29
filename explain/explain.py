import glob
import os
import numpy as np
import pandas as pd

import csv
from scipy.spatial import distance
from matplotlib import pyplot as plt
from adjustText import adjust_text
import itertools

n_components = 3

with open('../data/bird_names.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    bird_names = spamreader.next()

save_dir = '../data/result/cofactor12-14/'

n_params = len(glob.glob(os.path.join(save_dir, 'CoFacto_K' + str(n_components) + '_iter*.npz')))
params = np.load(os.path.join(save_dir, 'CoFacto_K%d_iter%d.npz' % (n_components, n_params - 1)))
V = params['V']

icx = 1
icy = 2
d = np.sqrt(V[:, icx] * V[:, icx] + V[:, icy] * V[:, icy])
threshold = 0.02
flag = d > threshold 

selected = [s for (s, b) in zip(bird_names, flag) if b]  
compx = V[flag, icx]
compy = V[flag, icy]

plt.scatter(compx, compy)
#for label, x, y in zip(selected, compx, compy):
#    plt.annotate(label, xy = (x, y), xytext = (0, 0), textcoords = 'offset points', ha = 'right', va = 'bottom')

texts = []
for xt, yt, s in zip(compx, compy, selected):
    if xt < 0.01:
        texts.append(plt.text(xt, yt, s, size=8, ha = 'right', va = 'bottom'))
    else:
        texts.append(plt.text(xt, yt, s, size=8, ha = 'left', va = 'bottom'))

plt.show()

#print [bird_names[i] for i in np.where(V[:, 0] > 0.1)[0]]
print [bird_names[i] for i in np.where(V[:, 1] < -0.08)[0]]








## calculate distance 
#bird_dist = distance.squareform(distance.pdist(V))
#
## similar species 
#sp = [0, 5, 20, 365, 800]
#
#np.fill_diagonal(bird_dist, np.inf) 
#sim_sp = np.argmin(bird_dist[sp, :], axis=1)
#
#
#
#
#
#print('Similar species pairs:')
#print(zip([bird_names[i] for i in sp], [bird_names[i] for i in sim_sp]))
#
#
#np.fill_diagonal(bird_dist, -np.inf) 
#dif_sp = np.argmax(bird_dist[sp, :], axis=1)
#print('Different species pairs:')
#print(zip([bird_names[i] for i in sp], [bird_names[i] for i in dif_sp]))
#
#
#
#
#
#
#
#
#
