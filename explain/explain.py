#import itertools
import glob
import os
#import sys
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#
import pandas as pd
#from scipy import sparse
#import seaborn as sns
#sns.set(context="paper", font_scale=1.5, rc={"lines.linewidth": 2}, font='DejaVu Serif')
#
#
#sys.path.append('../../cofactor/src/')
#import cofacto
#import rec_eval 

import csv
from scipy.spatial import distance
import matplotlib.pyplot as plt

n_components = 20


#train_data = pd.read_csv('../data/aggregated_obs_2005.csv')
#head = train_data[train_data.INDEX == -1]
#names = head.iloc[:, 1:]
#names.to_csv('../data/bird_names.csv', index=False)

with open('../data/bird_names.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    bird_names = spamreader.next()

save_dir = os.path.join('../data/', 'ML20M_ns%d_scale%1.2E' % (1, 0.03))

n_params = len(glob.glob(os.path.join(save_dir, 'CoFacto_K' + str(n_components) + '_iter*.npz')))
params = np.load(os.path.join(save_dir, 'CoFacto_K%d_iter%d.npz' % (n_components, n_params - 1)))
V = params['V']

plt.scatter(V[:, 0], V[:, 1])
plt.show()

#print [bird_names[i] for i in np.where(V[:, 0] < -0.1)[0]]
print [bird_names[i] for i in np.where(V[:, 1] > 0.12)[0]]








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
