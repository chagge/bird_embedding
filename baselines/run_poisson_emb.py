import numpy as np
import pandas as pd
import random


INPUT_DATA = '../data/subset_pa_201407/'

filename = INPUT_DATA + 'counts.npz'
loader = np.load(filename)

data = loader.data
row = loader.row
col = loader.col
n = len(data)

data_mat = np.concatenate((np.reshape(data, (n, 1)), np.reshape(data, (n, 1)), np.reshape(data, (n, 1))), axis=1) 


permind = random.shuffle(range(0, len(data)))
train_ind = permind[xrange(0,  (len(data) / 2))]
test_ind = permind[xrange((len(data) / 2), len(permind))]

numpy.savetxt(INPUT_DATA + 'p_emb_train.tsv', data_mat[train_ind, :], delimiter='\t')
numpy.savetxt(INPUT_DATA + 'p_emb_test.tsv', data_mat[test_ind, :], delimiter='\t')


