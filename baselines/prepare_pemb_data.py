import numpy as np
import pandas as pd
import random
import sys

sys.path.append('../prepare_data/')

from extract_counts import load_sparse_coo

INPUT_DATA = '../data/subset_pa_201407/'

filename = INPUT_DATA + 'counts.npz'
sp_mat = load_sparse_coo(filename)

data = sp_mat.data
data = np.round(data)
data[data == 0] = 1
sp_mat.data = data

permind = np.arange(0, sp_mat.shape[0])
random.shuffle(permind)
train_ind = permind[0:(len(permind) / 2)]
test_ind = permind[(len(permind) / 2) : len(permind)]

train_mat = sp_mat.tocsr()[train_ind].tocoo()
test_mat = sp_mat.tocsr()[test_ind].tocoo()

train_quad = np.c_[np.ones((len(train_mat.data), 1)), train_mat.row, train_mat.col, train_mat.data]
test_quad = np.c_[np.ones((len(test_mat.data), 1)), test_mat.row, test_mat.col, test_mat.data]

np.savetxt(INPUT_DATA + 'train.tsv', train_quad, delimiter='\t', fmt='%d')
np.savetxt(INPUT_DATA + 'test.tsv', test_quad, delimiter='\t', fmt='%d')


