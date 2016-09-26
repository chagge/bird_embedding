import numpy as np
import random
import sys

sys.path.append('../prepare_data/')

from extract_counts import load_sparse_coo

INPUT_DATA = '../data/subset_pa_201407/'

filename = INPUT_DATA + 'counts.npz'
sp_mat = load_sparse_coo(filename)

# round float numbers, which is missing and filled with averages, to integers
data = sp_mat.data
data = np.round(data)
data[data == 0] = 1
sp_mat.data = data

# separate the data into three sets
permind = np.arange(0, sp_mat.shape[0])
random.shuffle(permind)
train_size = round(len(permind) * (0.9 * 0.67))
val_size = round(len(permind) * (0.1 * 0.67))
test_size = len(permind) - (train_size + val_size)  
train_ind = permind[0 : train_size]
val_ind = permind[train_size : train_size + val_size]
test_ind = permind[train_size + val_size : ]

train_mat = sp_mat.tocsr()[train_ind].tocoo()
val_mat = sp_mat.tocsr()[val_ind].tocoo()
test_mat = sp_mat.tocsr()[test_ind].tocoo()

# write data sets into files
train_quad = np.c_[np.ones((len(train_mat.data), 1)), train_mat.row, train_mat.col, train_mat.data]
val_quad = np.c_[np.ones((len(val_mat.data), 1)), val_mat.row, val_mat.col, val_mat.data]
test_quad = np.c_[np.ones((len(test_mat.data), 1)), test_mat.row, test_mat.col, test_mat.data]

np.savetxt(INPUT_DATA + 'train.tsv', train_quad, delimiter='\t', fmt='%d')
np.savetxt(INPUT_DATA + 'validation.tsv', val_quad, delimiter='\t', fmt='%d')
np.savetxt(INPUT_DATA + 'test.tsv', test_quad, delimiter='\t', fmt='%d')


