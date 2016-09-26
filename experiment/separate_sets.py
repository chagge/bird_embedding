

import numpy as np
import sys
import scipy.sparse as sparse
sys.path.append('../prepare_data/')
from extract_counts import load_sparse_coo

def write_file_for_pemb(counts, filename):
    spmat = sparse.coo_matrix(counts)
    # write data sets into files
    quad = np.c_[np.ones((len(spmat.data), 1)), spmat.row, spmat.col, spmat.data]
    np.savetxt(filename, quad, delimiter='\t', fmt='%d')

def read_pemb_file(filename):
    quad = np.loadtxt(filename, delimiter='\t')
    counts = sparse.coo_matrix((quad[:, 3], (quad[:, 1], quad[:, 2]))).toarray()
    return counts


if __name__ == "__main__":

    rseed = 9
    np.random.seed(rseed)
    data_dir = '../data/subset_pa_201407/'
    obs_cov = np.load(data_dir + 'obs_covariates.npy')
    counts = load_sparse_coo(data_dir + 'counts.npz').toarray()


    # seperate the dataset into train, validation, and test sets
    tr_frac = 0.67
    val_frac = tr_frac * 0.1 

    index = np.arange(counts.shape[0])
    np.random.shuffle(index)
    # train
    ntr = np.round(counts.shape[0] * tr_frac)
    trind = index[0 : ntr] 
    # validation
    nval = np.round(counts.shape[0] * val_frac)
    valind = index[ntr : ntr + nval]
    # test
    stind = index[ntr + nval : ]

    # write the three datasets into files, which will be used by pemb
    write_file_for_pemb(counts[trind], data_dir + 'counts_train_' + str(rseed) + '.tsv')
    write_file_for_pemb(counts[valind], data_dir + 'counts_valid_' + str(rseed) + '.tsv')
    write_file_for_pemb(counts[stind], data_dir + 'counts_test_' + str(rseed) + '.tsv')

    # save obs covariates
    np.savetxt(data_dir + 'obscov_train_' + str(rseed) + '.csv', obs_cov[trind])
    np.savetxt(data_dir + 'obscov_valid_' + str(rseed) + '.csv', obs_cov[valind])
    np.savetxt(data_dir + 'obscov_test_' + str(rseed) + '.csv', obs_cov[stind])



