

import numpy as np
import sys
import os
import scipy.sparse as sparse
sys.path.append('../prepare_data/')
from extract_counts import load_sparse_coo

def write_file_for_pemb(counts, filename):
    spmat = sparse.coo_matrix(counts)
    # write data sets into files
    # pemb file format: user_id, item_id, session_id, count
    quad = np.c_[np.ones((len(spmat.data), 1)), spmat.col, spmat.row, spmat.data]
    np.savetxt(filename, quad, delimiter='\t', fmt='%d')

def read_pemb_file(filename):
    quad = np.loadtxt(filename, delimiter='\t')
    # pemb file format: user_id, item_id, session_id, count
    counts = sparse.coo_matrix((quad[:, 3], (quad[:, 2], quad[:, 1]))).toarray()
    return counts


if __name__ == "__main__":

    for fold in xrange(0, 10):
    
        np.random.seed(fold + 27)
        data_dir = '../data/subset_07/'

        print "use data in directory " + data_dir
        print "working on fold " + str(fold) + "..."

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



        fold_dir = data_dir + 'data_folds/' + str(fold) + '/'
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)

        nonzero_ind = np.where(np.sum(counts[trind], axis=0) > 0)[0]
        np.savetxt(fold_dir + 'nonzero_ind.csv', nonzero_ind, fmt='%d')

        counts = counts[:, nonzero_ind]

        # write the three datasets into files, which will be used by pemb
        write_file_for_pemb(counts[trind], fold_dir + '/train.tsv')
        write_file_for_pemb(counts[valind], fold_dir + '/validation.tsv')
        write_file_for_pemb(counts[stind], fold_dir + '/test.tsv')

        # save obs covariates
        np.savetxt(fold_dir + 'obscov_train.csv', obs_cov[trind])
        np.savetxt(fold_dir + 'obscov_valid.csv', obs_cov[valind])
        np.savetxt(fold_dir + 'obscov_test.csv', obs_cov[stind])

