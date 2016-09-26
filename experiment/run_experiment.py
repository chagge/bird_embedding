'''
This code runs experiment and compare different algorithms 
Created on Sep 17, 2016

@author: liuli

'''
import numpy as np
import sys
import cPickle as pickle 

sys.path.append('../prepare_data/')
from extract_counts import load_sparse_coo

sys.path.append('../infer/')
from emb_model import * 
from learn_emb import * 
from context import *


import scipy.sparse as sparse
from extract_counts import load_sparse_coo
from separate_sets import read_pemb_file


if __name__ == "__main__":

    rseed = 27
    np.random.seed(rseed)

    data_dir = '../data/subset_pa_201407/'
    fold = int(sys.argv[1]) 
    print 'experiment on data fold %d' % fold

    counts_train = read_pemb_file(data_dir + 'counts_train_' + str(fold) + '.tsv')
    counts_valid = read_pemb_file(data_dir + 'counts_valid_' + str(fold) + '.tsv')
    counts_test  = read_pemb_file(data_dir + 'counts_test_'  + str(fold) + '.tsv')

    nspecies = max(counts_train.shape[1], counts_valid.shape[1], counts_test.shape[1])
    counts_train = np.c_[counts_train, np.zeros((counts_train.shape[0], nspecies - counts_train.shape[1]))]
    counts_valid = np.c_[counts_valid, np.zeros((counts_valid.shape[0], nspecies - counts_valid.shape[1]))]
    counts_test  = np.c_[counts_test,  np.zeros((counts_test.shape[0],  nspecies - counts_test.shape[1]))]

    obscov_train = np.loadtxt(data_dir + 'obscov_train_' + str(fold) + '.csv')
    obscov_valid = np.loadtxt(data_dir + 'obscov_valid_' + str(fold) + '.csv')
    obscov_test  = np.loadtxt(data_dir + 'obscov_test_'  + str(fold) + '.csv')

    val_ind = np.arange(counts_train.shape[0], counts_train.shape[0] + counts_valid.shape[0])
    counts_train = np.r_[counts_train, counts_valid]
    obscov_train = np.r_[obscov_train, obscov_valid]



    print 'The embeddint task has %d tuples, %d species' % (counts_train.shape[0], counts_train.shape[1])
    

    opt_config = dict(eta=0.01, max_iter=1000000,  print_niter=1000, min_improve=1e-4, display=1)
    model_config = dict(K=10, sigma2a=0.1, sigma2b=0.1, sigma2r=0.1, link_func='exp', intercept_term=True, scale_context=True)
    valid_config = dict(valid_ind=val_ind)
    config = dict(opt_config=opt_config, model_config=model_config, valid_config=valid_config)


    context_train = counts_train.copy()
    context_test = counts_test.copy()
    if config['model_config']['scale_context']:
        s = counts_pos95percent(context_train)
        s[s <= 0] = 1
        context_train = context_train / s
        context_test = context_test / s

    model = learn(counts_train, context_train, obscov_train, config)
    test_llh = emb_model(counts_test, context_test, obscov_test, model_config, model, dict(cal_grad=False, cal_obj=True))

    result = dict(test_llh=test_llh) 

    print 'Test log likelihood is ' + str(test_llh['llh'])

    with open('experiment' + str(fold) + '.p', 'w') as fp:
        pickle.dump(dict(config=config, result=result, model=model), fp)



