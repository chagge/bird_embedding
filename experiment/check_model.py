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

    fold = 0 #int(sys.argv[1]) 
    data_dir = '../data/subset_pa_201407/data_folds/' + str(fold) + '/'
    print 'experiment on data fold %d' % fold
    
    counts_train = read_pemb_file(data_dir + 'train.tsv')
    counts_valid = read_pemb_file(data_dir + 'validation.tsv')
    counts_test  = read_pemb_file(data_dir + 'test.tsv')

    nspecies = max(counts_train.shape[1], counts_valid.shape[1], counts_test.shape[1])
    counts_train = np.c_[counts_train, np.zeros((counts_train.shape[0], nspecies - counts_train.shape[1]))]
    counts_valid = np.c_[counts_valid, np.zeros((counts_valid.shape[0], nspecies - counts_valid.shape[1]))]
    counts_test  = np.c_[counts_test,  np.zeros((counts_test.shape[0],  nspecies - counts_test.shape[1]))]

    context_train = counts_train.copy()
    context_test = counts_test.copy()

    obscov_test  = np.loadtxt(data_dir + 'obscov_test.csv')


    pkl_file = open('experiment' + str(fold) + '.pkl', 'rb') 
    result = pickle.load(pkl_file)
    model = result['model']
    config = result['config']
    model_config = config['model_config']

    if config['model_config']['scale_context']:
        s = counts_pos95percent(context_train)
        s[s <= 0] = 1
        context_train = context_train / s
        context_test = context_test / s



    print 'alpha ---------------------------------------'
    print model['alpha']
    print 'rho ---------------------------------------'
    print model['rho']
    print 'rho0 ---------------------------------------'
    print model['rho0']
    print 'beta ---------------------------------------'
    print model['beta']
    
    test_llh = emb_model(counts_test[:, model['nonzero_ind']], context_test[:, model['nonzero_ind']], obscov_test, model_config, model, dict(cal_grad=False, cal_obj=True))

    print 'Test log likelihood is ' + str(test_llh['llh'])
    print 'Test pos log likelihood is ' + str(test_llh['pos_llh'])




