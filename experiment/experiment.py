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


def config_to_filename(model_config, fold):

    filename = ('experiment' + '_k' + str(int(model_config['K']))
                             + '_lf' + model_config['link_func']  
                             + '_sc' + str(int(model_config['scale_context'])) 
                             + '_it' + str(int(model_config['intercept_term'])) 
                             + '_dz'  + str(int(model_config['downzero'])) 
                             + '_uo' + str(int(model_config['use_obscov'])) 
                             + '_sa' + str(int(model_config['sigma2a'])) 
                             + '_sb' + str(int(model_config['sigma2b'])) 
                             + '_f' + str(fold) + '.pkl')

    return filename


def fold_learn(K=10, sigma2ar=1, sigma2b=1, link_func='exp', intercept_term=True, scale_context=False, downzero=True, use_obscov=True, fold=0):

    data_dir = '../data/subset_pa_201407/data_folds/' + str(fold) + '/'
    print 'Experiment on data fold %d' + data_dir 
    
    counts_train = read_pemb_file(data_dir + 'train.tsv')
    counts_valid = read_pemb_file(data_dir + 'validation.tsv')
    counts_test  = read_pemb_file(data_dir + 'test.tsv')

    nspecies = max(counts_train.shape[1], counts_valid.shape[1], counts_test.shape[1])
    counts_train = np.c_[counts_train, np.zeros((counts_train.shape[0], nspecies - counts_train.shape[1]))]
    counts_valid = np.c_[counts_valid, np.zeros((counts_valid.shape[0], nspecies - counts_valid.shape[1]))]
    counts_test  = np.c_[counts_test,  np.zeros((counts_test.shape[0],  nspecies - counts_test.shape[1]))]

    obscov_train = np.loadtxt(data_dir + 'obscov_train.csv')
    obscov_valid = np.loadtxt(data_dir + 'obscov_valid.csv')
    obscov_test  = np.loadtxt(data_dir + 'obscov_test.csv')

    val_ind = np.arange(counts_train.shape[0], counts_train.shape[0] + counts_valid.shape[0])
    counts_train = np.r_[counts_train, counts_valid]
    obscov_train = np.r_[obscov_train, obscov_valid]

    print 'The embeddint task has %d tuples, %d species' % (counts_train.shape[0], counts_train.shape[1])
    
    opt_config = dict(eta=0.002, max_iter=1000000, batch_size=1,  print_niter=2000, min_improve=1e-4, display=1)
    model_config = dict(K=K, sigma2a=sigma2ar, sigma2b=sigma2b, sigma2r=sigma2ar, link_func=link_func, intercept_term=intercept_term, scale_context=scale_context, downzero=downzero, use_obscov=use_obscov)
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
    
    test_llh = emb_model(counts_test[:, model['nonzero_ind']], context_test[:, model['nonzero_ind']], obscov_test, model_config, model, dict(cal_grad=False, cal_obj=True))

    result = dict(test_llh=test_llh) 

    print 'Test log likelihood is ' + str(test_llh['llh'])

    filename = 'result/' + config_to_filename(model_config, fold)

    with open(filename, 'w') as fp:
        pickle.dump(dict(config=config, result=result, model=model), fp)


if __name__ == "__main__":

    rseed = 27
    np.random.seed(rseed)

    fold_learn(K=10, sigma2ar=1, sigma2b=1, link_func='exp', intercept_term=True, scale_context=False, downzero=True, use_obscov=True, fold=2)

