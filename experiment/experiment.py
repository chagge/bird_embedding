import numpy as np
import sys
import cPickle as pickle 
import os

sys.path.append('../prepare_data/')
from extract_counts import load_sparse_coo
sys.path.append('../infer/')
from emb_model import EmbModel 
sys.path.append('../util/')
from util import config_to_filename 

import scipy.sparse as sparse
from extract_counts import load_sparse_coo
from separate_sets import read_pemb_file


def fold_learn(cont_train=True, K=10, sigma2ar=1, sigma2b=1, 
               link_func='exp', intercept_term=True, 
               scale_context=False, normalize_context=True,
               downzero=True, use_obscov=True, zeroweight=1.0, 
               data_dir='../data/subset_pa/', fold=0):

    fold_dir = data_dir + 'data_folds/' + str(fold) + '/'

    print 'Experiment on data fold %d' + fold_dir 
    
    counts_train = read_pemb_file(fold_dir + 'train.tsv')
    counts_valid = read_pemb_file(fold_dir + 'validation.tsv')
    counts_test  = read_pemb_file(fold_dir + 'test.tsv')

    nspecies = max(counts_train.shape[1], counts_valid.shape[1], counts_test.shape[1])
    counts_train = np.c_[counts_train, np.zeros((counts_train.shape[0], nspecies - counts_train.shape[1]))]
    counts_valid = np.c_[counts_valid, np.zeros((counts_valid.shape[0], nspecies - counts_valid.shape[1]))]
    counts_test  = np.c_[counts_test,  np.zeros((counts_test.shape[0],  nspecies - counts_test.shape[1]))]

    obscov_train = np.loadtxt(fold_dir + 'obscov_train.csv')
    obscov_valid = np.loadtxt(fold_dir + 'obscov_valid.csv')
    obscov_test  = np.loadtxt(fold_dir + 'obscov_test.csv')

    #counts_valid = counts_test
    #obscov_valid = obscov_test

    #val_ind = np.arange(counts_train.shape[0], counts_train.shape[0] + counts_valid.shape[0])
    val_ind = np.random.choice(np.arange(0, counts_train.shape[0] + counts_valid.shape[0]), size=500, replace=False)

    counts_train = np.r_[counts_train, counts_valid]
    obscov_train = np.r_[obscov_train, obscov_valid]

    context_train = counts_train
    context_test = counts_test

    print 'The embeddint task has %d tuples, %d species' % (counts_train.shape[0], counts_train.shape[1])
    
    learn_config = dict(eta=0.02, max_iter=200000, batch_size=1,  print_niter=5000, min_improve=1e-3, display=1, valid_ind=val_ind)
    model_config = dict(cont_train=cont_train, K=K, sigma2a=sigma2ar, sigma2b=sigma2b, sigma2r=sigma2ar, 
                        link_func=link_func, intercept_term=intercept_term, 
                        scale_context=scale_context, normalize_context=normalize_context, 
                        downzero=downzero, use_obscov=use_obscov, zeroweight=zeroweight)
    config = dict(learn_config=learn_config, model_config=model_config)

    emb_model = EmbModel(config)
    #emb_model.sanity_check = True

    if ('cont_train' in model_config) and model_config['cont_train']:
        dummy_config = model_config.copy()        
        dummy_config['cont_train'] = False
        dummy_config['downzero'] = False 
        dummy_config['use_obscov'] = False 
        dummy_config['intercept_term'] = False 
        fname = data_dir + 'result/' + config_to_filename(dummy_config, fold)
        loader = pickle.load(open(fname))
        init_model = loader['emb_model'].model_param
    else:
        init_model = None

    train_log = emb_model.learn(counts_train, context_train, obscov_train, init_model)
    test_res = emb_model.test(counts_test, context_test, obscov_test)

    result = dict(train_log=train_log, test_res=test_res, emb_model=emb_model) 
    filename = data_dir + 'result/' + config_to_filename(model_config, fold)

    # create folder if not exist
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(filename, 'w') as fp:
        pickle.dump(result, fp)



if __name__ == "__main__":

    rseed = 27
    np.random.seed(rseed)

    fold_learn(cont_train=True, K=16, sigma2ar=100, sigma2b=100, 
               link_func='softplus', intercept_term=True, 
               scale_context=False, normalize_context=False,
               downzero=True, use_obscov=True, zeroweight=1.0, 
               data_dir='../data/subset_pa/', fold=0)


