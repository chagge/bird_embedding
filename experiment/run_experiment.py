'''
This code runs experiment and compare different algorithms 
Created on Sep 17, 2016

@author: liuli

'''
import numpy as np
import sys

sys.path.append('../prepare_data/')
from extract_counts import load_sparse_coo

sys.path.append('../infer/')
from learn_emb import learn_embedding 
from learn_emb import calculate_llh
from learn_emb import normalize_context
from plot_emb import plot_bird 

if __name__ == "__main__":

    np.random.seed(4)
    data_dir = '../data/subset_pa_201407/'
    obs_cov = np.load(data_dir + 'obs_covariates.npy')
    counts = load_sparse_coo(data_dir + 'counts.npz').toarray()
    context = counts.copy() #normalize_context(counts)

    # seperate a test set
    index = np.arange(counts.shape[0])
    np.random.shuffle(index)
    ntr = np.round(counts.shape[0] * 0.67)
    trind = index[0 : ntr] 
    stind = index[ntr : ]

    config = dict(intercept_term=True, link_func='exp', valid_frac=0.1, K=10, max_iter=30000, eta=0.001)
    model = learn_embedding(counts[trind], context[trind], obs_cov[trind], config)
     
    print model.keys()
    test_llh = calculate_llh(counts[stind], context[stind], obs_cov[stind], config, model)

    print test_llh


