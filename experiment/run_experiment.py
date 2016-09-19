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
from plot_emb import plot_bird 

if __name__ == "__main__":

    np.random.seed(4)
    data_dir = '../data/subset_pa_201407/'
    obs_cov = np.load(data_dir + 'obs_covariates.npy')
    counts = load_sparse_coo(data_dir + 'counts.npz').toarray()
    context = np.sqrt(counts.copy()) #(counts > 0).astype(float)
    K = 2 

    #test_gradient(counts, context, obs_cov)
    #raise Exception('Stop here')
   
    model = learn_embedding(counts, context, obs_cov, K)
     
    print model['alpha']
    plot_bird(model['alpha']) 



