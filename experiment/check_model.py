'''
This code check the training result 
Created on Sep 17, 2016

@author: liuli

'''
import numpy as np
import sys
import cPickle as pickle 

sys.path.append('../prepare_data/')
from extract_counts import load_sparse_coo
sys.path.append('../infer/')
from emb_model import EmbModel 
from experiment import config_to_filename

import scipy.sparse as sparse
from separate_sets import read_pemb_file


if __name__ == "__main__":


    data_dir = '../data/subset_pa_201407/'
    fold = 0 

    model_config = dict(K=10, sigma2a=100, sigma2r=100, sigma2b=100, link_func='exp', intercept_term=False, scale_context=True, downzero=True, use_obscov=True)
    filename = data_dir + 'result/experiment_k10_lfexp_sc0_nc1_it0_dz0_pl1_uo0_sa10000_sb10_f0.pkl'# + config_to_filename(model_config, fold=fold)
    filename = 'compare.pkl'# + config_to_filename(model_config, fold=fold)

    pkl_file = open(filename, 'rb') 
    output = pickle.load(pkl_file)
    model = output['emb_model']
    train_log = output['train_log']

    test_res = output['test_res']
    
    print model.model_param['alpha']
    print test_res


