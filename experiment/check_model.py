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
from emb_model import emb_model

from experiment import config_to_filename


import scipy.sparse as sparse
from extract_counts import load_sparse_coo
from separate_sets import read_pemb_file


if __name__ == "__main__":

    rseed = 27
    np.random.seed(rseed)

    fold = 0 

    model_config = dict(K=10, sigma2a=100, sigma2r=100, sigma2b=100, link_func='exp', intercept_term=False, scale_context=True, downzero=True, use_obscov=True)
    filename = 'result/' + config_to_filename(model_config, fold=0)

    pkl_file = open(filename, 'rb') 
    output = pickle.load(pkl_file)
    model = output['model']
    config = output['config']
    test_llh = output['result']['test_llh']
    model_config = config['model_config']

    print 'alpha ---------------------------------------'
    print model['alpha']
    print 'rho ---------------------------------------'
    print model['rho']
    if model_config['intercept_term']:
        print 'rho0 ---------------------------------------'
        print model['rho0']
    if model_config['use_obscov']:
        print 'beta ---------------------------------------'
        print model['beta']
    if model_config['downzero']:
        print 'beta0 ---------------------------------------'
        print model['beta0']
        
    
    print 'Test log likelihood is ' + str(test_llh['llh'])
    print 'Test pos log likelihood is ' + str(test_llh['pos_llh'])




