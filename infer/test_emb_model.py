''' This file implements the inference method for the bird embedding ''' 

import numpy as np
import sys

from emb_model import *
from context import *
import learn_emb 
sys.path.append('../prepare_data/')
from extract_counts import load_sparse_coo


def test_gradient(counts, context, obs_cov):

    #counts = counts[2].reshape((1, -1))
    #context = context[2].reshape((1, -1))
    #obs_cov = obs_cov[2].reshape((1, -1))

    nspecies = counts.shape[1]
    ncovar = obs_cov.shape[1]

    config = dict(intercept_term=True, link_func='softplus', valid_frac=0.1, K = 10, sigma2a=0.1, sigma2r=1, sigma2b=10)

    K = config['K'] 
    func_opt = dict(cal_obj=True, cal_grad=True, cal_llh=True)
    # intialize a parameter
    if config['intercept_term']:
        param = np.random.rand(nspecies *(2 * K + 1 + ncovar)) * 1e-1 - 0.05
    else:
        param = np.random.rand(nspecies *(2 * K + ncovar)) * 1e-1 - 0.05

    res1 = elbo(counts, context, obs_cov, config, param, func_opt=func_opt)
    mod1 = emb_model(counts, context, obs_cov, config, param, func_opt=func_opt)

    param2 = param.copy()
    dalpha = 1e-8 * np.random.rand(nspecies * K)
    drho =  1e-8  * np.random.rand(nspecies * K)
    drho0 =  1e-8  * np.random.rand(nspecies)
    dbeta = 1e-8 * np.random.rand(nspecies * ncovar)
    if config['intercept_term']:
        dparam = np.r_[dalpha, drho, drho0, dbeta] 
    else:
        dparam = np.r_[dalpha, drho, dbeta] 
    param2 = param + dparam

    res2 = elbo(counts, context, obs_cov, config, param2, func_opt=func_opt)
    mod2 = emb_model(counts, context, obs_cov, config, param2, func_opt=func_opt)

    diffv = res2['obj'] - res1['obj']
    diffp = np.dot(res1['grad'], dparam) 
    print 'elbo value difference is '
    print diffv
    print 'elbo first order difference'
    print diffp
    
    print '--------------------------------------------'

    diffv = mod2['obj'] - mod1['obj']
    diffp = np.dot(mod1['grad'], dparam) 
    print 'model value difference is '
    print diffv
    print 'model first order difference'
    print diffp

if __name__ == "__main__":

    np.random.seed(6)
    data_dir = '../data/subset_pa_201407/'
    obs_cov = np.load(data_dir + 'obs_covariates.npy')
    counts = load_sparse_coo(data_dir + 'counts.npz')
    counts = counts.toarray()
    context = counts_to_context(counts)
 

    #test_gradient(counts, context, obs_cov)
    #raise Exception('Stop here')

    opt_config = dict(eta=0.005, max_iter=100000,  print_niter=1000, valid_frac=0.1, min_improve=1e-3, display=1)
    model_config = dict(K=10, sigma2a=10, sigma2b=10, sigma2r=10, link_func='exp', intercept_term=True)
    valid_config = dict(valid_frac=0.1)
    config = dict(opt_config=opt_config, model_config=model_config, valid_config=valid_config)

    learn_emb.learn(counts, context, obs_cov, config)
   
