''' This file implements the inference method for the bird embedding ''' 

import numpy as np
import sys

from emb_model import *
from context import *
import learn_emb 
sys.path.append('../prepare_data/')
from extract_counts import load_sparse_coo


def test_gradient(counts, context, obs_cov):
    
    sind = np.arange(2, 30)
    counts = counts[sind, :].reshape(len(sind), -1)
    context = context[sind, :].reshape(len(sind), -1)
    obs_cov = obs_cov[sind, :].reshape(len(sind), -1)


    counts = np.round(counts)
    nspecies = counts.shape[1]

    ncovar = obs_cov.shape[1]

    for downzero in xrange(0, 2):
        for use_obscov in xrange(0, 2):
            for intercept_term in xrange(0, 2):
                for link_func in ['exp', 'softplus']:
                    config = dict(downzero=downzero, use_obscov=use_obscov, intercept_term=intercept_term, link_func=link_func, valid_frac=0.1, K = 10, sigma2a=0.1, sigma2r=1, sigma2b=10)

                    K = config['K'] 
                    func_opt = dict(cal_obj=True, cal_grad=True, cal_llh=True)
                    # intialize a parameter

                    param = np.random.rand(nspecies *(2 * K + config['intercept_term'] + (ncovar * config['use_obscov'] + 1) * config['downzero'])) * 1e-1 - 0.05

                    res1 = elbo(counts, context, obs_cov, config, param, func_opt=func_opt)
                    mod1 = emb_model(counts, context, obs_cov, config, param, func_opt=func_opt)

                    param2 = param.copy()
                    dalpha = 1e-8 * np.random.rand(nspecies * K)
                    drho   = 1e-8 * np.random.rand(nspecies * K)
                    drho0  = 1e-8 * np.random.rand(nspecies)
                    dbeta  = 1e-8 * np.random.rand(nspecies * ncovar)
                    dbeta0 = 1e-8 * np.random.rand(nspecies)

                    dparam = np.r_[dalpha, drho]
                    if config['intercept_term']:
                        dparam = np.r_[dparam, drho0]
                    
                    if config['downzero']:
                        if config['use_obscov']:
                            dparam = np.r_[dparam, dbeta]
                        dparam = np.r_[dparam, dbeta0]

                    param2 = param + dparam

                    res2 = elbo(counts, context, obs_cov, config, param2, func_opt=func_opt)
                    mod2 = emb_model(counts, context, obs_cov, config, param2, func_opt=func_opt)

                    diffv = res2['obj'] - res1['obj']
                    diffp = np.dot(res1['grad'], dparam) 
                    reld =  (diffv - diffp)  / np.abs(diffp)

                    print '--------------------------------------------'
                    print 'value difference of numerical grad and analytical grad (elbo) '
                    print diffv
                    print diffp
                    print reld
                    assert(reld < 1e-3)

                    diffv = mod2['obj'] - mod1['obj']
                    diffp = np.dot(mod1['grad'], dparam) 
                    reld =  (diffv - diffp)  / np.abs(diffp)
                    print 'relative difference of numerical grad and analytical grad (model)'
                    print diffv
                    print diffp
                    print reld
                    assert(reld < 1e-3)


if __name__ == "__main__":

    np.random.seed(6)
    data_dir = '../data/subset_pa_201407/'
    obs_cov = np.load(data_dir + 'obs_covariates.npy')
    counts = load_sparse_coo(data_dir + 'counts.npz')
    counts = counts.toarray()
    context = counts_to_context(counts)
 

    test_gradient(counts, context, obs_cov)
    raise Exception('Stop here')

    opt_config = dict(eta=0.005, max_iter=100000,  print_niter=1000, valid_frac=0.1, min_improve=1e-3, display=1)
    model_config = dict(K=10, sigma2a=10, sigma2b=10, sigma2r=10, link_func='exp', intercept_term=True)
    valid_config = dict(valid_frac=0.1)
    config = dict(opt_config=opt_config, model_config=model_config, valid_config=valid_config)

    learn_emb.learn(counts, context, obs_cov, config)
   
