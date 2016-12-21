''' This file implements the inference method for the bird embedding ''' 

import numpy as np
import sys

from emb_model import *
sys.path.append('../prepare_data/')
from extract_counts import load_sparse_coo

def test_gradient(counts, context, obs_cov):
    
    sind = np.arange(2, 3)
    counts = counts[sind, :].reshape(len(sind), -1)
    context = context[sind, :].reshape(len(sind), -1)
    obs_cov = obs_cov[sind, :].reshape(len(sind), -1)
    counts = np.round(counts)

    nspecies = counts.shape[1]
    ncovar = obs_cov.shape[1]

    config = dict(learn_config=None)

    for downzero in xrange(0, 2):
        for use_obscov in xrange(0, 2):
            for intercept_term in xrange(0, 2):
                for link_func in ['exp', 'softplus']:
                    for zeroweight in [0, 0.1, 0.5, 1]:
                        for normalize_context in xrange(0, 2):

                            model_config = dict(downzero=downzero, use_obscov=use_obscov, zeroweight=zeroweight, 
                                                intercept_term=intercept_term, link_func=link_func, 
                                                normalize_context=normalize_context, 
                                                K = 10, sigma2a=0.1, sigma2r=1, sigma2b=10, scale_context=0)

                            config['model_config'] = model_config

                            emb_model = EmbModel(config)

                            K = model_config['K'] 
                            param_vec = np.random.rand(nspecies *(2 * K + model_config['intercept_term'] + 
                                                       (ncovar * model_config['use_obscov'] + 1) * model_config['downzero'])) - 0.2 

                            
                            res1 = emb_model.elbo(counts, context, obs_cov, param_vec)
                            mod1 = emb_model.eval_grad(counts, context, obs_cov, param_vec)

                            dalpha = 1e-8 * np.random.rand(nspecies * K)
                            drho   = 1e-8 * np.random.rand(nspecies * K)
                            drho0  = 1e-8 * np.random.rand(nspecies)
                            dbeta  = 1e-8 * np.random.rand(nspecies * ncovar)
                            dbeta0 = 1e-8 * np.random.rand(nspecies)
                            dparam = np.r_[dalpha, drho]
                            if model_config['intercept_term']:
                                dparam = np.r_[dparam, drho0]
                            
                            if model_config['downzero']:
                                if model_config['use_obscov']:
                                    dparam = np.r_[dparam, dbeta]
                                dparam = np.r_[dparam, dbeta0]
                            param_vec2 = param_vec + dparam

                            res2 = emb_model.elbo(counts, context, obs_cov, param_vec2)
                            mod2 = emb_model.eval_grad(counts, context, obs_cov, param_vec2)

                            diffv = res2['obj'] - res1['obj']
                            diffp = np.dot(res1['grad'], dparam) 
                            reld =  (diffv - diffp)  / np.abs(diffp)

                            print '--------------------------------------------'
                            print model_config
                            print 'value difference of numerical grad and analytical grad (elbo) '
                            print diffv
                            print diffp
                            print reld
                            assert(reld < 1e-5)

                            diffv = mod2['obj'] - mod1['obj']
                            diffp = np.dot(mod1['grad'], dparam) 
                            reld =  (diffv - diffp)  / np.abs(diffp)
                            print 'relative difference of numerical grad and analytical grad (model)'
                            print diffv
                            print diffp
                            print reld
                            assert(reld < 1e-5)
                     

if __name__ == "__main__":

    np.random.seed(6)
    data_dir = '../data/subset_pa_201407/'
    obs_cov = np.load(data_dir + 'obs_covariates.npy')
    counts = load_sparse_coo(data_dir + 'counts.npz')
    counts = counts.toarray()
    context = counts
 

    test_gradient(counts, context, obs_cov)
    raise Exception('Pass all gradient checks. Stop here')

    opt_config = dict(eta=0.005, max_iter=100000,  print_niter=1000, valid_frac=0.1, min_improve=1e-3, display=1)
    model_config = dict(K=10, sigma2a=10, sigma2b=10, sigma2r=10, link_func='exp', intercept_term=True)
    valid_config = dict(valid_frac=0.1)
    config = dict(opt_config=opt_config, model_config=model_config, valid_config=valid_config)

    learn_emb.learn(counts, context, obs_cov, config)
   
