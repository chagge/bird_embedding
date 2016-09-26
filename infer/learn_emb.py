''' This file implements the inference method for the bird embedding ''' 

import numpy as np
from emb_model import *

def learn(counts, context, obs_cov, config):

    ntuple = counts.shape[0]
    nspecies = counts.shape[1]
    ncovar = obs_cov.shape[1]
    model_config = config['model_config']
    opt_config = config['opt_config']
    K = model_config['K']
    has_intc = int(model_config['intercept_term'])


    # seperate out a validation set
    if config['valid_config'].has_key('valid_ind'):
        valind = config['valid_config']['valid_ind']
        trind = np.delete(np.arange(ntuple), valind)

    else: 
        nval = np.round(ntuple * config['valid_config']['valid_frac'])
        rindex = np.arange(ntuple)
        np.random.shuffle(rindex)
        trind = rindex[nval : ]
        valind = rindex[0 : nval] 

    # initialize model parameters. no need to seperate now
    param = np.random.rand(nspecies *(2 * K + has_intc + ncovar)) * 1e-2

    #calculate objective before start
    val_llh = np.zeros((opt_config['max_iter'] / 1000 + 2, 2), dtype=float)
    func_opt = dict(cal_obj=True, cal_grad=False, cal_llh=False)
    val_llh[0, 1] =  emb_model(counts[valind, :], context[valind, :], obs_cov[valind, :], model_config, param, dict(cal_grad=False, cal_obj=True))['llh']

    best_param = param
    best_llh = val_llh[0, 1]

    # initialize G for adagrad
    G = np.zeros(param.shape) 
    eta = opt_config['eta'] 
    ns = 1 # number of samples in each calculation of gradient
    for it in xrange(1, opt_config['max_iter'] + 1):

        # randomly select ns instance and calculate gradient
        rind = np.random.choice(trind, size=ns, replace=False)
        grad = emb_model(counts[rind, :].reshape((ns, nspecies)), context[rind, :].reshape((ns, nspecies)), obs_cov[rind, :].reshape((ns, ncovar)), model_config, param, dict(cal_obj=False, cal_grad=True))['grad']

        # adagrad updates 
        G = G + grad * grad
        sg = np.sqrt(G + 1e-6)
        param = param - grad * eta / sg
        
        # print opt objective on the validation set
        if it % opt_config['print_niter'] == 0:
            res = emb_model(counts[valind, :], context[valind, :], obs_cov[valind, :], model_config, param, dict(cal_obj=True, cal_grad=False))
            print 'validation obj and llh are  ' + str(res['obj']) + '\t' + str(res['llh'])

            if res['llh'] > best_llh:
                best_llh = res['llh']
                best_param = param

            ibat = it / opt_config['print_niter']
            val_llh[ibat, 0] = it
            val_llh[ibat, 1] = res['llh']

            if ibat >= 3:
                perf1 = np.mean(val_llh[ibat - 3 : ibat - 1, 1])
                perf2 = np.mean(val_llh[ibat - 1 : ibat + 1, 1])
                if (perf2 - perf1) < (np.abs(perf1) * opt_config['min_improve']):
                    break
                
     
    res = emb_model(counts[valind, :], context[valind, :], obs_cov[valind, :], model_config, param, dict(cal_obj=True, cal_grad=False))
    if res['llh'] > best_llh:
        best_llh = res['llh']
        best_param = param

    val_llh[ibat + 1, 0] = it 
    val_llh[ibat + 1, 1] = res['llh']
    
    # dispatch optimized parameters

    alpha = param[0 : nspecies * K].reshape((nspecies, K)) 
    rho = param[nspecies * K : nspecies * K * 2].reshape((nspecies, K))
    if model_config['intercept_term']:
        rho0 = param[nspecies * K * 2 : nspecies * (K * 2 + 1)]
        beta = param[nspecies * (K * 2 + 1) : ].reshape((nspecies, ncovar))
        model = dict(alpha=alpha, rho=rho, rho0=rho0, beta=beta, val_llh=val_llh)
    else: 
        beta = param[nspecies * (K * 2) : ].reshape((nspecies, ncovar))
        model = dict(alpha=alpha, rho=rho, beta=beta, val_llh=val_llh)

    return model


# gradient check in the middle of the program
##################################################
#            if it > 3000:
#                invsl[invsl < it] = it
#                
#                func_opt = dict(cal_obj=True, cal_grad=True)
#                tempp = param.copy() 
#                trres = elbo(counts[trind, :], context[trind, :], obs_cov[trind, :], config, tempp, func_opt)
#                
#                direction = np.zeros(tempp.shape)
#                ind = np.arange(0, nspecies)
#                #delta[ind] = - grad[ind] * s 
#                direction[ind] = (np.random.rand(len(ind)) - 0.5) 
#                direction = direction / np.linalg.norm(direction)
#                delta = 1e-9
#
#                trres2 = elbo(counts[trind, :], context[trind, :], obs_cov[trind, :], config, tempp + delta * direction, func_opt)
#                trres3 = elbo(counts[trind, :], context[trind, :], obs_cov[trind, :], config, tempp + 2 * delta * direction, func_opt)
#
#
#                print '-------------------------------------'
#                print 'value diff'
#                print trres2['obj'] - trres['obj']
#                print 'first order diff'
#                print trres['grad'].dot(delta * direction)
#                print 'delta norm'
#                print '2 order'
#                print (trres3['obj'] + trres['obj'] - 2 * trres2['obj']) / delta / delta 
#
#


#############################################################################################3333
# AdaGrad taking separate consideration of the regularization term
#        if config['adagrad_reg']:
#            #adagrad with norm
#            h = np.r_[np.repeat(eta / config['sigma2a'], nspecies * K), 
#                      np.repeat(eta / config['sigma2r'], nspecies * K), 
#                      np.zeros(nspecies * (int(config['intercept_term']))), 
#                      np.repeat(eta / config['sigma2b'], nspecies * ncovar)]
#            invsl = h + sg;
#            invsl[invsl < it] = it
#            param =  (param * sg - eta * grad) / invsl 
#

