''' This file implements the inference method for the bird embedding ''' 

import numpy as np
import scipy
import scipy.special as special
from scipy.stats import poisson
import sys

sys.path.append('../prepare_data/')
from extract_counts import load_sparse_coo

def softplus(x):
    y = x.copy()
    y[x < 20] = np.log(1 + np.exp(x[x < 20]))
    return y

def grad_softplus(x):
    y = special.expit(x)
    return y

#@profile
def infer_emb_model(counts, context, obs_cov, config, param, func_opt):
    
    ntuple = counts.shape[0]
    nspecies = counts.shape[1]
    ncovar = obs_cov.shape[1]
    K = config['K']

    # dispatch parameters
    alpha = param[0 : nspecies * K].reshape((nspecies, K)) 
    rho = param[nspecies * K : nspecies * K * 2].reshape((nspecies, K))
    if config['intercept_term']:
        rho0 = param[nspecies * K * 2 : nspecies * (K * 2 + 1)]
        beta = param[nspecies * (K * 2 + 1) : ].reshape((nspecies, ncovar))
    else:
        beta = param[nspecies * (K * 2) : ].reshape((nspecies, ncovar))
        rho0 = np.zeros(nspecies) 

    # observation probabilities for all checklists x species
    U = special.expit(obs_cov.dot(beta.T)) 
    
    # get the context, 
    # R has the following meaning: each row of R removing the j-th element is the context (weight) of the j-th species for all j. 
    # alternatively speaking: element j_1 contributes the same amount to species j_2 and j_3 in their respective contexts. 
    # weight is included in the context. 0 means the element is not in the context of any other species 
    # Lots of tricks to play with R 
    R = context 

    # calculate lambda
    H0 = R * (np.sum(alpha * rho, axis=1))
    H = R.dot(alpha).dot(rho.T) - H0
    if config['intercept_term']:
        H = H + rho0
    
    if config['link_func'] == 'exp':
        linkfunc = np.exp
        grad_linkfunc = np.exp
    else:
        linkfunc = softplus
        grad_linkfunc = grad_softplus

    epsilon = 0.01 
    Lamb = linkfunc(H) 
    Lamb = Lamb + epsilon

    obj = None
    grad = None
    llh = None

    if func_opt['cal_llh'] or func_opt['cal_obj']:
        #some llh values will be -inf because the corresponding elements of U is 1 and their poisson.pmf are 0 
        llh = np.sum(np.log((1 - U) + U * poisson.pmf(counts, Lamb))) / ntuple
        obj = 0.5 * np.sum(alpha * alpha) + 0.5 * np.sum(rho * rho) + 0.5 * np.sum(beta * beta) - llh

    if func_opt['cal_grad']:

        Q = 1 - (1 - U) / ((1 - U) + U * poisson.pmf(counts, Lamb))

        # Temp is gradient of obj w.r.t. H
        if linkfunc == np.exp:
            Temp = Q * (counts - Lamb)
        else:
            Temp = Q * (counts / Lamb - 1) * grad_linkfunc(H)

        if ntuple == 1:
            Temp3 = (R * Temp).squeeze()
            gelbo_alpha = (R.T).dot(Temp.dot(rho)) - Temp3[:, np.newaxis] * rho
            gelbo_rho = (Temp.T).dot(R.dot(alpha)) - Temp3[:, np.newaxis] * alpha 
        else: 
            Temp1 = (R.T).dot(Temp)
            np.fill_diagonal(Temp1, 0)
            gelbo_alpha = Temp1.dot(rho)
            gelbo_rho = (Temp1.T).dot(alpha)

        grad_alpha = (alpha - gelbo_alpha / ntuple).ravel()
        grad_rho = (rho - gelbo_rho / ntuple).ravel()

        gelbo_beta =   ((Q - U ).T).dot(obs_cov)  
        grad_beta = (beta - gelbo_beta / ntuple).ravel()

        if config['intercept_term']:
            grad_rho0 = - np.mean(Temp, axis=0)
            grad = np.r_[grad_alpha, grad_rho, grad_rho0, grad_beta]
        else:
            grad = np.r_[grad_alpha, grad_rho, grad_beta]

    return dict(obj=obj, grad=grad, llh=llh)

def learn_embedding(counts, context, obs_cov, config):

    ntuple = counts.shape[0]
    nspecies = counts.shape[1]
    ncovar = obs_cov.shape[1]
    K = config['K']

    # seperate out a validation set
    rindex = np.arange(ntuple)
    np.random.shuffle(rindex)
    nval = np.round(ntuple * config['valid_frac'])
    valind = rindex[0 : nval] 
    trind = rindex[nval : ]

    # initialize model parameters. no need to seperate now
    if config['intercept_term']:
        param = np.random.rand(nspecies *(2 * K + 1 + ncovar)) * 1e-5
    else:
        param = np.random.rand(nspecies *(2 * K + ncovar)) * 1e-5


    #calculate objective before start
    #func_opt = dict(cal_obj=True, cal_grad=False, cal_llh=False)
    #valobj =  infer_emb_model(counts[valind, :], context[valind, :], obs_cov[valind, :], param, config, func_opt)

    # set parameters for adagrad
    G = np.zeros(param.shape) 
    eta = config['eta'] 
    for it in xrange(1, config['max_iter'] + 1):
        
        # randomly select one instance and calculate gradient
        rind = np.random.choice(trind, size=1, replace=False)
        func_opt = dict(cal_obj=False, cal_grad=True, cal_llh=False)
        res = infer_emb_model(counts[rind, :].reshape((1, nspecies)), context[rind, :].reshape((1, nspecies)), obs_cov[rind, :].reshape((1, ncovar)), config, param, func_opt)
        grad = res['grad']

        # update G and model parameter
        G = G + grad * grad
        param = param - grad * eta / np.sqrt(G + 1e-8)

        # print opt objective on the validation set
        if it % 100 == 0:
            func_opt = dict(cal_obj=True, cal_grad=False, cal_llh=False)
            valres =  infer_emb_model(counts[valind, :], context[valind, :], obs_cov[valind, :], config, param, func_opt)
            print 'validation objective is ' + str(valres['obj'])
        
        # set stop condition
        # if stop_condition:
        #    break
    
    # dispatch optimized parameters

    alpha = param[0 : nspecies * K].reshape((nspecies, K)) 
    rho = param[nspecies * K : nspecies * K * 2].reshape((nspecies, K))
    if config['intercept_term']:
        rho0 = param[nspecies * K * 2 : nspecies * (K * 2 + 1)]
        beta = param[nspecies * (K * 2 + 1) : ].reshape((nspecies, ncovar))
        model = dict(alpha=alpha, rho=rho, rho0=rho0, beta=beta)
    else: 
        beta = param[nspecies * (K * 2) : ].reshape((nspecies, ncovar))
        model = dict(alpha=alpha, rho=rho, beta=beta)

    return model

def calculate_llh(counts, context, obs_cov, config, model):
    # a little argument check
    if config['intercept_term'] and ('rho0' not in model):
        raise Exception('The option "intercept term" is on, but there is no rho0 term in the model.')
    elif (not config['intercept_term']) and ('rho0' in model):
        raise Exception('The option "intercept term" is off, but there is a intercept term in the model.')

    if config['intercept_term']:
        param = np.r_[model['alpha'].ravel(), model['rho'].ravel(), model['rho0'], model['beta'].ravel()]
    else:
        param = np.r_[model['alpha'].ravel(), model['rho'].ravel(), model['beta'].ravel()]

    func_opt = dict(cal_obj=False, cal_grad=False, cal_llh=True)
    res = infer_emb_model(counts, context, obs_cov, config, param, func_opt)
    return res['llh'] 

def normalize_context(counts):

    #
    def nmlz(x):
        if (np.sum(x > 0) > 0):
            return np.percentile(x[x > 0], q=95)
        else:
            return 1
    #

    context = counts.copy()
    s = np.apply_along_axis(nmlz, axis=1, arr=context)
    context = context / s[:, np.newaxis]

    return context


def test_gradient(counts, context, obs_cov):

    nspecies = counts.shape[1]
    ncovar = obs_cov.shape[1]

    config = dict(intercept_term=True, link_func='softplus', valid_frac=0.1, K = 10)
    
    func_opt = dict(cal_obj=True, cal_grad=True, cal_llh=True)
    # intialize a parameter
    if config['intercept_term']:
        param = np.random.rand(nspecies *(2 * K + 1 + ncovar)) * 1e-1
    else:
        param = np.random.rand(nspecies *(2 * K + ncovar)) * 1e-1

    res1 = infer_emb_model(counts, context, obs_cov, config, param, func_opt=func_opt)

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

    res2 = infer_emb_model(counts, context, obs_cov, config, param2, func_opt=func_opt)
    diffv = res2['obj'] - res1['obj']
    diffp = np.dot(res1['grad'], dparam) 
    print 'value difference is '
    print diffv
    print 'first order difference'
    print diffp

if __name__ == "__main__":

    np.random.seed(6)
    data_dir = '../data/subset_pa_201407/'
    obs_cov = np.load(data_dir + 'obs_covariates.npy')
    counts = load_sparse_coo(data_dir + 'counts.npz')
    counts = counts.toarray()
    context = counts.astype(float)

    #test_gradient(counts, context, obs_cov)
    #raise Exception('Stop here')
   
    config = dict(intercept_term=True, link_func='exp', valid_frac=0.1, K=10)
    learn_embedding(counts, context, obs_cov, config)
   
