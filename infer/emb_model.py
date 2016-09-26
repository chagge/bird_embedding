''' This file implements the inference method for the bird embedding ''' 

import numpy as np
import scipy
import scipy.special as special
from scipy.stats import poisson
import sys

def softplus(x):
    y = x.copy()
    y[x < 20] = np.log(1 + np.exp(x[x < 20]))
    return y

def grad_softplus(x):
    y = special.expit(x)
    return y

#@profile
def elbo(counts, context, obs_cov, model_config, param, func_opt):

    #counts = counts.copy()
    #context = context.copy()
    #obs_cov = obs_cov.copy()
    #config = model_config.copy()
    #param = param.copy()
    #func_opt = func_opt.copy()

    
    ntuple = counts.shape[0]
    nspecies = counts.shape[1]
    ncovar = obs_cov.shape[1]
    K = model_config['K']

    # dispatch parameters
    alpha = param[0 : nspecies * K].reshape((nspecies, K)) 
    rho = param[nspecies * K : nspecies * K * 2].reshape((nspecies, K))
    if model_config['intercept_term']:
        rho0 = param[nspecies * K * 2 : nspecies * (K * 2 + 1)]
        beta = param[nspecies * (K * 2 + 1) : ].reshape((nspecies, ncovar))
    else:
        beta = param[nspecies * (K * 2) : ].reshape((nspecies, ncovar))
        rho0 = np.zeros(nspecies) 

    # observation probabilities for all checklists x species
    U = special.expit(obs_cov.dot(beta.T)) 
    
    # get the context, 
    R = context 

    # calculate lambda
    H0 = R * (np.sum(alpha * rho, axis=1))
    H = R.dot(alpha).dot(rho.T) - H0
    if model_config['intercept_term']:
        H = H + rho0
    
    if model_config['link_func'] == 'exp':
        linkfunc = np.exp
        grad_linkfunc = np.exp
    else:
        linkfunc = softplus
        grad_linkfunc = grad_softplus

    epsilon = 0.01 
    Lamb = linkfunc(H) 
    Lamb = Lamb + epsilon

    llh = None
    grad = None
    obj = None

    if func_opt['cal_obj']: 
        llh = np.sum(np.log((1 - U) + U * poisson.pmf(counts, Lamb))) / ntuple
        obj = -llh

    if func_opt['cal_grad']:

        Q = 1 - (1 - U) / ((1 - U) + U * poisson.pmf(counts, Lamb))

        # Temp is gradient of obj w.r.t. H
        if linkfunc == np.exp:
            Temp = Q * (counts - Lamb)
        else:
            Temp = Q * (counts / Lamb - 1) * grad_linkfunc(H)

        if ntuple == 1: # fast calculate if only one example is used to calculate the gradient
            Temp3 = (R * Temp).squeeze()
            gelbo_alpha = (R.T).dot(Temp.dot(rho)) - Temp3[:, np.newaxis] * rho
            gelbo_rho = (Temp.T).dot(R.dot(alpha)) - Temp3[:, np.newaxis] * alpha 
            gelbo_beta =   ((Q - U ).T).dot(obs_cov)  

            gelbo_alpha = gelbo_alpha.ravel()
            gelbo_rho   = gelbo_rho.ravel()
            gelbo_beta  = gelbo_beta.ravel()
        else: 
            Temp1 = (R.T).dot(Temp)
            np.fill_diagonal(Temp1, 0)
            gelbo_alpha = Temp1.dot(rho)
            gelbo_rho = (Temp1.T).dot(alpha)
            gelbo_beta = ((Q - U ).T).dot(obs_cov)  

            gelbo_alpha = (gelbo_alpha / ntuple).ravel()
            gelbo_rho   = (gelbo_rho   / ntuple).ravel()
            gelbo_beta  = (gelbo_beta  / ntuple).ravel()

        if model_config['intercept_term']:
            gelbo_rho0 = np.mean(Temp, axis=0)
            gelbo = np.r_[gelbo_alpha, gelbo_rho, gelbo_rho0, gelbo_beta]
        else:
            gelbo = np.r_[gelbo_alpha, gelbo_rho, gelbo_beta]

        grad = -gelbo

    return dict(llh=llh, obj=obj, grad=grad)


def emb_model(counts, context, obs_cov, model_config, param, func_opt):

    ntuple = counts.shape[0]
    nspecies = counts.shape[1]
    ncovar = obs_cov.shape[1]
    K = model_config['K']
    
    # handling two different format of model parameters
    if isinstance(param, dict): # params in a dict
        model = param
        # a little argument check
        if model_config['intercept_term'] and ('rho0' not in model):
            raise Exception('The option "intercept term" is on, but there is no rho0 term in the model.')
        elif (not model_config['intercept_term']) and ('rho0' in model):
            raise Exception('The option "intercept term" is off, but there is a intercept term in the model.')
        alpha = model['alpha'].ravel()
        beta  = model['beta'].ravel()
        rho   = model['rho'].ravel()
        if model_config['intercept_term']:
            rho0 = model['rho0']
            param = np.r_[alpha, rho, rho0, beta]
        else:
            param = np.r_[alpha, rho, beta]

    elif isinstance(param, np.ndarray): # params in a vector 
        alpha = param[0 : nspecies * K]
        rho   = param[nspecies * K : nspecies * 2 * K]
        if model_config['intercept_term']:
            rho0   = param[nspecies * 2 * K : nspecies * 2 * (K + 1)]
            beta  = param[nspecies * 2 * (K + 1) : ]
        else:
            beta  = param[nspecies * 2 * K : ]

    res = elbo(counts, context, obs_cov, model_config, param, func_opt)

    sigma2a = model_config['sigma2a'] 
    sigma2b = model_config['sigma2b'] 
    sigma2r = model_config['sigma2r'] 

    obj = None
    reg = None
    if func_opt['cal_obj']:
        rega = 0.5 * np.sum(alpha * alpha) / sigma2a 
        regr = 0.5 * np.sum(rho * rho) / sigma2r 
        regb = 0.5 * np.sum(beta * beta) / sigma2b
        reg = rega + regr + regb
        obj = reg  + res['obj']
    
    if func_opt['cal_grad']:
        # stack parameters into a vector
        if model_config['intercept_term']:
            temp = np.r_[alpha / sigma2a, rho / sigma2r, np.zeros(rho0.shape), beta / sigma2b]
        else:
            temp = np.r_[alpha / sigma2a, rho / sigma2r, beta / sigma2b]
        res['grad'] = temp + res['grad']

    return dict(obj=obj, reg=reg, grad=res['grad'], llh=res['llh'])



