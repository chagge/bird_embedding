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

    
    ntuple = counts.shape[0]
    nspecies = counts.shape[1]
    if not obs_cov is None: 
        ncovar = obs_cov.shape[1]
    K = model_config['K']
    downzero = model_config['downzero']
    has_intercept = int(model_config['intercept_term'])

    # dispatch parameters
    alpha = param[0 : nspecies * K].reshape((nspecies, K)) 
    rho = param[nspecies * K : nspecies * K * 2].reshape((nspecies, K))
    tail_ind = nspecies * K * 2

    if model_config['intercept_term']:
        rho0 = param[tail_ind : tail_ind + nspecies]
        tail_ind = tail_ind + nspecies

    if downzero:
        # observation probabilities for all checklists x species
        if not obs_cov is None: 
            beta = param[tail_ind : tail_ind + nspecies * ncovar].reshape((nspecies, ncovar))
            tail_ind = tail_ind + nspecies * ncovar
            beta0 = param[tail_ind : ]
            U = special.expit(obs_cov.dot(beta.T) + beta0[None, :])
        else:
            beta0 = param[tail_ind : ]
            U = np.tile(special.expit(beta0), (ntuple, 1))
    
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

    epsilon = 0.001 
    Lamb = linkfunc(H) 
    Lamb = Lamb + epsilon

    llh = None
    pos_llh = None
    grad = None
    obj = None

    if func_opt['cal_obj']: 

        if downzero:
            ins_llh = np.log((1 - U) * (counts == 0) + U * poisson.pmf(counts, Lamb))
            ins_pos_llh = poisson.logpmf(counts[counts > 0], Lamb[counts > 0])
            ins_llh[counts > 0] = np.log(U[counts > 0]) + ins_pos_llh
        else:
            ins_llh = poisson.logpmf(counts, Lamb)
            ins_pos_llh = poisson.logpmf(counts[counts > 0], Lamb[counts > 0])

        llh = np.mean(ins_llh)
        pos_llh = np.mean(ins_pos_llh)
        obj = -llh

    if func_opt['cal_grad']:

        if linkfunc == np.exp:
            Temp = (counts - Lamb)
        else:
            Temp = (counts / Lamb - 1) * grad_linkfunc(H)

        if downzero:
            Q = 1 - (1 - U) / ((1 - U) + U * poisson.pmf(counts, Lamb))
            Q[counts > 0] = 1
            Temp = Q * Temp

            # Temp is gradient of obj w.r.t. H


        if ntuple == 1: # fast calculate if only one example is used to calculate the gradient
            Temp3 = (R * Temp).squeeze()
            gelbo_alpha = (R.T).dot(Temp.dot(rho)) - Temp3[:, np.newaxis] * rho
            gelbo_rho = (Temp.T).dot(R.dot(alpha)) - Temp3[:, np.newaxis] * alpha 

            gelbo_alpha = gelbo_alpha.ravel()
            gelbo_rho   = gelbo_rho.ravel()
        else: 
            Temp1 = (R.T).dot(Temp)
            np.fill_diagonal(Temp1, 0)
            gelbo_alpha = Temp1.dot(rho)
            gelbo_rho = (Temp1.T).dot(alpha)

            gelbo_alpha = (gelbo_alpha / (ntuple * nspecies)).ravel()
            gelbo_rho   = (gelbo_rho   / (ntuple * nspecies)).ravel()

        gelbo = np.r_[gelbo_alpha, gelbo_rho]

        if model_config['intercept_term']:
            gelbo_rho0 = np.mean(Temp, axis=0) / nspecies
            gelbo = np.r_[gelbo, gelbo_rho0]
        
        
        if downzero:
            Temp4 = Q - U 
            Temp4[counts > 0] = Temp4[counts > 0] + (1 - Q[counts > 0]) * U[counts > 0]
            if not obs_cov is None: 
                gelbo_beta = (Temp4.T).dot(obs_cov)  
                gelbo_beta  = (gelbo_beta  / (ntuple * nspecies)).ravel()
                gelbo = np.r_[gelbo, gelbo_beta]

            gelbo_beta0 =  np.mean(Temp4, axis=0) / nspecies
            gelbo = np.r_[gelbo, gelbo_beta0]

        grad = -gelbo

    return dict(llh=llh, obj=obj, grad=grad, pos_llh=pos_llh)


def emb_model(counts, context, obs_cov, model_config, param, func_opt):

    ntuple = counts.shape[0]
    nspecies = counts.shape[1]
    if not obs_cov is None:
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
        rho   = model['rho'].ravel()
        param = np.r_[alpha, rho]

        if model_config['intercept_term']:
            rho0 = model['rho0']
            param = np.r_[param, rho0]
        if model_config['downzero']:
            if not obs_cov is None:
                beta = model['beta'].ravel()
                param = np.r_[param, beta]

            beta0 = model['beta0']
            param = np.r_[param, beta0]

    elif isinstance(param, np.ndarray): # params in a vector 
        alpha = param[0 : nspecies * K]
        rho   = param[nspecies * K : nspecies * 2 * K]
        tail_ind = nspecies * 2 * K

        if model_config['intercept_term']:
            rho0 = param[tail_ind : tail_ind + nspecies]
            tail_ind = tail_ind + nspecies

        if model_config['downzero']:
            if not obs_cov is None:
                beta = param[tail_ind : tail_ind + nspecies * ncovar]
                tail_ind = tail_ind + nspecies * ncovar
            
            beta0 = param[tail_ind : tail_ind + nspecies] 
            tail_ind = tail_ind + nspecies

        assert(tail_ind == len(param))

    res = elbo(counts, context, obs_cov, model_config, param, func_opt)

    sigma2a = model_config['sigma2a'] 
    sigma2r = model_config['sigma2r'] 

    if model_config['downzero'] and (not obs_cov is None):
        sigma2b = model_config['sigma2b'] 

    obj = None
    reg = None
    if func_opt['cal_obj']:
        rega = 0.5 * np.sum(alpha * alpha) / sigma2a 
        regr = 0.5 * np.sum(rho * rho) / sigma2r 
        if model_config['downzero'] and (not obs_cov is None):
            regb = 0.5 * np.sum(beta * beta) / sigma2b
        else: 
            regb = 0

        reg = rega + regr + regb
        obj = reg  + res['obj']
    
    if func_opt['cal_grad']:
        # stack parameters into a vector
        grad_elbo = res['grad']

        grad_elbo[0 : nspecies * K] = grad_elbo[0 : nspecies * K] + alpha / sigma2a
        grad_elbo[nspecies * K : nspecies * 2 * K] = grad_elbo[nspecies * K : nspecies * 2 * K] + rho / sigma2r
        tail_ind = nspecies * 2 * K

        if model_config['intercept_term']:
            tail_ind = tail_ind + nspecies

        if model_config['downzero'] and (not obs_cov is None):
            grad_elbo[tail_ind : tail_ind + nspecies * ncovar] = grad_elbo[tail_ind : tail_ind + nspecies * ncovar] + beta / sigma2b

        grad = grad_elbo

    return dict(obj=obj, reg=reg, grad=grad, llh=res['llh'], pos_llh=res['pos_llh'])



