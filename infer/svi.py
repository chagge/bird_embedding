''' This file implements the inference method for the bird embedding ''' 


import numpy as np
import scipy
import sys

sys.path.append('../prepare_data/')
from extract_counts import load_sparse_coo

def expit(x):
    return 1 / (1 + np.exp(-x))

def elbo(obs_cov, counts, param, it):
    
    ntuple = counts.shape[0]
    nspecies = counts.shape[1]
    ncovar = obs_cov.shape[1]

    theta = param[0 : nspecies * (nspecies - 1)] 
    beta = param[nspecies * (nspecies - 1) : ]

    # observation probabilities for all checklists x species
    U = expit(np.dot(obs_cov, beta.reshape((nspecies, ncovar)).T)) 
    
    # calculate lambda
    # fill theta to a matrix that has all diagnals zero
    D = np.eye(nspecies, dtype=bool)
    A = np.zeros((nspecies, nspecies))
    A[~D] = theta
    Emb = np.dot(counts, A.T)
    Lamb = np.exp(Emb)

    #calculate q, the parameter of the variational parameter
    #Q = None
    #if Q is None:
    Q = np.ones((ntuple, nspecies)) 
    flag = counts == 0
    Q[flag] = (U[flag] * np.exp(-Lamb[flag])) / (1 - U[flag] + U[flag] * np.exp(-Lamb[flag]))

    gllh_theta = np.dot((Q * counts).T, counts) - np.dot((Q * Lamb).T, counts) 
    grad_theta = theta - gllh_theta[~D] / ntuple

    gllh_beta =  - np.dot(((1 - Q) * U).T, obs_cov) + np.dot((Q * (1 - U)).T, obs_cov) 
    grad_beta = beta - gllh_beta.ravel() / ntuple

    #grad_Q = np.log(1 - U[counts == 0]) - (- Lamb[counts == 0] + np.log(U[counts == 0])) + np.log(Q[counts==0]) - np.log(1 - Q[counts == 0]) 

    #return dict(obj=obj, grad_theta=grad_theta, grad_beta=grad_beta, Q=Q, grad_Q = grad_Q)
    grad = np.r_[grad_theta, grad_beta]

    norm = 0.5 * np.sum(theta * theta) + 0.5 * np.sum(beta * beta)
    llh = np.sum((1 - Q) * np.log(1 - U) + Q * (counts * Emb - Lamb + np.log(U))) 
    entropy = - np.sum((1 - Q[Q != 1]) * np.log(1 - Q[Q != 1])) - np.sum(Q[Q != 0] * np.log(Q[Q != 0]))
    obj = norm - (llh + entropy) / ntuple

    return dict(obj=obj, grad=grad)


def elbo_softplus(obs_cov, counts, param, it):
    
    ntuple = counts.shape[0]
    nspecies = counts.shape[1]
    ncovar = obs_cov.shape[1]

    theta = param[0 : nspecies * (nspecies - 1)] 
    beta = param[nspecies * (nspecies - 1) : ]

    # observation probabilities for all checklists x species
    U = expit(np.dot(obs_cov, beta.reshape((nspecies, ncovar)).T)) 
    
    # calculate lambda
    # fill theta to a matrix that has all diagnals zero
    alpha = 1.0 
    D = np.eye(nspecies, dtype=bool)
    A = np.zeros((nspecies, nspecies))
    A[~D] = theta
    Lamb = np.dot(counts, A.T) / alpha
    Lamb[Lamb < 20] = np.log(1 + np.exp(Lamb[Lamb < 20]))
    Lamb = Lamb * alpha

    if np.sum(np.isnan(Lamb)) > 0:
        print 'Nan values in Lamb' 
    #calculate q, the parameter of the variational parameter
    #Q = None
    #if Q is None:
    Q = np.ones((ntuple, nspecies)) 
    flag = counts == 0
    Q[flag] = (U[flag] * np.exp(-Lamb[flag])) / (1 - U[flag] + U[flag] * np.exp(-Lamb[flag]))


    if np.sum(np.isnan(Q)) > 0:
        print 'Nan values in Q' 

    Temp = np.dot(counts, A.T) / alpha
    Temp[Temp <= 20] = np.exp(Temp[Temp <= 20]) / (1 + np.exp(Temp[Temp <= 20])) # all these elements are in (0, 1)
    Temp[Temp > 20] = 1 
    
    Lamb_trunc = Lamb
    Lamb_trunc[Lamb < 1e-6] = 1e-6
    M = Q * (counts / Lamb_trunc - 1) * Temp

    if np.sum(np.isnan(M)) > 0:
        print np.sum(np.isnan(Q))  
        print np.sum(np.isnan(Lamb))  
        print np.sum(np.isnan(counts))  
        print np.sum(np.isnan(Temp)) 
        print np.min(np.dot(counts, A.T) / alpha) 

        raise Exception('Divided by zero') 

    gllh_theta = np.dot(M.T, counts) 
    grad_theta = theta - gllh_theta[~D] / ntuple

    gllh_beta =  - np.dot(((1 - Q) * U).T, obs_cov) + np.dot((Q * (1 - U)).T, obs_cov) 
    grad_beta = beta - gllh_beta.ravel() / ntuple

    #grad_Q = np.log(1 - U[counts == 0]) - (- Lamb[counts == 0] + np.log(U[counts == 0])) + np.log(Q[counts==0]) - np.log(1 - Q[counts == 0]) 

    #return dict(obj=obj, grad_theta=grad_theta, grad_beta=grad_beta, Q=Q, grad_Q = grad_Q)
    grad = np.r_[grad_theta, grad_beta]

    norm = 0.5 * np.sum(theta * theta) + 0.5 * np.sum(beta * beta)
    llh = np.sum((1 - Q) * np.log(1 - U) + Q * (counts * np.log(Lamb) - Lamb + np.log(U))) 
    entropy = - np.sum((1 - Q[Q != 1]) * np.log(1 - Q[Q != 1])) - np.sum(Q[Q != 0] * np.log(Q[Q != 0]))
    obj = norm - (llh + entropy) / ntuple

    return dict(obj=obj, grad=grad)









def test_gradient(obs_cov, counts):

    nspecies = counts.shape[1]
    ncovar = obs_cov.shape[1]

    # intialize a parameter
    param = np.random.rand(nspecies *(nspecies - 1 + ncovar)) * 1e-4

    res1 = elbo_softplus(obs_cov, counts, param)

    param2 = param.copy()

    dtheta = 1e-8 * np.random.rand(nspecies * (nspecies - 1)) 
    param2[0 : nspecies * (nspecies - 1)] = param[0 : nspecies * (nspecies - 1)] + dtheta

    dbeta = 1e-8 * np.random.rand(nspecies * ncovar)
    param2[nspecies * (nspecies - 1) : ] = param[nspecies * (nspecies - 1) : ] + dbeta

    res2 = elbo_softplus(obs_cov, counts, param2) 
    diffv = res2['obj'] - res1['obj']
    diffp = np.dot(res1['grad'], np.r_[dtheta, dbeta]) 
    print 'value difference is '
    print diffv
    print 'first order difference'
    print diffp

    ## the follow part tests the derivative with respect to Q, which should be 0
    ## to test the following part, the program need to accept and return Q. Q does not depend on the data.
    ## add small permutation to Q and test the derivative with respect Q
    #Q = res1['Q'].copy()
    #Q[counts == 0] = Q[counts == 0] + 1e-8 * np.random.rand(np.sum(counts == 0)) 
    #Q[Q >= 1] = 1
    #Q[Q <= 0] = 0

    #dQ = (Q[counts == 0] - res1['Q'][counts == 0]).ravel()
    ## second order derivative with respect to Q
    #h = 1 / res1['Q'][counts == 0] + 1 / (1 - res1['Q'][counts == 0])

    #res3 = elbo(obs_cov, counts, param, Q) 

    #taylor2 = 0.5 * sum(dQ * h * dQ)
    #print 'Value difference is '
    #print res3['obj'] - res1['obj']
    #print 'The first order difference is zero'
    #print 'The second order difference is'
    #print taylor2

if __name__ == "__main__":

    np.random.seed(2)
    data_dir = '../data/subset_pa_201407/'
    obs_cov = np.load(data_dir + 'obs_covariates.npy')
    counts = load_sparse_coo(data_dir + 'counts.npz').toarray()

    #test_gradient(obs_cov, counts)
   
    ntuple = counts.shape[0]
    nspecies = counts.shape[1]
    ncovar = obs_cov.shape[1]

    param = np.random.rand(nspecies *(nspecies - 1 + ncovar)) * 1e-4

    for it in xrange(1, 100000):

        rind = np.random.randint(0, ntuple, size=10) 
        
        if it == 2 and False:
            print('---------------------------------------------------------') 
            print(max(param))
        res = elbo_softplus(obs_cov[rind, :], counts[rind, :], param, it)
        grad = res['grad']
        
        if it == 2 and False:
            print('---------------------------------------------------------') 

        param = param - grad / (it + 10)

        if it % 30 == 0:
            fullres = elbo_softplus(obs_cov[rind, :], counts[rind, :], param, it)
            print 'objective is ' + str(fullres['obj'])

    
