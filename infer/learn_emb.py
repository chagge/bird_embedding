''' This file implements the inference method for the bird embedding ''' 


import numpy as np
import scipy
import scipy.special as special
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

def elbo(counts, context, obs_cov, K, param):
    
    ntuple = counts.shape[0]
    nspecies = counts.shape[1]
    ncovar = obs_cov.shape[1]

    alpha = param[0 : nspecies * K].reshape((nspecies, K)) 
    rho = param[nspecies * K : nspecies * K * 2].reshape((nspecies, K))
    rho0 = param[nspecies * K * 2 : nspecies * (K * 2 + 1)]
    beta = param[nspecies * (K * 2 + 1) : ].reshape((nspecies, ncovar))

    # observation probabilities for all checklists x species
    U = special.expit(np.dot(obs_cov, beta.T)) 
    
    # get the context, 
    # R has the following meaning: each row of R removing the j-th element is the context (weight) of the j-th species for all j. 
    # alternatively speaking: element j_1 contributes the same amount to species j_2 and j_3 in their respective contexts. 
    # weight is included in the context. 0 means the element is not in the context of any other species 
    # there is a lot of game to play with R 
    R = context 

    # calculate lambda
    H0 = np.dot(R, np.diag(np.sum(alpha * rho, axis=1)))
    H = np.dot(R, np.dot(alpha, rho.T)) - H0
    H = H + np.tile(rho0, (ntuple, 1))
    
    if np.sum(np.isnan(U)) > 0:
        raise Exception('NaN value in U')
    #calculate q, the parameter of the variational parameter
    if np.sum(np.isnan(H)) > 0:
        raise Exception('NaN value in H')

    linkfunc = np.exp #softplus
    grad_linkfunc = np.exp #grad_softplus

    epsilon = 0.01 
    Lamb = linkfunc(H) 
    Lamb = Lamb + epsilon

    if np.sum(np.isnan(Lamb)) > 0:
        raise Exception('NaN value in Lambda')


    if np.sum(np.isinf(Lamb)) > 0:
        raise Exception('Inf value in Lambda')
 
    Q = np.ones((ntuple, nspecies)) 
    flag = counts == 0
    Q[flag] = (U[flag] * np.exp(-Lamb[flag])) / (1 - U[flag] + U[flag] * np.exp(-Lamb[flag]))

    if np.sum(np.isnan(Q)) > 0:
        nanflag = np.isnan(Q)
        raise Exception('NaN value in Q')

    if linkfunc == np.exp:
        Temp = Q * (counts - Lamb)
    else:
        Temp = Q * (counts / Lamb - 1) * grad_linkfunc(H)

    Temp1 = np.dot(R.T, Temp)
    np.fill_diagonal(Temp1, 0)

    gllh_alpha = np.dot(Temp1, rho)
    grad_alpha = (alpha - gllh_alpha / ntuple).ravel()
    
    gllh_rho = np.dot(Temp1.T, alpha)
    grad_rho = (rho - gllh_rho / ntuple).ravel()

    grad_rho0 = - np.mean(Temp, axis=0)

    gllh_beta =  - np.dot(((1 - Q) * U).T, obs_cov) + np.dot((Q * (1 - U)).T, obs_cov) 
    grad_beta = (beta - gllh_beta / ntuple).ravel()

    grad = np.r_[grad_alpha, grad_rho, grad_rho0, grad_beta]

    norm = 0.5 * np.sum(alpha * alpha) + 0.5 * np.sum(rho * rho) + 0.5 * np.sum(beta * beta)
    flag0 = Q > 1e-10
    flag1 = Q < 1 - 1e-10
    llh = np.sum((1 - Q[flag1]) * np.log(1 - U[flag1])) + np.sum(Q[flag0] * (counts[flag0] * np.log(Lamb[flag0]) - Lamb[flag0] + np.log(U[flag0]))) 
    entropy = - np.sum((1 - Q[flag1]) * np.log(1 - Q[flag1])) - np.sum(Q[flag0] * np.log(Q[flag0]))
    obj = norm - (llh + entropy) / ntuple 

    return dict(obj=obj, grad=grad)

def learn_embedding(counts, context, obs_cov, K):

    ntuple = counts.shape[0]
    nspecies = counts.shape[1]
    ncovar = obs_cov.shape[1]

    param = np.random.rand(nspecies *(2 * K + 1 + ncovar)) * 1e-5
    rindex = np.arange(ntuple)
    np.random.shuffle(rindex)
    
    valind = rindex[0 : ntuple/5] 
    trind = rindex[ntuple/5 : ]

    valobj =  elbo(counts[valind, :], context[valind, :], obs_cov[valind, :], K, param)

    G = np.zeros(param.shape) 
    eta = 0.01
    for it in xrange(1, 10000):
        
        #print('iteration ' + str(it) + ':' + str(max(param)))
        rind = np.random.choice(trind, size=1, replace=False)
        res = elbo(counts[rind, :].reshape((1, nspecies)), context[rind, :].reshape((1, nspecies)), obs_cov[rind, :].reshape((1, ncovar)), K, param)

        grad = res['grad']
        obj = res['obj']

        G = G + grad * grad
        param = param - grad * eta / np.sqrt(G + 1e-8)

        if it % 10 == 0:
            #res = elbo(obs_cov, counts, K, param, it)
            valres =  elbo(counts[valind, :], context[valind, :], obs_cov[valind, :], K, param)
            print 'objective is ' + str(res['obj']) + '; validation objective is ' + str(valres['obj'])
            #if valres['obj'] > valobj:
            #    break

            valobj = valres['obj']

    alpha = param[0 : nspecies * K].reshape((nspecies, K)) 
    rho = param[nspecies * K : nspecies * K * 2].reshape((nspecies, K))
    rho0 = param[nspecies * K * 2 : nspecies * (K * 2 + 1)]
    beta = param[nspecies * K * 2 : ].reshape((nspecies, ncovar))

    model = dict(alpha=alpha, rho=rho, rho0=rho0, beta=beta)
    return model

 


def test_gradient(counts, context, obs_cov):

    nspecies = counts.shape[1]
    ncovar = obs_cov.shape[1]
    K = 10

    # intialize a parameter
    param = np.random.rand(nspecies *(2 * K + 1 + ncovar)) * 1e-3

    res1 = elbo(counts, context, obs_cov, K, param)

    param2 = param.copy()
    dalpha = 1e-8 * np.random.rand(nspecies * K)
    drho =  1e-8  * np.random.rand(nspecies * K)
    drho0 =  1e-8  * np.random.rand(nspecies)
    dbeta = 1e-8 * np.random.rand(nspecies * ncovar)
    dparam = np.r_[dalpha, drho, drho0, dbeta] 
    param2 = param + dparam

    res2 = elbo(counts, context, obs_cov, K, param2)
    diffv = res2['obj'] - res1['obj']
    diffp = np.dot(res1['grad'], dparam) 
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

    np.random.seed(6)
    data_dir = '../data/subset_pa_201407/'
    obs_cov = np.load(data_dir + 'obs_covariates.npy')
    counts = load_sparse_coo(data_dir + 'counts.npz').toarray()
    context = (counts > 0).astype(float)
    K = 10

    test_gradient(counts, context, obs_cov)
    raise Exception('Stop here')
   
    learn_embedding(counts, context, obs_cov, K)
   
