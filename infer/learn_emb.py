''' This file implements the inference method for the bird embedding ''' 


import numpy as np
import scipy
import sys

sys.path.append('../prepare_data/')
from extract_counts import load_sparse_coo


def expit(x):
    y = np.ones(x.shape)
    y[x > -20] = 1 / (1 + np.exp(-x[x > -20]))
    return y

def elbo(counts, context, obs_cov, K, param):
    
    ntuple = counts.shape[0]
    nspecies = counts.shape[1]
    ncovar = obs_cov.shape[1]

    alpha = param[0 : nspecies * K] 
    rho = param[nspecies * K : nspecies * K * 2]
    beta = param[nspecies * K * 2 : ]

    # observation probabilities for all checklists x species
    U = expit(np.dot(obs_cov, beta.reshape((nspecies, ncovar)).T)) 
    
    # get the context, 
    # R has the following meaning: each row of R removing the j-th element is the context (weight) of the j-th species for all j. 
    # alternatively speaking: element j_1 contributes the same amount to species j_2 and j_3 in their respective contexts. 
    # weight is included in the context. 0 means the element is not in the context of any other species 
    # there is a lot of game to play with R 
    R = context 


    # calculate lambda
    alpha = alpha.reshape((nspecies, K))
    rho = rho.reshape((nspecies, K))
    
    H0 = np.dot(R, np.diag(np.sum(alpha * rho, axis=1)))
    H = np.dot(R, np.dot(alpha, rho.T)) - H0
    
    if np.sum(np.isnan(U)) > 0:
        raise Exception('NaN value in U')

    #calculate q, the parameter of the variational parameter


    epsilon = 1e-3
    Lamb = H.copy()
    Lamb[H < 20] = np.log(1 + np.exp(H[H < 20])) 
    Lamb = Lamb + epsilon

    if np.sum(np.isnan(Lamb)) > 0:
        raise Exception('NaN value in Lambda')
    if np.sum(np.isinf(Lamb)) > 0:
        raise Exception('Inf value in Lambda')
 
    Q = np.ones((ntuple, nspecies)) 
    flag = counts == 0
    Q[flag] = (U[flag] * np.exp(-Lamb[flag])) / (1 - U[flag] + U[flag] * np.exp(-Lamb[flag]))

    if np.sum(np.isnan(Q)) > 0:
        raise Exception('NaN value in Lambda')

    Temp = Q * (counts / Lamb - 1)  * expit(H)
    Temp1 = np.dot(R.T, Temp)

    np.fill_diagonal(Temp1, 0)
    gllh_alpha = np.dot(Temp1, rho)
    grad_alpha = (alpha - gllh_alpha / ntuple).ravel()
    
    gllh_rho = np.dot(Temp1.T, alpha)
    grad_rho = (rho - gllh_rho / ntuple).ravel()

    gllh_beta =  - np.dot(((1 - Q) * U).T, obs_cov) + np.dot((Q * (1 - U)).T, obs_cov) 
    grad_beta = beta - gllh_beta.ravel() / ntuple

    grad = np.r_[grad_alpha, grad_rho, grad_beta]

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

    param = np.random.rand(nspecies *(2 * K + ncovar)) * 1e-3
    rindex = np.arange(ntuple)
    np.random.shuffle(rindex)
    
    valind = rindex[0 : ntuple/5] 
    trind = rindex[ntuple/5 : ]

    valobj =  elbo(counts[valind, :], context[valind, :], obs_cov[valind, :], K, param)
    for it in xrange(1, 1000):
        
        #print('iteration ' + str(it) + ':' + str(max(param)))
        rind = np.random.choice(trind, size=50, replace=False)
        res = elbo(counts[rind, :], context[rind, :], obs_cov[rind, :], K, param)

        grad = res['grad']
        obj = res['obj']
        
        param = param - grad / (it + 100)

        if it % 50 == 0:
            #res = elbo(obs_cov, counts, K, param, it)
            valres =  elbo(counts[valind, :], context[valind, :], obs_cov[valind, :], K, param)
            print 'objective is ' + str(res['obj']) + '; validation objective is ' + str(valres['obj'])
            if valres['obj'] > valobj:
                break

            valobj = valres['obj']

    alpha = param[0 : nspecies * K] 
    rho = param[nspecies * K : nspecies * K * 2]
    beta = param[nspecies * K * 2 : ]

    model = dict(alpha=alpha, rho=rho, beta=beta)
    return model

 


def test_gradient(counts, context, obs_cov):

    nspecies = counts.shape[1]
    ncovar = obs_cov.shape[1]
    K = 10

    # intialize a parameter
    param = np.random.rand(nspecies *(2 * K + ncovar)) * 1e-3

    res1 = elbo(counts, context, obs_cov, K, param)

    param2 = param.copy()
    dalpha = 1e-8 * np.random.rand(nspecies * K)
    drho =  1e-8  * np.random.rand(nspecies * K)
    dbeta = 1e-8 * np.random.rand(nspecies * ncovar)
    dparam = np.r_[dalpha, drho, dbeta] 
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

    np.random.seed(4)
    data_dir = '../data/subset_pa_201407/'
    obs_cov = np.load(data_dir + 'obs_covariates.npy')
    counts = load_sparse_coo(data_dir + 'counts.npz').toarray()
    context = (counts > 0).astype(float)
    K = 10

    #test_gradient(counts, context, obs_cov)
    #raise Exception('Stop here')
   
    learn_embedding(counts, context, obs_cov, K)
   
