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

def elbo(obs_cov, counts, K, param, it):
    
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
    R = (counts > 0).astype(float)


    # calculate lambda
    alpha = alpha.reshape((nspecies, K))
    rho = rho.reshape((nspecies, K))
    
    H0 = np.dot(R, np.diag(np.sum(alpha * rho, axis=1)))
    H = np.dot(R, np.dot(alpha, rho.T)) - H0
    
    tempv = R[3, ].copy()
    tempv[5] = 0
    assert np.abs(H[3, 5] - np.dot(np.dot(tempv, alpha), rho[5, ])) < 1e-9, 'The matrix calculation is not correct'


    #if np.sum(np.isnan(U)) > 0:
    #    raise Exception('NaN value in U')

    #if np.sum(np.isnan(Lamb)) > 0:
    #    raise Exception('NaN value in Lambda')
    #if np.sum(np.isinf(Lamb)) > 0:
    #    raise Exception('Inf value in Lambda')
   

    #calculate q, the parameter of the variational parameter
    #Q = None
    #if Q is None:

    #if np.sum(np.isnan(Q)) > 0:
    #    raise Exception('NaN value in Lambda')

    epsilon = 1e-3
    Lamb = H
    Lamb[H < 20] = np.log(1 + np.exp(H[H < 20])) 
    Lamb = Lamb + epsilon

    Q = np.ones((ntuple, nspecies)) 
    flag = counts == 0
    Q[flag] = (U[flag] * np.exp(-Lamb[flag])) / (1 - U[flag] + U[flag] * np.exp(-Lamb[flag]))



    def sub_func(H, counts, R, Q):
        epsilon = 1e-3
        Lamb = H
        Lamb[H < 20] = np.log(1 + np.exp(H[H < 20])) 
        Lamb = Lamb + epsilon

        Temp = Q * (counts / Lamb - 1)  * expit(H)

        Temp1 = np.tile(np.sum(Temp * R, axis=0), (nspecies, 1))
        np.fill_diagonal(Temp1, 0)

        gllh_alpha = np.dot(Temp1, rho)
        grad_alpha = (alpha - gllh_alpha / ntuple).ravel()
        
        Temp2 = np.dot(Temp.T, R)
        np.fill_diagonal(Temp2, 0)
        gllh_rho = np.dot(Temp2, alpha)
        grad_rho = (rho - gllh_rho / ntuple).ravel()

        gllh_beta =  - np.dot(((1 - Q) * U).T, obs_cov) + np.dot((Q * (1 - U)).T, obs_cov) 
        grad_beta = beta - gllh_beta.ravel() / ntuple

        #grad_Q = np.log(1 - U[counts == 0]) - (- Lamb[counts == 0] + np.log(U[counts == 0])) + np.log(Q[counts==0]) - np.log(1 - Q[counts == 0]) 

        grad = np.r_[grad_alpha, grad_rho, grad_beta]

        norm = 0.5 * np.sum(alpha * alpha) + 0.5 * np.sum(rho * rho) + 0.5 * np.sum(beta * beta)
        llh = np.sum((1 - Q) * np.log(1 - U) + Q * (counts * np.log(Lamb) - Lamb + np.log(U))) 
        entropy = - np.sum((1 - Q[Q != 1]) * np.log(1 - Q[Q != 1])) - np.sum(Q[Q != 0] * np.log(Q[Q != 0]))
        obj = norm - (llh + entropy) / ntuple
        
        gradH = - Q * (counts / Lamb - 1)  * expit(H) / ntuple
        return dict(obj=obj, grad=grad, gradH=gradH)

    print 'Test gradient H'


    res1 = sub_func(H, counts, R, Q)
    obj1 = res1['obj']
    gradH = res1['gradH']


    dH = np.random.rand(H.shape[0], H.shape[1]) * 1e-9

    epsilon = 1e-3
    Lamb = H
    Lamb[H < 20] = np.log(1 + np.exp(H[H < 20])) 
    Lamb = Lamb + epsilon

    H = H + dH

    epsilon = 1e-3
    Lamb2 = H
    Lamb2[H < 20] = np.log(1 + np.exp(H[H < 20])) 
    Lamb2 = Lamb2 + epsilon

    print 
    print 'avg diff is ' + str(np.sum(Lamb2 - Lamb) / Lamb.shape[0] / Lamb.shape[1])
   
    res2 = sub_func(H, counts, R, Q)
    obj2 = res2['obj']
    print 'value diff:' + str(obj2 - obj1)
    print 'gradi diff:' + str(np.sum(dH * gradH))


    raise Exception('Test done')   

    sub_res = sub_func(H, counts, R)

    return dict(obj=sub_res['obj'], grad=sub_res['grad'])


def elbo_softplus(obs_cov, counts, param, it=-1):
    
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
    epsilon = 1e-4
    D = np.eye(nspecies, dtype=bool)
    A = np.zeros((nspecies, nspecies))
    A[~D] = theta
    Lamb = np.dot(counts, A.T) / alpha
    Lamb[Lamb < 20] = np.log(1 + np.exp(Lamb[Lamb < 20]))
    Lamb = Lamb * alpha + epsilon 

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
    
    M = Q * (counts / Lamb - 1) * Temp

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
    K = 10

    # intialize a parameter
    param = np.random.rand(nspecies *(2 * K + ncovar)) * 1e-3

    res1 = elbo(obs_cov, counts, K, param, 1)

    param2 = param.copy()

    dalpha = 1e-8 * np.random.rand(nspecies * K)
    dalpha[K + 1 : ] = 0
    drho =  0  * np.random.rand(nspecies * K)
    dbeta = 0 * np.random.rand(nspecies * ncovar)
    dparam = np.r_[dalpha, drho, dbeta] 
    param2 = param + dparam

    res2 = elbo(obs_cov, counts, K, param2, 1)
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

    test_gradient(obs_cov, counts)
    raise Exception('Stop here')
   
    ntuple = counts.shape[0]
    nspecies = counts.shape[1]
    ncovar = obs_cov.shape[1]

    param = np.random.rand(nspecies *(2 * nspecies - 2 + ncovar)) * 1e-4

    for it in xrange(1, 100000):
        
        #print('iteration ' + str(it) + ':' + str(max(param)))
        rind = np.random.randint(0, ntuple, size=1000) 
        
        if it == 2 and False:
            print('---------------------------------------------------------') 
            print(max(param))
        res = elbo_softplus(obs_cov[rind, :], counts[rind, :], param, it)
        grad = res['grad']
        
        if it == 2 and False:
            print('---------------------------------------------------------') 

        param = param - grad / (it + 1000)

        if it % 3 == 0:
            fullres = elbo_softplus(obs_cov[rind, :], counts[rind, :], param, it)
            print 'objective is ' + str(fullres['obj'])

    