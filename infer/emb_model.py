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

def pos95percent(x):
    if (np.sum(x > 0) > 0):
        return np.percentile(x[x > 0], q=95)
    else:
        return 0

    

class EmbModel:
    """class of embedding model"""
    model_config = None 
    learn_config = None
    model_param = None 
    context_scale = None

    def __init__(self, config):
        self.model_config = config['model_config']
        self.learn_config = config['learn_config']

    def initialize_param(self, model_param=None, param_vec=None, sizes=None):

        if not model_param == None:
            self.model_param = model_param
            return
        
        if not param_vec == None:
            self.model_param = self.__dispatch_param(param_vec, sizes)
            return

        K = self.model_config['K']
        nspecies = sizes['nspecies']

        self.model_param = dict()    
        self.model_param['alpha'] = (np.random.rand(nspecies, K) - 0.5) * 1e0
        self.model_param['rho'] = (np.random.rand(nspecies, K) - 0.5) * 1e0

        if self.model_config['intercept_term']:
            self.model_param['rho0'] = np.random.rand(nspecies) * 1e-2

        if self.model_config['downzero']:
            self.model_param['beta0'] = np.random.rand(nspecies) * 1e-2
            if self.model_config['use_obscov']:
                nvar = sizes['ncovar'] 
                self.model_param['beta'] = np.random.rand(nspecies, nvar) * 1e-2
    
        if self.model_config['scale_context']:
            context_scale = np.ones(nspecies) 
  
    def __collect_param(self, model_param, model_config): # params in a dict
            # a little argument check
            alpha = model_param['alpha'].ravel()
            rho   = model_param['rho'].ravel()
            param = np.r_[alpha, rho]
    
            if model_config['intercept_term']:
                rho0 = model_param['rho0']
                param = np.r_[param, rho0]
            if model_config['downzero']:
                if model_config['use_obscov']:
                    beta = model_param['beta'].ravel()
                    param = np.r_[param, beta]
    
                beta0 = model_param['beta0']
                param = np.r_[param, beta0]

            return param
    
    def __dispatch_param(self, param_vec, sizes): # params in a dict

            nspecies = sizes['nspecies'] 
            model_config = self.model_config 
            K = model_config['K']

            model_param = dict()
            model_param['alpha'] = param_vec[0 : nspecies * K].reshape((nspecies, K))
            model_param['rho']   = param_vec[nspecies * K : nspecies * 2 * K].reshape((nspecies, K))
            tail_ind = nspecies * 2 * K
    
            if model_config['intercept_term']:
                model_param['rho0'] = param_vec[tail_ind : tail_ind + nspecies]
                tail_ind = tail_ind + nspecies
    
            if model_config['downzero']:
                if model_config['use_obscov']:
                    ncovar = sizes['ncovar'] 
                    model_param['beta'] = param_vec[tail_ind : tail_ind + nspecies * ncovar].reshape((nspecies, ncovar))
                    tail_ind = tail_ind + nspecies * ncovar
                
                model_param['beta0'] = param_vec[tail_ind : tail_ind + nspecies] 
                tail_ind = tail_ind + nspecies
    
            assert(tail_ind == len(param_vec))

            return model_param 


    #@profile
    def elbo(self, counts, context, obs_cov, param, cal_obj=True, cal_grad=True):

        # set up constant numbers 
        ntuple = counts.shape[0]
        nspecies = counts.shape[1]
        ncovar=None
        if self.model_config['use_obscov']:
            ncovar = obs_cov.shape[1]
        K = self.model_config['K']


        # dispatch parameters if necessary
        model_param = param if isinstance(param, dict) else self.__dispatch_param(param, dict(nspecies=nspecies, ncovar=ncovar))

        if self.model_config['downzero']:
            # observation probabilities for all checklists x species
            beta0 = model_param['beta0']
            if self.model_config['use_obscov']: 
                beta = model_param['beta']
                U = special.expit(obs_cov.dot(beta.T) + beta0[None, :])
            else:
                U = np.tile(special.expit(beta0), (ntuple, 1))

        # get the context, 
        alpha = model_param['alpha']
        rho = model_param['rho']

        # calculate lambda
        R = context

        H0 = R * (np.sum(alpha * rho, axis=1))[None, :]
        H = R.dot(alpha).dot(rho.T) - H0

        # verify that matrix calculation is correct
        #tempi = 2
        #tempj = 0
        #print R.shape
        #tempc = R[tempi, :]
        #tempc[tempj] = 0
        #tempv = tempc.dot(alpha).dot(rho[tempj, :])
        #print tempv 
        #print H[tempi, tempj]
        #assert(np.abs(H[tempi, tempj] - tempv) < 1e-9)

        if self.model_config['normalize_context']:
            normalizer = np.sum(R, axis=1)[:, None] - R
            normalizer[normalizer == 0] = 1
            H = H / normalizer
        
        if self.model_config['intercept_term']:
            rho0 = model_param['rho0']
            H = H + rho0
        
        if self.model_config['link_func'] == 'exp':
            linkfunc = np.exp
            grad_linkfunc = np.exp
        else:
            linkfunc = softplus
            grad_linkfunc = grad_softplus
    
        epsilon = 0.00001 
        Lamb = linkfunc(H) 
        Lamb = Lamb + epsilon

        llh = None
        pos_llh = None
        grad = None
        obj = None
    
        if cal_obj: 
    
            if self.model_config['downzero']:
                ins_llh = np.log((1 - U) * (counts == 0) + U * poisson.pmf(counts, Lamb))
                ins_pos_llh = poisson.logpmf(counts[counts > 0], Lamb[counts > 0])
                ins_llh[counts > 0] = np.log(U[counts > 0]) + ins_pos_llh
                overall_weight = counts.size
            else:
                ins_llh = poisson.logpmf(counts, Lamb)
                ins_pos_llh = ins_llh[counts > 0]
                ins_llh[counts == 0] = ins_llh[counts == 0] * self.model_config['zeroweight']
                overall_weight = counts.size - np.sum(counts == 0) * (1 - self.model_config['zeroweight']) 

            pos_llh = np.mean(ins_pos_llh)
            llh = np.sum(ins_llh) / overall_weight 

            obj = -llh

            if np.isnan(llh):
                lnan = np.sum(np.isnan(Lamb))
                llhnan = np.sum(np.isnan(ins_llh))
                raise Exception('NaN value in log-likelihood! Some intermediate values: #nan in Lambda is ' + str(lnan) + ', and #nan in instance llh is ' + str(llhnan) + '.')
    
        if cal_grad:
    
            if linkfunc == np.exp:
                Temp = (counts - Lamb)
            else:
                Temp = (counts / Lamb - 1) * grad_linkfunc(H)
    

            grad_counts = ntuple * nspecies

            if self.model_config['downzero']:
                Q = 1 - (1 - U) / ((1 - U) + U * poisson.pmf(counts, Lamb))
                Q[counts > 0] = 1
                Temp = Q * Temp # Temp is gradient of obj w.r.t. H
                grad_counts = ntuple * nspecies
            else:
                Q = np.ones_like(counts)
                Q[counts == 0] = self.model_config['zeroweight']
                Temp = Q * Temp 
                grad_counts = np.sum(Q) 


            Dprod = Temp / normalizer if self.model_config['normalize_context'] else Temp

            if ntuple == 1: # fast calculate if only one example is used to calculate the gradient

                Temp3 = (R * Dprod).squeeze()
                gelbo_alpha = (R.T).dot(Dprod.dot(rho)) - Temp3[:, np.newaxis] * rho
                gelbo_rho = (Dprod.T).dot(R.dot(alpha)) - Temp3[:, np.newaxis] * alpha 
            else: 
                Temp1 = (R.T).dot(Dprod)
                np.fill_diagonal(Temp1, 0)
                gelbo_alpha = Temp1.dot(rho)
                gelbo_rho = (Temp1.T).dot(alpha)

            gelbo_alpha = gelbo_alpha.ravel()
            gelbo_rho   = gelbo_rho.ravel()
            gelbo = np.r_[gelbo_alpha, gelbo_rho]

            if self.model_config['intercept_term']:
                gelbo_rho0 = np.sum(Temp, axis=0) 
                gelbo = np.r_[gelbo, gelbo_rho0]
           
            if self.model_config['downzero']:
                Temp4 = Q - U 
                Temp4[counts > 0] = Temp4[counts > 0] + (1 - Q[counts > 0]) * U[counts > 0]
                if self.model_config['use_obscov']: 
                    gelbo_beta = (Temp4.T).dot(obs_cov)  
                    gelbo_beta  = gelbo_beta.ravel()
                    gelbo = np.r_[gelbo, gelbo_beta]

                gelbo_beta0 =  np.sum(Temp4, axis=0)
                gelbo = np.r_[gelbo, gelbo_beta0]

            else: # manually downweight zeros
                beta0_len = nspecies if self.model_config['downzero'] else 0 
                beta_len = nspecies * ncovar if (self.model_config['downzero'] and self.model_config['use_obscov']) else 0 
                gelbo = np.r_[gelbo, np.zeros(beta_len + beta0_len)]
 
    
            if grad_counts != 0:
                gelbo = gelbo / grad_counts
                
            grad = -gelbo
            
            if np.sum(np.isnan(grad)) > 0:
                qnan = np.sum(np.isnan(Q))
                raise Exception('NaN value in gradient! Some intermediate values: #nan in Q is ' + str(qnan))
    
        return dict(llh=llh, obj=obj, grad=grad, pos_llh=pos_llh)


    def eval_grad(self, counts, context, obs_cov, param, cal_obj=True, cal_grad=True):
    
        sizes = dict(nspecies=counts.shape[1])
        if self.model_config['use_obscov']: sizes['ncovar'] = obs_cov.shape[1]  

        model_param = param if isinstance(param, dict) else self.__dispatch_param(param, sizes)
        res = self.elbo(counts, context, obs_cov, model_param, cal_obj=cal_obj, cal_grad=cal_grad)

        model_config = self.model_config
        sigma2a = model_config['sigma2a'] 
        sigma2r = model_config['sigma2r'] 
        if self.model_config['downzero'] and self.model_config['use_obscov']:
            sigma2b = model_config['sigma2b'] 
    
        obj = None
        reg = None
        grad = None
        if cal_obj:

            reg = 0.5 * np.sum(model_param['alpha'] ** 2) / sigma2a \
                 + 0.5 * np.sum(model_param['rho'] ** 2) / sigma2r 

            if model_config['downzero'] and model_config['use_obscov']:
                reg = reg + 0.5 * np.sum(model_param['beta'] ** 2) / sigma2b
    
            obj = reg  + res['obj']
        
        if cal_grad:
            # stack parameters into a vector
            temp_param = model_param.copy()
            temp_param['alpha'] = temp_param['alpha'] / sigma2a
            temp_param['rho'] = temp_param['rho'] / sigma2r
            if model_config['downzero'] and model_config['use_obscov']:
                temp_param['beta'] = temp_param['beta'] / sigma2b
                
            # remove parameters that do not need regularization
            if model_config['downzero']:
                temp_param['beta0'] = np.zeros_like(temp_param['beta0']) 
            if model_config['intercept_term']: 
                temp_param['rho0'] = np.zeros_like(temp_param['rho0']) 
        
            grad_reg = self.__collect_param(temp_param, model_config)
            grad = res['grad'] + grad_reg
    
        return dict(obj=obj, reg=reg, grad=grad, llh=res['llh'], pos_llh=res['pos_llh'])


    def learn(self, counts, context, obs_cov):
        
        model_config = self.model_config 
        learn_config = self.learn_config 

        ntuple = counts.shape[0]
        nspecies = counts.shape[1]
        K = model_config['K']
        print 'Learning a embedding problem with %d checklists and %d species' % (ntuple, nspecies)
    
        # seperate out a validation set
        if learn_config.has_key('valid_ind'):
            valind = learn_config['valid_ind']
            trind = np.delete(np.arange(ntuple), valind)
        else: 
            nval = np.round(ntuple * learn_config['valid_frac'])
            rindex = np.arange(ntuple)
            np.random.shuffle(rindex)
            trind = rindex[nval : ]
            valind = rindex[0 : nval] 
        
        has_obscov = model_config['downzero'] and model_config['use_obscov']
        sizes = dict(nspecies=nspecies, ncovar=obs_cov.shape[1]) if has_obscov else dict(nspecies=nspecies)
        if self.model_param == None:
            self.initialize_param(sizes=sizes)

        if self.model_config['scale_context']:
            s = np.apply_along_axis(pos95percent, axis=0, arr=context)
            s[s <= 0] = 1
            self.context_scale = s
            context = context / s[None, :] 
        
        init_model_param = self.model_param.copy()

        model_param = self.model_param
        # get initialized param !!! model_param must be consistent with param_vec
        param_vec = self.__collect_param(model_param, model_config)
    
        # calculate objective before start
        obscov_val = obs_cov[valind, :] if model_config['use_obscov'] else None
        val_llh = np.zeros((learn_config['max_iter'] / learn_config['print_niter'] + 3, 2), dtype=float)
        # store value of the 0-th evaluation
        val_llh[0, 1] = self.eval_grad(counts[valind, :], context[valind, :], obscov_val, model_param, cal_obj=True, cal_grad=False)['llh']
    
        best_param = model_param 
        best_llh = val_llh[0, 1]
    
        # initialize G for adagrad
        G = np.zeros(param_vec.shape) 
        eta = learn_config['eta'] 
        ns = learn_config['batch_size'] # number of samples in each calculation of gradient
        for it in xrange(1, learn_config['max_iter'] + 1):
    
            # randomly select ns instance and calculate gradient
            rind = np.random.choice(trind, size=ns, replace=False)
            if (rind.ndim == 0): # if only select one instance, make the index an array
                rind.shape = (1) 
            batch_counts = counts[rind, :]
            batch_context = context[rind, :]
            batch_obs = obs_cov[rind, :] if model_config['use_obscov'] else None
            grad = self.eval_grad(batch_counts, batch_context, batch_obs, model_param, cal_obj=False, cal_grad=True)['grad']
    
            # adagrad updates 
            G = G + grad * grad
            sg = np.sqrt(G + 1e-6)
            param_vec = param_vec - grad * eta / sg
            model_param = self.__dispatch_param(param_vec, sizes)
            
            # print opt objective on the validation set
            if it % learn_config['print_niter'] == 0:
                res = self.eval_grad(counts[valind, :], context[valind, :], obscov_val, model_param, dict(cal_obj=True, cal_grad=False))
                print 'validation obj and llh are  ' + str(res['obj']) + '\t' + str(res['llh'])
    
                if res['llh'] > best_llh:
                    best_llh = res['llh']
                    best_param = model_param
    
                ibat = it / learn_config['print_niter']
                val_llh[ibat, 0] = it
                val_llh[ibat, 1] = res['llh']
    
                if ibat >= 3:
                    perf1 = np.mean(val_llh[ibat - 3 : ibat - 1, 1])
                    perf2 = np.mean(val_llh[ibat - 1 : ibat + 1, 1])
                    if (perf2 - perf1) < (np.abs(perf1) * learn_config['min_improve']):
                        break
                    
        res = self.eval_grad(counts[valind, :], context[valind, :], obscov_val, model_param, cal_obj=True, cal_grad=False)
        if res['llh'] > best_llh:
            best_llh = res['llh']
            best_param = model_param 
    
        val_llh[ibat + 1, 0] = it 
        val_llh[ibat + 1, 1] = res['llh']
        val_llh = val_llh[0 : ibat + 2, :]
        
        self.model_param = best_param
    
        retv = dict(init_model_param=init_model_param, val_llh_iter=val_llh)
        return retv

    def test(self, counts, context, obs_cov):

        if self.model_config['scale_context']:
            context = context / self.context_scale[None, :]

        res = self.eval_grad(counts, context, obs_cov, self.model_param, cal_obj=True, cal_grad=False)

        return res




