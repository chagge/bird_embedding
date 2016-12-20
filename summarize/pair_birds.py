''' This code plot the vectors of birds '''

import sys
import numpy as np
import cPickle as pickle 

import csv
import pandas 
from matplotlib import pyplot as plt
sys.path.append('../lib/tsne')
from tsne import tsne
sys.path.append('../experiment')
from experiment import config_to_filename
sys.path.append('../infer')
from emb_model import softplus
from scipy.stats import poisson
sys.path.append('../prepare_data/')
from separate_sets import read_pemb_file

def kldiv(prob1, prob2):
    kld = prob1 * (np.log(prob1) - np.log(prob2)) + (1 - prob1) * (np.log(1 - prob1) - np.log(1 - prob2))
    return kld

def bird_name_dict():
    taxonomy = pandas.read_csv('../data/taxonomy.csv', header=0) 
    bird_dict = dict(zip(taxonomy['SCI_NAME'], taxonomy['PRIMARY_COM_NAME']))
    return bird_dict

def score_birds(score, bird_names):
    ind = np.argsort(- score)
    names = [bird_names[i] for i in ind] 
    pairs = zip(names, score[ind])
    topten = pairs[0 : 10] 

    print('method & ' + ''.join([(name + ' & ') for name in names]) + '\\\\')

    return topten 



if __name__ == "__main__":

    data_dir = '../data/subset_pa/'
    fold = 0 

    # read in bird names
    bird_dict = bird_name_dict()
    fold_dir = data_dir + 'data_folds/' + str(fold) + '/'
    species_ind = np.loadtxt(fold_dir + 'nonzero_ind.csv', dtype=int)
    with open(data_dir + 'abd_bird_names.csv', 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        sci_names = spamreader.next()
    
    bird_names = [bird_dict[sci_names[ind]] for ind in species_ind]

    # read in data
    train_file = data_dir + 'data_folds/' + str(fold) + '/train.tsv'
    counts_train = read_pemb_file(train_file)
    occurance = (counts_train > 0).astype(float)
   

    # embedding from liping's model
    model_config = dict(K=20, sigma2a=100, sigma2r=100, sigma2b=100, 
                        link_func='softplus', intercept_term=False, 
                        scale_context=False, normalize_context=False,
                        downzero=True, use_obscov=True, zeroweight=1.0)

    filename = data_dir + 'result/' + config_to_filename(model_config, fold=fold)

    pkl_file = open(filename, 'rb') 
    output = pickle.load(pkl_file)
    model = output['emb_model'].model_param
    alpha = model['alpha']
    rho = model['rho']
    rho0 = np.zeros(rho.shape[0]) 
    #rho0 = model['rho0']

    raptors = ['Peregrine_Falcon', 'Bald_Eagle', 'Red-tailed_Hawk', 'Red-shouldered_Hawk', "Cooper's_Hawk", 
               'Sharp-shinned_Hawk', 'Broad-winged_Hawk']
    preys = ['Ovenbird', 'American_Robin'] 

    raptor_ind = [bird_names.index(name) for name in raptors]

    prey = 'House_Sparrow'
    prey_ind = bird_names.index(prey)

    print('Prey is %s' % prey)
   
    raptor_alpha = alpha[raptor_ind, :]
    raptor_rho = rho[raptor_ind, :]

    prey_alpha = alpha[prey_ind, :]
    prey_rho = rho[prey_ind, :]
   
    lamb = softplus(raptor_rho.dot(prey_alpha) + rho0[raptor_ind])
    nzprob = 1 - poisson.pmf(np.zeros(lamb.shape), lamb)
    co_score = nzprob * np.log(nzprob / (1 - nzprob))
    print('=================================================')
    print("co-purchase: compare contribution of alpha to other birds")
    print(score_birds(co_score, raptors))


    lamb = softplus(prey_rho.dot(raptor_alpha.T) + rho0[prey_ind])
    nzprob = 1 - poisson.pmf(np.zeros(lamb.shape), lamb)
    co_score = nzprob * np.log(nzprob / (1 - nzprob))
    print('=================================================')
    print("co-purchase: compare contribution of alpha to other birds")
    print(score_birds(co_score, raptors))


    lamb1 = softplus(np.tile(rho0[prey_ind], len(raptors)))
    nzprob1 = 1 - poisson.pmf(np.zeros(lamb1.shape), lamb1)
    lamb2 = softplus(prey_rho.dot(raptor_alpha.T) + rho0[prey_ind])
    nzprob2 = 1 - poisson.pmf(np.zeros(lamb2.shape), lamb2)
    kl_score = kldiv(nzprob1, nzprob2)
    print('=================================================')
    print("kl-divergence: kl divergence")
    print(score_birds(kl_score, raptors))

    lamb1 = softplus(rho0[raptor_ind])
    nzprob1 = 1 - poisson.pmf(np.zeros(lamb1.shape), lamb1)
    lamb2 = softplus(raptor_rho.dot(prey_alpha) + rho0[raptor_ind])
    nzprob2 = 1 - poisson.pmf(np.zeros(lamb2.shape), lamb2)
    kl_score = kldiv(nzprob1, nzprob2)
    print('=================================================')
    print("kl-divergence: kl divergence")
    print(score_birds(kl_score, raptors))



    stop()




    subject = raptors[5]
    ibird = bird_names.index(subject)

    mean_pos = np.mean(occurance, axis=0) 
    flag = occurance[:, ibird] > 0
    cond_mp = np.mean(occurance[flag, :], axis=0)
    #cond_mp[ibird] = 1e-9
    info_score = np.log(cond_mp / mean_pos)
    
    print('=================================================')
    print('a very basic measure')
    print(score_birds(info_score, bird_names))


    lamb = softplus(rho.dot(alpha[ibird, :]) + rho0)
    nzprob = 1 - poisson.pmf(np.zeros(lamb.shape), lamb)
    co_score = nzprob * np.log(nzprob / (1 - nzprob))
    print('=================================================')
    print("co-purchase: compare contribution of alpha to other birds")
    print(score_birds(co_score, bird_names))

    lamb = softplus(rho[ibird, :].dot(alpha.T) + rho0[ibird])
    nzprob = 1 - poisson.pmf(np.zeros(lamb.shape), lamb)
    co_score = nzprob * np.log(nzprob / (1 - nzprob))

    print('=================================================')
    print("co-purchase: compare the contribution of other birds' alpha")
    print(score_birds(co_score, bird_names))


    lamb1 = softplus(rho0)
    nzprob1 = 1 - poisson.pmf(np.zeros(lamb1.shape), lamb1)
    lamb2 = softplus(rho.dot(alpha[ibird, :]) + rho0)
    nzprob2 = 1 - poisson.pmf(np.zeros(lamb2.shape), lamb2)
    kl_score = kldiv(nzprob1, nzprob2)
    print('=================================================')
    print("kl-divergence: kl divergence")
    print(score_birds(kl_score, bird_names))

    lamb1 = softplus(rho0[ibird : ibird + 1])
    nzprob1 = 1 - poisson.pmf(np.zeros(lamb1.shape), lamb1)
    lamb2 = softplus(rho[ibird, :].dot(alpha.T) + rho0[ibird])
    nzprob2 = 1 - poisson.pmf(np.zeros(lamb2.shape), lamb2)
    kl_score = kldiv(nzprob1, nzprob2)
    print('=================================================')
    print("kl-divergence: kl divergence")
    print(score_birds(kl_score, bird_names))




