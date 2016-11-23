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

def bird_name_dict():
    taxonomy = pandas.read_csv('../data/taxonomy.csv', header=0) 
    bird_dict = dict(zip(taxonomy['SCI_NAME'], taxonomy['PRIMARY_COM_NAME']))
    return bird_dict

if __name__ == "__main__":

    # read in bird names
    bird_dict = bird_name_dict()
    fold_dir = data_dir + 'data_folds/' + str(fold) + '/'
    species_ind = np.loadtxt(fold_dir + 'nonzero_ind.csv', dtype=int)
    with open(data_dir + 'abd_bird_names.csv', 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        sci_names = spamreader.next()
    
    bird_names = [bird_dict[sci_names[ind]] for ind in species_ind]



    data_dir = '../data/subset_pa_201407/'
    fold = 0 

    # embedding from liping's model
    model_config = dict(K=20, sigma2a=100, sigma2r=100, sigma2b=100, 
                        link_func='softplus', intercept_term=True, 
                        scale_context=False, normalize_context=False,
                        downzero=True, use_obscov=True, zeroweight=1.0)

    filename = data_dir + 'result/' + config_to_filename(model_config, fold=fold)

    pkl_file = open(filename, 'rb') 
    output = pickle.load(pkl_file)
    model = output['emb_model'].model_param
    alpha = model['alpha']
    rho = model['rho']
    b = model['b']


    subject = 'Bald_Eagle'
    ibird = bird_names.index(subject)

    lamb = softplus(rho.dot(alpha[ibird, :]) + b)
    nzprob = 1 - poisson.pmf(np.zeros(lamb.shape), lamb)
    co_score = nzprob * log(nzprob / (1 - nzprob))

    ind = np.argsort(- co_score)
    co_names = [bird_names[i] for i in ind] 

    print(zip(co_names, co_score))





