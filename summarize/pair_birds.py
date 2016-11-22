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
from matplotlib.backends.backend_pdf import PdfPages

def bird_name_dict():
    taxonomy = pandas.read_csv('../data/taxonomy.csv', header=0) 
    bird_dict = dict(zip(taxonomy['SCI_NAME'], taxonomy['PRIMARY_COM_NAME']))
    return bird_dict

def bird_color_code(sci_names):
    taxonomy = pandas.read_csv('../data/taxonomy.csv', header=0) 
    order_dict = dict(zip(taxonomy['PRIMARY_COM_NAME'], taxonomy['ORDER_NAME']))
    orders = [order_dict[name] for name in sci_names]

    orderlist = list(set(orders))
    colorlist = ['rgb(146, 73,  0)',   'rgb(219, 209, 0)',   'rgb(36,  255, 36)', 
                 'rgb(255, 255, 109)', 'rgb(73,  0,   146)', 'rgb(0,   109, 219)', 
                 'rgb(182, 109, 255)', 'rgb(109, 182, 255)', 'rgb(182, 219, 255)', 
                 'rgb(255, 182, 119)', 'rgb(255, 109, 182)', 'rgb(0,   146, 146)', 
                 'rgb(0,   73,  73)',  'rgb(10,  10,  10)',  'rgb(146, 0,   0)',
                 'rgb(0,   91,  146)', 'rgb(73,  91,  109)', 'rgb(91, 128,   200)']

    colors = [colorlist[orderlist.index(order)] for order in orders] 

    return colors 


if __name__ == "__main__":

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
    

    

    score = rho.dot(alpha.T)
    diff = score - score.T

    sind = np.argsort(diff.flatten())
    
    subscript = np.unravel_index(sind, diff.shape)

    # read in bird names
    bird_dict = bird_name_dict()
    fold_dir = data_dir + 'data_folds/' + str(fold) + '/'
    species_ind = np.loadtxt(fold_dir + 'nonzero_ind.csv', dtype=int)
    with open(data_dir + 'abd_bird_names.csv', 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        sci_names = spamreader.next()
    
    bird_names = [bird_dict[sci_names[ind]] for ind in species_ind]


    ihawk = bird_names.index("American_Robin")
    chind = np.argsort(- score[ihawk, :])

    for i in xrange(0, 20):
        ibird = ihawk
        jbird = chind[i] 
        print "---------------------------------------------------"
        print "bird %s like %s with score: %f" % (bird_names[ibird], bird_names[jbird], rho[ibird, :].dot(alpha[jbird, :]))

        ibird = ibird + jbird
        jbird = ibird - jbird 
        ibird = ibird - jbird 
        print "bird %s like %s with score: %f" % (bird_names[ibird], bird_names[jbird], rho[ibird, :].dot(alpha[jbird, :]))




