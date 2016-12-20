'''
This code take a subset of ebird data by year, day range, and location 
Created on Aug 26, 2016

@author: liuli

'''

import pandas as pd
import numpy as np
import scipy.sparse as sparse
import csv


def save_sparse_coo(filename, smat):
    print(vars(smat))
    np.savez(filename, data=smat.data, row=smat.row, col=smat.col, shape=smat.shape)

def load_sparse_coo(filename):
    loader = np.load(filename)
    return sparse.coo_matrix((loader['data'], (loader['row'], loader['col'])),
                     shape = loader['shape'])


def denoise(x):

    x_back = x.copy()
    nzflag = x > 0 
    y = x[nzflag]

    #print "y------------------"
    #print y

    sind = np.argsort(y) 
    z = np.sort(y)

    #print "z------------------"
    
    #print z

    counter = np.arange(1, len(z) + 1)
    mean = np.cumsum(z).astype(float) / counter
    var = np.sqrt(np.cumsum(z * z).astype(float) / counter - mean * mean)

    #print "mean and var------------------"
    #print mean
    #print var

    for i in xrange((len(z) - 1), 0, -1):
        if z[i] > var[i-1] * 10 + mean[i - 1]:
            z[i] = mean[i - 1]
        else:
            break

    y[sind] = z

    x[nzflag] = y

    if not (i < (len(z)  - 1)) :
        if np.sum(x_back != x) > 0:
            raise Exception('Not recovering original array')
            
    return x

if __name__ == '__main__':
    
    folder = '../data/subset_pa/'

    print 'Reading in data ...'
    
    ebird_data = pd.read_csv(folder + 'obs_subset.csv')
    
    counts = ebird_data.loc[:, 'Zenaida_macroura':].as_matrix()
    
    def fill_positive_mean(column):
        pos_val = column[column > 0]
        if len(pos_val) > 0:
            pos_mean = np.mean(pos_val)
        else:
            pos_mean = 1 
        column[np.isnan(column)] = pos_mean
        return column
        
    print '\nFilling NaN values...'
    counts = np.apply_along_axis(fill_positive_mean, 0, counts)


    print 'remove rare species'
    abd_species = np.sum(counts > 0, axis=0) > 5 
    counts = counts[:, abd_species]

    with open('../data/bird_names.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        bird_names = reader.next()

    assert(len(bird_names) == 934)
    abd_names = [bird_names[i] for i in np.arange(0, len(abd_species))[abd_species]] 

    with open(folder + 'abd_bird_names.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(abd_names)


    print '\nRemoving outliers...'
    counts = np.apply_along_axis(denoise, 0, counts)


    path = folder + 'counts.npz'
    print '\nSaving to file ' + path + '...'
    smat = sparse.coo_matrix(counts)
    save_sparse_coo(path, smat)


    


    

