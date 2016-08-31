'''
This code take a subset of ebird data by year, day range, and location 
Created on Aug 26, 2016

@author: liuli

'''

import pandas as pd
import numpy as np
import scipy.sparse as sparse


def save_sparse_coo(filename, smat):
    print(vars(smat))
    np.savez(filename, data=smat.data, row=smat.row, col=smat.col, shape=smat.shape)

def load_sparse_coo(filename):
    loader = np.load(filename)
    return sparse.coo_matrix((loader['data'], (loader['row'], loader['col'])),
                     shape = loader['shape'])



if __name__ == '__main__':
    
    folder = '../data/subset_pa_201407/'

    print 'Reading in data ...'
    
    ebird_data = pd.read_csv(folder + 'obs_subset_y2014_d180-210.csv')
    
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

    path = folder + 'counts.npz'
    print '\nSaving to file ' + path + '...'
    smat = sparse.coo_matrix(counts)
    save_sparse_coo(path, smat)


    


    

