

import numpy as np
import sys
import os
import scipy.sparse as sparse
sys.path.append('../prepare_data/')
from extract_counts import load_sparse_coo

import matplotlib.pyplot as plt

if __name__ == "__main__":

   
    data_dir = '../data/subset_pa_201407/'
    obs_cov = np.load(data_dir + 'obs_covariates.npy')
    counts = load_sparse_coo(data_dir + 'counts.npz').toarray()


    species = 5 

    bin_width = 50
    max_count = max(counts[:, species])

    stop = np.ceil(float(max_count) / bin_width) * bin_width + 1

    freq, edges = np.histogram(counts[:, species], np.arange(0, stop, bin_width)) 

    print edges
    print max(counts[:, species])

    scale = 1

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%d' % np.round(np.power(height, scale)),
                    ha='center', va='bottom')

    fig, ax = plt.subplots(figsize=(12, 8))

    rects = ax.bar(np.arange(0, len(freq)), np.power(freq, 1.0 / scale), 0.5)
    autolabel(rects)
    ax.set_xticks(np.arange(0, len(edges)))
    ax.set_xticklabels(tuple(edges.astype(int)), fontsize=20)

    plt.show()    


