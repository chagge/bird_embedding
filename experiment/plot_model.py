'''
This code runs experiment and compare different algorithms 
Created on Sep 17, 2016

@author: liuli

'''
import numpy as np
import sys
import cPickle as pickle 
from experiment import config_to_filename
import matplotlib.pyplot as plt
import seaborn as sns


def is_good_convergence(llh_seq):
    llh_seq1 = np.r_[- np.inf, llh_seq]
    llh_seq2 = np.r_[llh_seq, 0]

    if np.sum((llh_seq2 - llh_seq1) < 0) > 0:
        return False
    return True


def retrieve_result(model_config):
     
    llh = np.zeros(10)
    pos_llh = np.zeros(10)
    index_list = [0, 1, 2, 3, 4, 6, 7, 8] 

    for fold in index_list: #xrange(0, 10):
        filename = 'result/' + config_to_filename(model_config, fold)

        pkl_file = open(filename, 'rb') 
        output = pickle.load(pkl_file)
        test_llh = output['result']['test_llh']

        val_llh = output['model']['val_llh']
        val_llh_seq = val_llh[val_llh[:, 0] > 0, 1]

        #if is_good_convergence(val_llh_seq):
        #    raise Exception('Bad convergence ' + str(val_llh_seq) + ' with configuration ' + str(model_config) + ' on fold ' + str(fold))

        print '' + str(fold) + ': ' + str(test_llh)
    
        llh[fold] = test_llh['llh']
        pos_llh[fold] = test_llh['pos_llh']

    return dict(llh=llh, pos_llh=pos_llh)



def bar_plot(mean, std, legend_names):

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%1.2f' % height,
                    ha='center', va='bottom')


    M = mean.shape[0]
    N = mean.shape[1]

    ind = np.arange(N)  # the x locations for the groups
    width = 1.0 / (M + 1)   # the width of the bars

    colors = sns.color_palette("Paired") 

    plt.rcParams.update({'font.size': 22})


    fig, ax = plt.subplots(figsize=(12, 8))

    temp_list = []
    for tp in xrange(0, M):
        rects = ax.bar(ind + tp * width, mean[tp, :], width, color=colors[tp], yerr=std[tp, :], error_kw=dict(lw=2, capsize=5, capthick=2))
        autolabel(rects)
        temp_list.append(rects[0])


    plt.rcParams.update({'font.size': 18})
    ax.set_ylim([0, 12])
    ax.set_ylabel('negative log-likelihood', fontsize=20)
    ax.set_title('Predictive log-likelihood per item with different settings', fontsize=22)
    ax.set_xticks(ind + width)
    ax.set_xticklabels(('per item', 'per positive item'), fontsize=20)

    ax.legend(tuple(temp_list), tuple(legend_names), fontsize=20)

    plt.show()    

if __name__ == "__main__":

    #configs = []
    #configs.append(dict(K=10, sigma2a=100, sigma2r=100, sigma2b=100, link_func='exp', intercept_term=False, scale_context=True, downzero=False, use_obscov=False))
    #configs.append(dict(K=10, sigma2a=100, sigma2r=100, sigma2b=100, link_func='exp', intercept_term=False, scale_context=True, downzero=True, use_obscov=False))
    #configs.append(dict(K=10, sigma2a=100, sigma2r=100, sigma2b=100, link_func='exp', intercept_term=False, scale_context=True, downzero=True, use_obscov=True))

    #legend_names = ['no dw zeros', 'dw zeros w/o covariates', 'dw zeros w/ covariates']
    
    #configs = []
    #configs.append(dict(K=10, sigma2a=100, sigma2r=100, sigma2b=100, link_func='exp', intercept_term=False, scale_context=False, downzero=True, use_obscov=True))
    #configs.append(dict(K=10, sigma2a=100, sigma2r=100, sigma2b=100, link_func='exp', intercept_term=False, scale_context=True, downzero=True, use_obscov=True))
    #legend_names = ['not scale context', 'scale context']


    configs = []
    configs.append(dict(K=10, sigma2a=100, sigma2r=100, sigma2b=100, link_func='exp', intercept_term=False, scale_context=True, downzero=True, use_obscov=True))
    configs.append(dict(K=10, sigma2a=100, sigma2r=100, sigma2b=100, link_func='exp', intercept_term=True, scale_context=True, downzero=True, use_obscov=True))
    legend_names = ['no intercept', 'intercept']


    #configs = []
    #configs.append(dict(K=10, sigma2a=100, sigma2r=100, sigma2b=100, link_func='exp', intercept_term=True, scale_context=True, downzero=True, use_obscov=True))
    #configs.append(dict(K=10, sigma2a=100, sigma2r=100, sigma2b=100, link_func='softplus', intercept_term=True, scale_context=True, downzero=True, use_obscov=True))
    #legend_names = ['exp', 'softplus']

    #configs = []
    #configs.append(dict(K=5, sigma2a=100, sigma2r=100, sigma2b=100, link_func='exp', intercept_term=False, scale_context=True, downzero=True, use_obscov=True))
    #configs.append(dict(K=10, sigma2a=100, sigma2r=100, sigma2b=100, link_func='exp', intercept_term=False, scale_context=True, downzero=True, use_obscov=True))
    #configs.append(dict(K=20, sigma2a=100, sigma2r=100, sigma2b=100, link_func='exp', intercept_term=False, scale_context=True, downzero=True, use_obscov=True))

    #legend_names = ['K=5', 'K=10', 'K=20']

    #configs = []
    #configs.append(dict(K=10, sigma2a=1, sigma2r=100, sigma2b=100, link_func='exp', intercept_term=False, scale_context=True, downzero=True, use_obscov=True))
    #configs.append(dict(K=10, sigma2a=10, sigma2r=100, sigma2b=100, link_func='exp', intercept_term=False, scale_context=True, downzero=True, use_obscov=True))
    #configs.append(dict(K=10, sigma2a=100, sigma2r=100, sigma2b=100, link_func='exp', intercept_term=False, scale_context=True, downzero=True, use_obscov=True))

    #legend_names = ['sigma2=1', 'sigma2=10', 'sigma2=100']
 


    mean = np.zeros((len(configs), 2))
    std = np.zeros((len(configs), 2))

    for ival in xrange(0, len(configs)):

        model_config = configs[ival] 

        res = retrieve_result(model_config)
        mean[ival, 0] = np.mean(res['llh'])
        mean[ival, 1] = np.mean(res['pos_llh'])

        std[ival, 0] = np.std(res['llh'])
        std[ival, 1] = np.std(res['pos_llh'])

    mean = - mean

    bar_plot(mean, std, legend_names)


