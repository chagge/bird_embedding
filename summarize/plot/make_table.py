'''
This code plot experiment results
Created on Sep 17, 2016

@author: liuli

'''
import numpy as np
import sys
import cPickle as pickle 

sys.path.append('../experiment/')
from experiment import config_to_filename
import matplotlib.pyplot as plt
import seaborn as sns
import itertools as itl


def is_good_convergence(llh_seq):
    llh_seq1 = np.r_[- np.inf, llh_seq]
    llh_seq2 = np.r_[llh_seq, 0]

    if np.sum((llh_seq2 - llh_seq1) < 0) > 0:
        return False
    return True


def retrieve_result(data_dir, model_config):
     
    llh = np.zeros(10)
    pos_llh = np.zeros(10)
    #index_list = [0, 1, 2, 3, 4, 6, 7, 8] 
    index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 

    for fold in index_list: #xrange(0, 10):
        filename = data_dir + 'result/' + config_to_filename(model_config, fold)

        pkl_file = open(filename, 'rb') 
        output = pickle.load(pkl_file)
        test_llh = output['test_res']

        #val_llh = output['model']['val_llh']
        #val_llh_seq = val_llh[val_llh[:, 0] > 0, 1]

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
    ax.set_ylim([0, np.max(mean + std) * 1.28])
    ax.set_ylabel('negative log-likelihood', fontsize=20)
    ax.set_title('Predictive log-likelihood per count with different settings', fontsize=22)
    ax.set_xticks(ind + width)
    ax.set_xticklabels(('per count', 'per positive count'), fontsize=20)

    ax.legend(tuple(temp_list), tuple(legend_names), fontsize=20)

    plt.show()    

if __name__ == "__main__":

    data_dir = '/rigel/dsi/users/ll3105/bird_data/subset_pa_201407/'

    base_config = dict(K=10, sigma2a=100, sigma2r=100, sigma2b=100, link_func='exp', intercept_term=False, 
                        scale_context=False, normalize_context=False, downzero=False, use_obscov=False, zeroweight=1.0)

    #base_config = dict(K=10, sigma2a=100, sigma2r=100, sigma2b=100, link_func='softplus', intercept_term=True, 
    #                    scale_context=False, normalize_context=False, downzero=True, use_obscov=True, zeroweight=1.0)
    
    comparison = 'scale_context'
    
    if comparison == 'K':
        legend_names = ['K = 5', 'K = 10', 'K = 20']
        configs = [base_config.copy() for i in xrange(0, len(legend_names))] 
        configs[0]['K'] = 5
        configs[1]['K'] = 10
        configs[2]['K'] = 20
    
    elif comparison == 'link_func':
        legend_names = ['exp', 'softplus']
        configs = [base_config.copy() for i in xrange(0, len(legend_names))] 
        configs[0]['K'] = 5
        configs[0]['link_func'] = 'exp'
        configs[1]['link_func'] = 'softplus'

    elif comparison == 'intercept_term':
        legend_names = ['w/o intercept term', 'w/ intercept term']
        configs = [base_config.copy() for i in xrange(0, len(legend_names))] 
        configs[0]['intercept_term'] = False
        configs[1]['intercept_term'] = True

    elif comparison == 'scale_context':
        legend_names = ['no measure', 'scale context by species/column', 'scale context by checklist/row']
        configs = [base_config.copy() for i in xrange(0, len(legend_names))] 
        configs[0]['scale_context'] = False
        configs[0]['normalize_context'] = False

        configs[1]['scale_context'] = True
        configs[1]['normalize_context'] = False

        configs[2]['scale_context'] = False
        configs[2]['normalize_context'] = True
    

    elif comparison == 'downzero':
        legend_names = ['weight = 1.0', 'weight = 0.5', 'weight = 0.1', 'weight = 0.0', 'bias only', 'observation']
        configs = [base_config.copy() for i in xrange(0, len(legend_names))] 
        configs[0]['downzero'] = False
        configs[0]['use_obscov'] = False
        configs[0]['zeroweight'] = 1.0

        configs[1]['downzero'] = False
        configs[1]['use_obscov'] = False
        configs[1]['zeroweight'] = 0.5

        configs[2]['downzero'] = False
        configs[2]['use_obscov'] = False
        configs[2]['zeroweight'] = 0.1

        configs[3]['downzero'] = False
        configs[3]['use_obscov'] = False
        configs[3]['zeroweight'] = 0.0

        configs[4]['downzero'] = True
        configs[4]['use_obscov'] = False
        configs[4]['zeroweight'] = 1.0


        configs[5]['downzero'] = True
        configs[5]['use_obscov'] = True
        configs[5]['zeroweight'] = 1.0



    mean = np.zeros((len(configs), 2))
    std = np.zeros((len(configs), 2))

    for ival in xrange(0, len(configs)):

        model_config = configs[ival] 

        res = retrieve_result(data_dir, model_config)
        mean[ival, 0] = np.mean(res['pos_llh'])
        mean[ival, 1] = np.mean(res['llh'])

        std[ival, 0] = np.std(res['pos_llh'])
        std[ival, 1] = np.std(res['llh'])

    mean = - mean

    bar_plot(mean, std, legend_names)


