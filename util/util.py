import pandas
import csv
import numpy as np

def config_to_filename(model_config, fold):

    if 'cont_train' in model_config:
       filename = ('experiment' + '_ct' + str(int(model_config['cont_train']))
                                 + '_k' + str(int(model_config['K']))
                                 + '_lf' + model_config['link_func']  
                                 + '_sc' + str(int(model_config['scale_context'])) 
                                 + '_nc' + str(int(model_config['normalize_context'])) 
                                 + '_it' + str(int(model_config['intercept_term'])) 
                                 + '_dz'  + str(int(model_config['downzero'])) 
                                 + '_pl'  + str(int(model_config['zeroweight'])) 
                                 + '_uo' + str(int(model_config['use_obscov'])) 
                                 + '_sa' + str(int(model_config['sigma2a'])) 
                                 + '_sb' + str(int(model_config['sigma2b'])) 
                                 + '_f' + str(fold) + '.pkl')

    else:

        filename = ('experiment' + '_k' + str(int(model_config['K']))
                                 + '_lf' + model_config['link_func']  
                                 + '_sc' + str(int(model_config['scale_context'])) 
                                 + '_nc' + str(int(model_config['normalize_context'])) 
                                 + '_it' + str(int(model_config['intercept_term'])) 
                                 + '_dz'  + str(int(model_config['downzero'])) 
                                 + '_pl'  + str(int(model_config['zeroweight'])) 
                                 + '_uo' + str(int(model_config['use_obscov'])) 
                                 + '_sa' + str(int(model_config['sigma2a'])) 
                                 + '_sb' + str(int(model_config['sigma2b'])) 
                                 + '_f' + str(fold) + '.pkl')

 
    return filename


def get_species(data_dir, fold):

    taxonomy = pandas.read_csv(data_dir + 'taxonomy.csv', header=0) 
    bird_dict = dict(zip(taxonomy['SCI_NAME'], taxonomy['PRIMARY_COM_NAME']))
    
    with open(data_dir + 'abd_bird_names.csv', 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        sci_names = spamreader.next()
    
    fold_dir = data_dir + 'data_folds/' + str(fold) + '/'
    species_ind = np.loadtxt(fold_dir + 'nonzero_ind.csv', dtype=int)
    bird_names = [bird_dict[sci_names[ind]] for ind in species_ind]


    return bird_names


