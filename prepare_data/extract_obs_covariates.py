'''
This code take a subset of ebird data by year, day range, and location 
Created on Aug 26, 2016

@author: liuli

'''

import pandas as pd
import numpy as np

if __name__ == '__main__':
    
    folder = '../data/subset_pa/'

    print 'Reading in data ...'
    
    ebird_data = pd.read_csv(folder + 'obs_subset.csv')
  
    covariates = ebird_data.loc[:, ['EFFORT_HRS', 'EFFORT_DISTANCE_KM', 'EFFORT_AREA_HA', 'NUMBER_OBSERVERS']].as_matrix()


    print 'Calculating covariates for count types ...'
    type0 = ['P21']
    type1 = ['P22', 'P34']
    type2 = ['P23', 'P35']
    type3 = ['P20']
    type4 = ['P48']

    count_type = ebird_data['COUNT_TYPE']
    vec0 = np.array(count_type.apply(lambda x: x in type0), dtype=np.int8) 
    vec1 = np.array(count_type.apply(lambda x: x in type1), dtype=np.int8) 
    vec2 = np.array(count_type.apply(lambda x: x in type2), dtype=np.int8) 
    vec3 = np.array(count_type.apply(lambda x: x in type3), dtype=np.int8) 
    vec4 = np.array(count_type.apply(lambda x: x in type4), dtype=np.int8) 
        
    covar_type = np.vstack((vec0, vec1, vec2, vec3, vec4)).T

    print 'Calculating covariates for observation time ...'

    obs_time = ebird_data['TIME'].as_matrix()
    covar_time = np.vstack((np.logical_and(0 <= obs_time, obs_time < 7), 
                                 np.logical_and(7 <=  obs_time, obs_time < 12), 
                                 np.logical_and(12 <= obs_time, obs_time < 18), 
                                 np.logical_and(18 <= obs_time, obs_time < 24))).T 
       
    covariates = np.concatenate((covariates, covar_type, covar_time), axis=1)

    def fill_nan(x):
        nanflag = np.isnan(x)
        if np.sum(nanflag) > 0:
            fillval = np.mean(x[np.logical_not(nanflag)])
            x[nanflag] = fillval
            print "fill " + str(np.sum(nanflag)) + " nan valules " + str(fillval) + " with the mean of " + str(len(nanflag) - np.sum(nanflag)) + " values."

        return 0
    
    print str(np.sum(np.isnan(covariates))) + ' NaN values in covariates before fill'
    np.apply_along_axis(fill_nan, 0, covariates)
    print str(np.sum(np.isnan(covariates))) + ' NaN values in covariates after fill'
    

    print 'Saving covariates to file'

    np.save(folder + 'obs_covariates.npy', covariates)

    
