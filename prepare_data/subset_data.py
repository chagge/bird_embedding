'''
This code take a subset of ebird data by year, day range, and location 
Created on Aug 26, 2016

@author: liuli

'''

import pandas
import stixel_define as sd
import numpy
import sys

if __name__ == '__main__':
    
    print 'Reading in data ...'
    
    # selection conditions for rows
    # day range =  180 ~ 209
    # area range = lat: 39.716 ~ 42, long: -80.5 ~ -74.3
    # individual checklist
    
    # selection conditions for columns
    # shown in the variable "flag"
    
    for year in xrange(2014, 2015):
        data_path = '../data/ebird/' + str(year) + '/checklists.csv' 
        print 'Data path is ' + data_path
    
        df = pandas.read_csv(data_path, sep=',', header='infer', na_values=['X', '?'], na_filter=True, keep_default_na=True)
    
    
        # if only take obs from day 180 to day 209 
        #flag = (180 <= df['DAY']) & (df['DAY'] < 210) 

        # if only take obs from day 180 to day 209 in PA area 
        flag = ((180 <= df['DAY']) & (df['DAY'] < 210) 
            & ((39 + 43.0/60) < df['LATITUDE']) & (df['LATITUDE'] < 42)
            & (-(80 + 31.0/60) < df['LONGITUDE']) & (df['LONGITUDE'] < -74.3))
    
        # if only take out eastern wood pewee
        #df = df.loc[flag, ['LATITUDE','LONGITUDE','DAY','TIME', 'COUNT_TYPE', 'EFFORT_HRS', 'EFFORT_DISTANCE_KM', 
        #                 'EFFORT_AREA_HA', 'NUMBER_OBSERVERS', 'PRIMARY_CHECKLIST_FLAG', 'Contopus_virens']]
    
    
        # if to take all species 
        df = df.loc[flag, :]

        #df.to_csv('../data/obs_subset_y' + str(year) + '_d180-210_ewp' + '.csv')
        df.to_csv('../data/subset_pa_201407/obs_subset.csv')
        #df.to_csv('../data/subset_07/obs_subset.csv')



