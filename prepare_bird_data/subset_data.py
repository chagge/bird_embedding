'''
This code take a subset of ebird data by year, day range, and species
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
    # state = PA
    # individual checklist

    # selection conditions for columns
    # localtion (LONG LAT)
    # travel distance
    # duration

    year = 2014 

    data_path = '/nfs/stak/students/l/liuli/liping/ebird/ebird_2014/' + str(year) + '/checklists.csv' 
    print 'Data path is ' + data_path

    df = pandas.read_csv(data_path, sep=',', header='infer', na_values=['X', '?'], na_filter=True, keep_default_na=True)


    flag = (108 <= df['DAY']) & (df['DAY'] < 210) 
    
    df = df.loc(flag, 'LAT', 'LONG')

    obs.to_csv('../data/obs_subset' + str(year) + '_range180-210' + '.csv')

