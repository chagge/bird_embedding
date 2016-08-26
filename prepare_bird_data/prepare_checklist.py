'''
This code reads in ebird data and aggregate bird observations by 1degree lat x 1degree long x 1 day stixels
for all bird species
Created on Aug 16, 2016

@author: liuli

'''

import pandas
import stixel_define as sd
import numpy
import sys

if __name__ == '__main__':
    
    print 'Reading in data ...'

    if len(sys.argv) < 2:
        error('Not enough arguments')

    year = sys.argv[1]

    data_path = '/nfs/stak/students/l/liuli/liping/ebird/ebird_2014/' + str(year) + '/checklists.csv' 
    print 'Data path is ' + data_path

    df = pandas.read_csv(data_path, sep=',', header='infer', na_values=['X', '?'], na_filter=True, keep_default_na=True)

    
    print 'Filling NaN values as 1 ...'
    # replace missing values as 1
    df = df.fillna(1) 

    print 'Calculating indices ...'
    good_flag, index = sd.get_stixel_index(df['YEAR'], df['DAY'], df['LATITUDE'], df['LONGITUDE'])
    df = df.loc[good_flag, 'Zenaida_macroura':]
    print 'Get ' + str(df.shape[0]) + ' checklists.'

    print 'Aggregating data ...'
    df.insert(0, 'INDEX', index)
    obs = df.groupby('INDEX').aggregate(numpy.sum)

    print 'Writing data to a CSV file...'
    obs.to_csv('../data/aggregated_obs_' + str(year) + '.csv')

