
'''
StixelDef class provide the method of defining stixels (time x space) 
Author: Liping Liu
Last update: 08/17/2016
'''

import numpy
import pandas 
import math 

start_year = 2005
end_year = 2015 
               
start_long = -125
end_long = -65 

start_lat = 15 
end_lat = 50

leap_year = map(lambda x : int(x % 4 == 0), range(start_year, start_year + 20))
leap_counts = numpy.insert(numpy.cumsum(leap_year), 0, 0)

num_day = (end_year - start_year) * 365 + leap_counts[end_year - start_year] 
num_lat = end_lat - start_lat
num_long = end_long - start_long

def get_stixel_index(year, day, latitude, longitude):
    
    day_ind = (year - start_year) * 365 + leap_counts[(year - start_year).tolist()] + day - 1
    lat_ind = latitude.apply(lambda x: int(math.floor(x - start_lat)))
    long_ind = longitude.apply(lambda x: int(math.floor(x - start_long)))
    
    good_flag = (day_ind >= 0) & (lat_ind >= 0) &  (long_ind >= 0)

    print '' + str(sum(~ good_flag)) + ' records are out of boundary.'
    print pandas.concat([day[~ good_flag], latitude[~ good_flag], longitude[~ good_flag]], axis=1)

    #def value_check(x): 
    #    if(x < 0): 
    #        raise IndexError('Negative value in indices')

    #day_ind.apply(value_check)
    #lat_ind.apply(value_check)
    #long_ind.apply(value_check)

    index = day_ind * (num_lat * num_long) + long_ind * num_lat + lat_ind
    index = index[good_flag]

    return good_flag, index



