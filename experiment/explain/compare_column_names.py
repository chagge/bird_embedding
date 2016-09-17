#import itertools
import os
import csv

with open('../data/aggregated_obs_2014.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    names14 = spamreader.next()

base = '' 
for iyear in xrange(05, 15):
    with open('../data/aggregated_obs_2014.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        head = spamreader.next()

    if not base:
        base = head
    elif base != head:
        print 'different head in year ' + str(iyear) 

    print head[0 : 10]
        
