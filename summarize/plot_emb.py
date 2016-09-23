''' This code plot the vectors of birds '''

import glob
import os
import numpy as np
import pandas as pd

import csv
from scipy.spatial import distance
from matplotlib import pyplot as plt
from adjustText import adjust_text
import itertools

def plot_bird(alpha):

    n_components = 2
    icx = 0
    icy = 1

    with open('../data/bird_names.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        bird_names = spamreader.next()
    
    d = np.sqrt(alpha[:, icx] * alpha[:, icx] + alpha[:, icy] * alpha[:, icy])
    threshold = 0.0
    flag = d > threshold 
    
    selected = [s for (s, b) in zip(bird_names, flag) if b]  
    compx = alpha[flag, icx]
    compy = alpha[flag, icy]
    
    plt.scatter(compx, compy)
    #for label, x, y in zip(selected, compx, compy):
    #    plt.annotate(label, xy = (x, y), xytext = (0, 0), textcoords = 'offset points', ha = 'right', va = 'bottom')
    
    texts = []
    for xt, yt, s in zip(compx, compy, selected):
        if xt < 0.01:
            texts.append(plt.text(xt, yt, s, size=8, ha = 'right', va = 'bottom'))
        else:
            texts.append(plt.text(xt, yt, s, size=8, ha = 'left', va = 'bottom'))
    
    plt.show()
    
    # print some strange birds 

    #print [bird_names[i] for i in np.where(alpha[:, 0] > 0.1)[0]]
    #print [bird_names[i] for i in np.where(alpha[:, 1] < -0.08)[0]]


