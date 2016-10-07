''' This code plot the vectors of birds '''

import sys
import numpy as np

import csv
from matplotlib import pyplot as plt
sys.path.append('../lib/tsne')
from tsne import tsne
from matplotlib.backends.backend_pdf import PdfPages

def plot_bird(alpha):

    # read in bird names
    with open('../data/bird_names.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        bird_names = spamreader.next()

    # perform t-SNE embedding
    vis_data = tsne(alpha)

    # get an indicator for whether ploting a data point or not
    compx = vis_data[:, 0]
    compy = vis_data[:, 1]
    d = np.sqrt(compx * compx + compy * compy)
    threshold = 0.0
    flag = d > threshold 
    
    # get a subset of data
    selected = [s for (s, b) in zip(bird_names, flag) if b]  
    compx = vis_data[flag, 0]
    compy = vis_data[flag, 1]
    
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

    pp = PdfPages('plot.pdf')
    plt.savefig(pp, format='pdf')
    
    # print some strange birds 
    #print [bird_names[i] for i in np.where(vis_data[:, 0] > 0.1)[0]]

if __name__ == "__main__":

    fold = 5 

    model_config = dict(K=10, sigma2a=100, sigma2r=100, sigma2b=100, link_func='exp', intercept_term=False, scale_context=True, downzero=True, use_obscov=True)
    filename = 'result/' + config_to_filename(model_config, fold=5)
    pkl_file = open(filename, 'rb') 
    model = output['model']

    print model['alpha']



