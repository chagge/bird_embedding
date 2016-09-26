''' This file implements the methods of calculating context''' 

import numpy as np

def pos95percent(x):
    if (np.sum(x > 0) > 0):
        return np.percentile(x[x > 0], q=95)
    else:
        return 0



def counts_to_context(counts):


    context = counts.copy()
    s = np.apply_along_axis(pos95percent, axis=0, arr=context)
    s[s <= 0] = 1
    context = context / s

    return context

def counts_pos95percent(counts):

    s = np.apply_along_axis(pos95percent, axis=0, arr=counts)
    return s
