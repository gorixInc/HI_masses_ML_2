# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 22:30:02 2021

@author: Gordei
"""

import numpy as np
from random import shuffle
from math import floor
import pandas as pd
from sklearn.preprocessing import scale

def create_tvt_sets(df_path, p_train, p_valid):
    arecibo_df = pd.read_csv(df_path)
    x_values = arecibo_df.drop(['MHI', 'Name'], axis=1).values.astype('float32')
    x_values = scale(x_values)
    y_values = arecibo_df['MHI'].values.astype('float32')
    y_values = np.reshape(y_values, (-1, 1))
    
    n_data = len(x_values)
    n_train = floor(n_data*p_train)
    n_valid = floor(n_data*p_valid)
    
    c = list(zip(x_values, y_values))
    shuffle(c)
    x_shuf, y_shuf = zip(*c)
    
    train_set = (x_shuf[0:n_train], y_shuf[0:n_train])
    valid_set = (x_shuf[n_train:n_train + n_valid], 
                 y_shuf[n_train:n_train + n_valid])
    test_set = (x_shuf[n_train + n_valid:n_data], 
                y_shuf[n_train + n_valid:n_data])
    
    return train_set, valid_set, test_set


def select_with_MHI_mean(nancay_path, calculate_distance, target_mean=8.07,
                            sample_size = 100, max_mean_dist = 2,
                            acc_dev = 0.2):
    nancay_df = pd.read_csv(nancay_path)
    
    nancay_values = nancay_df.drop('Name', axis = 1).values        
    
    best_value_index = 0 
    last_dist = 0
    sample = []
    to_ignore = []
    for i in range(sample_size):
        smallest_suit_sep = 999
        for j in range(len(nancay_values)):
            if j in to_ignore:
                continue
            value = nancay_values[j]
            distance = calculate_distance(target_mean, value[10])
            if(abs(distance) > max_mean_dist):
                continue
            sep_from_needed_value = distance + last_dist
            if(abs(sep_from_needed_value) < acc_dev and
               sep_from_needed_value < smallest_suit_sep):       
                smallest_suit_sep = sep_from_needed_value
                best_value_index = j
                break
        sample.append(nancay_df.values[best_value_index])
        last_dist = calculate_distance(target_mean, 
                                       nancay_values[best_value_index][10])
        
        to_ignore.append(best_value_index)
    sample = np.array(sample)
    sample_df = pd.DataFrame(sample, columns = nancay_df.columns)
    return sample_df