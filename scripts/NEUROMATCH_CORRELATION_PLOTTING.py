# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:55:26 2024

@author: athan
"""

import os
import pickle
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from scipy import stats

# Initialize lists to store correlations and p-values
correlations = []
p_values = []

# Path to the directory containing pickle files
path = ''

# Loop through each pickle file in the directory
for filename in os.listdir(path):
    if filename.endswith('.pickle'):
        # Load the pickle file
        with open(os.path.join(path, filename), 'rb') as f:
            data = pickle.load(f)

        # Assuming 'stimulus_table' and 'rewards' are keys in the dictionary loaded from the pickle file
        stimulus_table = data['stimulus_table']
        rewards = data['rewards']
        
        CHANGE_OCCURED = stimulus_table[stimulus_table['is_change']]

        pairs = pd.DataFrame(list(itertools.product(CHANGE_OCCURED['start_time'], rewards['timestamps'])),
                             columns=['start_time', 'timestamps'])

        pairs['diff'] = pairs['start_time'] - pairs['timestamps']

        pairs = pairs.query('abs(diff) <= 0.5')

        pairs = pairs.drop_duplicates()

        plt.scatter(pairs.index, pairs['diff'])
        plt.xlabel('Index')
        plt.ylabel('Time Difference')
        plt.show()
        
    
        corr, p_val = stats.spearmanr(pairs.index, pairs['diff'])
        print('Spearman correlation:', corr)


        # corr, p_val = stats.pearsonr(pairs.index, pairs['diff'])
        # print('Pearsons:', corr)
        
        # Store the correlation and p-value
        correlations.append(corr)
        p_values.append(p_val)
        
        # plt.bar(range(len(correlations)), correlations)
        # plt.xlabel('Subject')
        # plt.ylabel('Correlation')
        # plt.xticks(range(len(correlations)), [f'Subject {i+1}' for i in range(len(correlations))])
        # plt.show()

        


import os
import matplotlib.pyplot as plt
import scipy.stats as stats
import glob
import pickle

# Initialize lists to store correlations and p-values
correlations = []
p_values = []

path = ''

# Loop through each pickle file in the directory
for filename in os.listdir(path):
    if filename.endswith('.pickle'):
        # Load the pickle file
        with open(os.path.join(path, filename), 'rb') as f:
            data= pickle.load(f)

    # Replace negative values in 'speed' column with 0
    data['running']['speed'] =  data['running']['speed'].clip(lower=0)

    # Loop through rows in 'running' DataFrame
    for i, row in  data['running'].iterrows():
        # Calculate Spearman correlation
        correlation, p_value = stats.spearmanr( data['running']['speed'], data['running']['timestamps'])
        correlations.append(correlation)
        p_values.append(p_value)

        # Create scatter plot
        plt.scatter(row['timestamps'], row['speed'])
        plt.xlabel('Timestamps')
        plt.ylabel('Speed')
        plt.text(0.05, 0.9, 'Spearman correlation: %.3f' % correlation, transform=plt.gca().transAxes)
        plt.show()
