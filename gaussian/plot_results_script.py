""" 
This script plots the average and standard deviation of the norm of the 
difference between the true and estimated parameter values.

There are separate plots for the difference in delta, the offset parameter,
and sigma, the standard deviation of the generating process.

There are two sets of plots: one comparing the results across several methods
and the other comparing HVAE with and without tempering. 
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from conf import params
from plot_utils import parse_pickle_save, plot_avg_diff

# Plot specifications
colours = ['b', 'g', 'r', 'c', 'k']
markers = ['d', '+', 's', '*', 'x']
fig_size = (6, 4.5)

# Method names for the tests and plot legends
#   'all' corresponds to the test of all methods
#   'temp' corresponding to the tempering test
methods_all = ['HVAE_1', 'HVAE_2', 'NF_1', 'NF_2', 'VB']
methods_temp = ['HVAE_1', 'HVAE_2', 'HVAE_notemp_1', 'HVAE_notemp_2']
plotnames_all = [r'HVAE, $K={0}$'.format(params['HVAE_K_1']),
                 r'HVAE, $K={0}$'.format(params['HVAE_K_2']),
                 r'NF, $K={0}$'.format(params['NF_K_1']),
                 r'NF, $K={0}$'.format(params['NF_K_2']),
                 'VB']
plotnames_temp = [r'HVAE, $K={0}$'.format(params['HVAE_K_1']),
                  r'HVAE, $K={0}$'.format(params['HVAE_K_2']),
                  r'HVAE, $K={0}$, $\beta_0=1$'.format(params['HVAE_K_1']),
                  r'HVAE, $K={0}$, $\beta_0=1$'.format(params['HVAE_K_2'])]

# Extract the parameter difference information from the pickle files
diffs_all = parse_pickle_save(methods_all)
diffs_temp = parse_pickle_save(methods_temp)

# Make the plots testing all methods with error bars
if not os.path.exists('plots'):
    os.makedirs('plots')

plot_avg_diff(colours, markers, 1, fig_size, plotnames_all, diffs_all,
    r'Error in estimate of $\theta$')
plt.savefig('plots/all_methods.png')

# Make the plot comparing tempering vs. no tempering
# \beta_0 = 1 corresponds to no tempering
plot_avg_diff(colours, markers, 1, fig_size, plotnames_temp, diffs_temp,
    r'Error in estimate of $\theta$')
plt.savefig('plots/temp_vs_no_temp.png')
