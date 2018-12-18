"""
This script compares HVAE, VAE, and NF models for a 
Gaussian model with Gaussian prior
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.stats
import time
import pickle
import os

from conf import params
from experiment_classes import HVAE, NF, VB

seed = 12345
np.random.seed(seed)
tf.set_random_seed(seed)

def run_test(d):
    """ 
    Run the gaussian test with dimension d 
    """

    ######### Problem Specification

    # Data generation parameters
    prior_mu_z = np.zeros(d, dtype=np.float32)    # Prior mean
    prior_sigma_z = np.eye(d, dtype=np.float32)   # Prior covariance matrix

    # True model parameters
    num_range = np.arange(-(d-1)/2, (d+1)/2, dtype=np.float32)

    t_delta =  num_range / 5 

    if d == 1:
        t_sigma = np.ones(1)
    else: 
        # Allow sigma to range from 0.1 to 1
        t_sigma = 36/(10*(d-1)**2) * num_range**2 + 0.1 

    ######### Variable Initialization

    # Initial model parameters - same across all methods
    init_delta = prior_mu_z.copy()
    init_log_sigma = 3 * np.ones(d)

    # Initial HVAE variational parameters
    init_T = 5.
    init_eps = 0.005 * np.ones(d)
    max_eps = params['max_eps'] * np.ones(d)
    init_logit_eps = np.log(init_eps/(max_eps - init_eps))
    init_log_T_0 = np.log(init_T - 1)

    # Initial NF variational parameters
    init_u_pre_reparam = scipy.stats.truncnorm.rvs(-2, 2, scale=0.1, size=d)
    init_w = scipy.stats.truncnorm.rvs(-2, 2, scale=0.1, size=d)
    init_b = 0.1

    # Initial VAE parameters
    init_mu_z = prior_mu_z.copy()
    init_log_sigma_z = np.ones(d)

    ######### Set up models

    HVAE_model_1 = HVAE(
        ['delta', 'log_sigma', 'logit_eps', 'log_T_0'],
        [init_delta, init_log_sigma, init_logit_eps, init_log_T_0], 
        'HVAE_1', d, params['HVAE_K_1'])
    HVAE_model_2 = HVAE(
        ['delta', 'log_sigma', 'logit_eps', 'log_T_0'],
        [init_delta, init_log_sigma, init_logit_eps, init_log_T_0], 
        'HVAE_2', d, params['HVAE_K_2'])

    HVAE_model_notemp_1 = HVAE(
        ['delta', 'log_sigma', 'logit_eps'],
        [init_delta, init_log_sigma, init_logit_eps], 
        'HVAE_notemp_1', d, params['HVAE_K_1'])
    HVAE_model_notemp_2 = HVAE(
        ['delta', 'log_sigma', 'logit_eps'], 
        [init_delta, init_log_sigma, init_logit_eps],
        'HVAE_notemp_2', d, params['HVAE_K_2'])

    NF_model_1 = NF(
        ['delta', 'log_sigma', 'u_pre_reparam', 'w', 'b'],
        [init_delta, init_log_sigma, init_u_pre_reparam, init_w, init_b],
        'NF_1', d, params['NF_K_1'])
    NF_model_2 = NF(
        ['delta', 'log_sigma', 'u_pre_reparam', 'w', 'b'],
        [init_delta, init_log_sigma, init_u_pre_reparam, init_w, init_b],
        'NF_2', d, params['NF_K_2'])

    VB_model = VB(['delta', 'log_sigma', 'mu_z', 'log_sigma_z'], 
        [init_delta, init_log_sigma, init_mu_z, init_log_sigma_z], 'VB', d)

    model_list = [HVAE_model_1, HVAE_model_2, HVAE_model_notemp_1, 
        HVAE_model_notemp_2, NF_model_1, NF_model_2, VB_model]
    
    ######### Generate Training Data & Save - One for each test

    train_data_list = []

    for i in range(params['n_tests']):
        z = np.random.multivariate_normal(prior_mu_z, prior_sigma_z)
        x = np.random.multivariate_normal(z + t_delta, np.diag(t_sigma**2), 
            size=params['n_data'])
        train_data_list.append(x)

    # Folder should have already been created in the initializations
    data_path = os.path.join('save', str(d), 'train_data.p')
    pickle.dump(train_data_list, open(data_path, 'wb')) 

    ######### Train models

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Store the final parameter values for all test runs in this dictionary
    final_params = {}

    for m in model_list:

        final_values = []

        for i in range(params['n_tests']):
            (delta, sigma) = m.train(sess, train_data_list[i], i)
            final_values.append((delta, sigma))

        final_params[m.model_name] = final_values.copy()

    ######### Test models using difference between parameters

    param_diffs = {}

    for m in model_list:

        diffs = []

        for i in range(params['n_tests']):
            delta = final_params[m.model_name][i][0]
            sigma = final_params[m.model_name][i][1]

            delta_diff = np.sum((delta - t_delta)**2)
            sigma_diff = np.sum((sigma - t_sigma)**2)

            diffs.append((delta_diff, sigma_diff))

        param_diffs[m.model_name] = diffs.copy()

    # Save parameter differences in a pickle file
    diff_path = os.path.join('save', str(d), 'all_diffs.p')
    pickle.dump(param_diffs, open(diff_path, 'wb'))

def main():
    """ Run the gaussian test across dimensions """
    dims = params['dims']

    for d in dims:
        print('**** Running test for d={0:d} ****'.format(d))
        run_test(d)

if __name__ == "__main__":
    main()
