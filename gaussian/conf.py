""" Specify experiment parameters here """

params = {

    # PROBLEM SPECIFICATION
    'dims': [1, 2, 3, 5, 11, 25, 51, 101, 201, 301], # Dimensions to test
    'n_data': 10000,                                 # Number of data points

    # TEST HYPERPARAMETERS
    'n_tests': 10,      # Number of experiments to run

    # GLOBAL OPTIMIZATION PARAMETERS
    'n_iter': 30000,    # Number of optimization iterations
    'n_batch': 10,      # Number of points for ELBO estimate
    'rms_eta': 0.001,   # Stepsize for RMSProp
    'save_every': 10,   # Save parameter information every so often
    'print_every': 500, # Print less often than save

    # HVAE HYPERPARAMETERS
    'HVAE_K_1': 1,    # Number of leapfrog/cooling steps for HVAE flow 1
    'HVAE_K_2': 10,   # Number of leapfrog/cooling steps for HVAE flow 2
    'max_eps': 0.5,   # Maximum leapfrog step size per dimension 

    # NF HYPERPARAMETERS
    'NF_K_1': 1,    # Number of flow steps for NF flow 1
    'NF_K_2': 30,   # Number of flow steps for NF flow 2
    
}
