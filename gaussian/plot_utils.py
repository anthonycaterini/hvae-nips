import pickle
import pdb
import numpy as np
import matplotlib.pyplot as plt
import os

from conf import params

def parse_pickle_save(method_names):
    """ 
    Retrive information from pickle files for the given methods

    Args:
        method_names: List of strings of names

    Returns:
        avg_diff: Average of the squared two-norm difference between
            the estimated value for theta and the true value
     """
    dims = params['dims']

    # These are all means/sds of differences in the parameters
    mean_diffs = np.zeros((len(method_names), len(dims)))

    # Loop through all dimensions and methods, calculating summary statistics
    for i in range(len(dims)):
        diffs_path = os.path.join('save', str(dims[i]), 'all_diffs.p')
        all_diffs = pickle.load(open(diffs_path, 'rb'))

        for j in range(len(method_names)):

            list_for_method = all_diffs[method_names[j]]

            delta_mean = np.nanmean([x[0] for x in list_for_method])
            sigma_mean = np.nanmean([x[1] for x in list_for_method])

            mean_diffs[j,i] = (delta_mean + sigma_mean)

    return mean_diffs

def plot_avg_diff(colours, markers, fig_idx, fig_size, plotnames,
    means, plot_title):
    """ 
    Plot comparison between average difference in parameter estimates
    across methods

    Args:
        colours: Line colours
        markers: Line markers
        fig_idx: Figure number
        fig_size: Figure size
        plotnames: Method names for legend
        means: Numpy array containing the average differences
        plot_title: Title of the plot

    There is no return value, but a matplotlib plot is produced
    with the block=False option to allow the script to continue 
    running.
    """
    dims = params['dims']

    plt.close()

    plt.figure(fig_idx, figsize=fig_size)

    # Some aesthetic considerations
    ax = plt.subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # Means and stds are both numpy arrays of size 
    #   len(plotnames) x len(dims) 
    for j in range(len(plotnames)):
        c = colours[j]
        style = c + markers[j] + '-'
        plt.plot(dims, means[j,:], style, label=plotnames[j])

    ax.set_yscale('log')

    plt.legend(loc='best')
    plt.title(plot_title, fontsize=16)
    plt.xlabel('Dimensionality')
    plt.show(block=False)