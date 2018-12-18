import os
import numpy as np

import matplotlib
matplotlib.use('Agg') # To allow images to be made without showing them
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_training_curve(train_elbo, val_elbo, args):
    """
    Plot the results of training
    """

    plt.close()
    epochs = len(train_elbo)
    x_vals = np.arange(1, epochs+1)
    
    with plt.style.context('ggplot'):

        # Make x ticks integers
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.semilogy(x_vals, -train_elbo)
        plt.semilogy(x_vals, -val_elbo)

        plt.title('Training and Validation Negative ELBO Per Epoch')
        plt.legend(['Train', 'Validation'], loc='best')
        plt.xlabel('Epoch Number')
        plt.ylabel('Negative ELBO')

        plot_path = os.path.join(args.plot_dir, 
            args.model_signature + '_training.png')
        plt.savefig(plot_path)

    print('Saved training curve to {}'.format(plot_path))

def plot_digit_samples(samples, args):
    """
    Plot samples from the generative network in a grid
    """

    grid_h = args.sample_grid_h
    grid_w = args.sample_grid_w
    mnist_h = 28
    mnist_w = 28

    assert args.n_gen_samples == grid_h*grid_w

    # Turn the samples into one large image
    tiled_img = np.zeros((mnist_h*grid_h, mnist_w*grid_w))

    for idx, image in enumerate(samples):
        i = idx % grid_w
        j = idx // grid_w

        top = j * mnist_h
        bottom = (j+1) * mnist_h
        left = i * mnist_w
        right = (i+1) * mnist_w
        tiled_img[top:bottom, left:right] = image

    # Save the new image
    plt.close()
    plt.title('MNIST Visualization')
    plt.axis('off')
    plt.imshow(tiled_img, cmap='gray')

    img_path = os.path.join(args.plot_dir, 
        args.model_signature + '_samples.png')
    plt.savefig(img_path)

    print('Saved samples to {}'.format(img_path))
