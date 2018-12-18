# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import datetime as dt
import os
import pdb
import pickle
import smtplib
from email.mime.text import MIMEText

import numpy as np
import tensorflow as tf

from data import MNIST
from models import VAE, HVAE
from utils import set_gpus
from training import train, validate, evaluate
from adamax import AdamaxOptimizer
from plotting import plot_training_curve, plot_digit_samples


parser = argparse.ArgumentParser(
    description='Hamiltonian Variational Auto-Encoder')

# Model settings
parser.add_argument('-m', '--model', type=str, choices=['cnn', 'hvae'], 
    metavar='MODEL', help='Model to use', required=True)
parser.add_argument('-n_z', '--z_dim', type=int, default=64, metavar='ZDIM',
    help='Latent space dimension (default: 64)')

parser.add_argument('-s', '--seed', type=int, metavar='RANDOM_SEED', 
    help='Random seed. If not provided, resort to random')

# Required arguments for HVAE
parser.add_argument('-K', type=int, metavar='FLOW_STEPS', 
    help='Number of flow steps')
parser.add_argument('-T', '--temp_method', type=str, 
    choices=['none', 'fixed', 'free'], metavar='TEMP_METHOD', 
    help=r'Tempering Method for HVAE (one of "none", "fixed", or "free")')

# Optional arguments for HVAE
parser.add_argument('-ve', '--vary_eps', type=str, choices=['true', 'false'],
    metavar='VARY_LF_EPSILON', default='false', 
    help='Allow the leapfrog epsilon to vary across flow steps')

# Initialization arguments for HVAE
parser.add_argument('--max_lf', type=float, default=0.5, 
    metavar='MAX_LF_STEP_SIZE', help='Maximum step size for leapfrog')
parser.add_argument('--init_lf', type=float, default=0.2, 
    metavar='INIT_LF_STEP_SIZE', help='Initial step size for leapfrog')
parser.add_argument('--init_alpha', type=float, default=0.9, 
    metavar='INIT_ALPHA', help='Initial alpha values for free tempering')
parser.add_argument('--init_T_0', type=float, default=1.5, metavar='INIT_T_0',
    help='Initial value for T_0 for fixed tempering')

# Optimization settings - most need not be changed from defaults
parser.add_argument('-e', '--epochs', type=int, default=2000, 
    metavar='EPOCHS', help='Number of training epochs (default: 2000)')
parser.add_argument('-es', '--early_stopping_epochs', type=int, default=100,
    metavar='EARLY_STOPPING', 
    help='Number of early stopping epochs (default: 100)')
parser.add_argument('--n_val', type=int, default=10000, 
    metavar='VALIDATION_SIZE', 
    help='Number of datapoints in validation set (default: 10000)')

parser.add_argument('-bs', '--n_batch', type=int, default=100, 
    metavar='BATCH_SIZE', help='Batch size for training (default: 100)')
parser.add_argument('-lr', '--learn_rate', type=float, default=0.0005, 
    metavar='LEARNING_RATE', help='learning rate (default 0.0005)')
parser.add_argument('-eps', '--adamax_eps', type=float, default=1e-7,
    metavar='ADAMAX_EPS', help='Adamax epsilon (default 10^(-7))')

# Evaluation settings. We use batches of points for the test to save memory
parser.add_argument('-tbs', '--n_batch_test', type=int, default=10,
    metavar='TEST_BATCH_SIZE', help='Batch size for test (default: 10)')
parser.add_argument('-vbs', '--n_batch_val', type=int, default=10000,
    metavar='VAL_BATCH_SIZE', 
    help='Batch size for validation (default: 10000)')
parser.add_argument('-n_IS', type=int, default=1000, metavar='N_IMP_SAMPLES',
    help='Number of importance samples for test (default:1000)')
parser.add_argument('--n_nll_runs', type=int, default=3, 
    metavar='N_NLL_RUNS', help='Number of runs for NLL calc (default: 3)')

parser.add_argument('--n_gpu', type=int, default=1, metavar='N_GPUS', 
    help='Number of GPUs to use (default: 1)')

# Logging options
parser.add_argument('-pi', '--print_interval', type=int, default=50,
    metavar='PRINT_INTERVAL', help='Print every <print_interval> batches')
parser.add_argument('-c_dir', '--checkpoint_dir', type=str, 
    default='checkpoints', metavar='CHECKPOINT_DIR', 
    help='Model checkpoint directory')
parser.add_argument('-lf', '--logfile', type=str, default='log.txt', 
    help='Log file for experiments.')
parser.add_argument('--pickle_dir', type=str, default='pickle', 
    help='Pickle save directory')

parser.add_argument('--sender', type=str, help=r"Sender's email address")
parser.add_argument('--receiver', type=str, help='Where to send email')

# Plotting options
parser.add_argument('--plot_dir', type=str, default='plots',
    help='Plot directory')
parser.add_argument('--n_gen_samples', type=int, default=64,
    help='Number of samples from generative model to plot')
parser.add_argument('--sample_grid_h', type=int, default=8,
    help='Height of grid of samples from generative model')
parser.add_argument('--sample_grid_w', type=int, default=8,
    help='Width of grid of samples from generative model')

args = parser.parse_args()

if args.seed is None:
    args.seed = np.random.randint(1, 100000)
np.random.seed(args.seed)
tf.set_random_seed(args.seed)


def run(args):

    print('\nSettings: \n', args, '\n')

    args.model_signature = str(dt.datetime.now())[0:19].replace(' ', '_')
    args.model_signature = args.model_signature.replace(':', '_')

    ########## Find GPUs 
    (gpu_config, n_gpu_used) = set_gpus(args.n_gpu)

    ########## Data, model, and optimizer setup
    mnist = MNIST(args)
    
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])

    if args.model == 'hvae':
        if not args.K:
            raise ValueError('Must set number of flow steps when using HVAE')
        elif not args.temp_method:
            raise ValueError('Must set tempering method when using HVAE')
        model = HVAE(args, mnist.avg_logit)
    elif args.model == 'cnn':
        model = VAE(args, mnist.avg_logit)
    else:
        raise ValueError('Invalid model choice')

    elbo = model.get_elbo(x, args)
    nll = model.get_nll(x, args)

    optimizer = AdamaxOptimizer(learning_rate=args.learn_rate, 
        eps=args.adamax_eps)
    opt_step = optimizer.minimize(-elbo)

    ########## Tensorflow and saver setup
    sess = tf.Session(config=gpu_config)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    savepath = os.path.join(args.checkpoint_dir, args.model_signature, 
        'model.ckpt')

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    ########## Test that GPU memory is sufficient
    if n_gpu_used > 0:
        try:
            x_test = mnist.next_test_batch()
            (t_e, t_n) = sess.run((elbo, nll), {x: x_test})
            mnist.batch_idx_test = 0 # Reset batch counter if it works
        except:
            raise MemoryError(
                """
                Likely insufficient GPU memory
                Reduce test batch by lowering the -tbs parameter
                """
                )

    ########## Training Loop

    train_elbo_hist = []
    val_elbo_hist = []

    # For early stopping
    best_elbo = -np.inf
    es_epochs = 0
    epoch = 0

    train_times = []

    for epoch in range(1, args.epochs+1):

        t0 = time.time()
        train_elbo = train(epoch, mnist, opt_step, elbo, x, args, sess)
        train_elbo_hist.append(train_elbo)
        train_times.append(time.time()-t0)
        print('One epoch took {:.2f} seconds'.format(time.time()-t0))

        val_elbo = validate(mnist, elbo, x, sess)
        val_elbo_hist.append(val_elbo)

        if val_elbo > best_elbo:

            # Save the model that currently generalizes best
            es_epochs = 0
            best_elbo = val_elbo
            saver.save(sess, savepath)
            best_model_epoch = epoch

        elif args.early_stopping_epochs > 0:

            es_epochs += 1

            if es_epochs >= args.early_stopping_epochs:
                print('***** STOPPING EARLY ON EPOCH {} of {} *****'.format(
                    epoch, args.epochs))
                break

        print('--> Early stopping: {}/{} (Best ELBO: {:.4f})'.format(
            es_epochs, args.early_stopping_epochs, best_elbo))
        print('\t Current val ELBO: {:.4f}\n'.format(val_elbo))

        if np.isnan(val_elbo):
            raise ValueError('NaN encountered!')

    train_times = np.array(train_times)
    mean_time = np.mean(train_times)
    std_time = np.std(train_times)
    print('Average train time per epoch: {:.2f} +/- {:.2f}'.format(
        mean_time, std_time))

    ########## Evaluation

    # Restore the best-performing model
    saver.restore(sess, savepath)

    test_elbos = np.zeros(args.n_nll_runs)
    test_nlls = np.zeros(args.n_nll_runs)

    for i in range(args.n_nll_runs):

        print('\n---- Test run {} of {} ----\n'.format(i+1, args.n_nll_runs))
        (test_elbos[i], test_nlls[i]) = evaluate(mnist, elbo, nll, x, 
                                                args, sess)

    mean_elbo = np.mean(test_elbos)
    std_elbo = np.std(test_elbos)

    mean_nll = np.mean(test_nlls)
    std_nll = np.std(test_nlls)

    print('\nTest ELBO: {:.2f} +/- {:.2f}'.format(mean_elbo, std_elbo))
    print('Test NLL: {:.2f} +/- {:.2f}'.format(mean_nll, std_nll))

    ########## Logging, Saving, and Plotting

    with open(args.logfile, 'a') as ff:
        print('----------------- Test ID {} -----------------'.
            format(args.model_signature), file=ff)
        print(args, file=ff)
        print('Stopped after {} epochs'.format(epoch), file=ff)
        print('Best model from epoch {}'.format(best_model_epoch), file=ff)
        print('Average train time per epoch: {:.2f} +/- {:.2f}'.format(
            mean_time, std_time), file=ff)

        print('FINAL VALIDATION ELBO: {:.2f}'.format(val_elbo_hist[-1]), 
            file=ff)
        print('Test ELBO: {:.2f} +/- {:.2f}'.format(mean_elbo, std_elbo), 
            file=ff)
        print('Test NLL: {:.2f} +/- {:.2f}\n'.format(mean_nll, std_nll),
            file=ff)

    if not os.path.exists(args.pickle_dir):
        os.makedirs(args.pickle_dir)

    train_dict = {'train_elbo': train_elbo_hist, 'val_elbo': val_elbo_hist, 
        'args': args}
    pickle.dump(train_dict, open(os.path.join(args.pickle_dir, 
        args.model_signature + '.p'), 'wb'))

    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)

    tf_gen_samples = model.get_samples(args)
    np_gen_samples = sess.run(tf_gen_samples)
    plot_digit_samples(np_gen_samples, args)

    plot_training_curve(train_elbo_hist, val_elbo_hist, args)

    ########## Email notification upon test completion

    try:

        msg_text = """Test completed for ID {0}.

        Parameters: {1}

        Test ELBO: {2:.2f} +/- {3:.2f}
        Test NLL: {4:.2f} +/- {5:.2f} """.format(
            args.model_signature, args, mean_elbo, std_elbo, mean_nll, std_nll)

        msg = MIMEText(msg_text)
        msg['Subject'] = 'Test ID {0} Complete'.format(args.model_signature)
        msg['To'] = args.receiver
        msg['From'] = args.sender

        s = smtplib.SMTP('localhost')
        s.sendmail(args.sender, [args.receiver], msg.as_string())
        s.quit()

    except:

        print('Unable to send email from sender {0} to receiver {1}'.
            format(args.sender, args.receiver))


if __name__ == "__main__":

    run(args)
