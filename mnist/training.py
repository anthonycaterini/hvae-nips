import pdb
import time
import numpy as np

def train(epoch, mnist, opt_step, elbo, x, args, sess):
    """
    Train the model for one epoch by taking steps of the optimizer
    """
    n_batches = mnist.batches_in_train
    train_elbo = np.zeros(n_batches)

    for i in range(n_batches):

        train_batch = mnist.next_train_batch()

        _, np_elbo = sess.run((opt_step, elbo), {x: train_batch})

        train_elbo[i] = np_elbo

        if (i+1) % args.print_interval == 0:
            print('Epoch: {0:d} [{1:d}/{2:d}] ({3:.1f}%); \tELBO: {4:.6f}'.
                format(epoch, i+1, n_batches, 100*((i+1)/n_batches), np_elbo))

    avg_elbo = np.sum(train_elbo)/n_batches
    print('====> Epoch: {:3d} Average train ELBO: {:.4f}'.format(
        epoch, avg_elbo))

    return avg_elbo

def validate(mnist, elbo, x, sess):
    """
    Evaluate the model, just with ELBO, over the validation set
    """
    n_batches = mnist.batches_in_val
    np_elbo = 0

    for i in range(n_batches):

        val_batch = mnist.next_val_batch()
        np_elbo += sess.run(elbo, {x: val_batch})

    np_elbo /= n_batches

    return np_elbo

def evaluate(mnist, elbo, nll, x, args, sess):
    """
    Evaluate the model with ELBO and NLL over the test set
    """
    n_batches = mnist.batches_in_test
    sum_elbo = 0
    sum_nll = 0

    t_total = time.time()
    t0 = time.time()
    for i in range(n_batches):

        test_batch = mnist.next_test_batch()
        
        (batch_elbo, batch_nll) = sess.run((elbo, nll), {x: test_batch})
        sum_elbo += batch_elbo
        sum_nll += batch_nll

        if (i+1) % args.print_interval == 0:
            print(("Batch {0:d}/{1:d}; \tELBO: {2:.2f};"
                "\tNLL: {3:.2f};\ttime: {4:.2f}s").format(
                    i+1, n_batches, batch_elbo, batch_nll, time.time()-t0))
            t0 = time.time()

    test_elbo = sum_elbo / n_batches
    test_nll = sum_nll / n_batches
    print('====> Avg. ELBO: {0:.2f}, Avg. NLL: {1:.2f}, Time: {2:.2f}s'.
        format(test_elbo, test_nll, time.time()-t_total))

    return test_elbo, test_nll
