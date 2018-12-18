import tensorflow as tf 
import numpy as np
import pdb

class MNIST():
    """
    Class for dealing with MNIST, including batch sizes, validation data, and
    dynamic binarization of the batch (not used in validation)
    """

    def __init__(self, args):

        mnist = tf.contrib.learn.datasets.load_dataset('mnist')
        
        # Will do the separation of training and validation manually
        full_train = np.concatenate(
            (mnist.train.images, mnist.validation.images))
        np.random.shuffle(full_train)

        # Static binarization of validation set - easier for early stopping
        self.validation = np.random.binomial(
            1, full_train[0:args.n_val,:].copy()).astype(np.float32)

        self.train = full_train[args.n_val:,:].copy()
        self.test = mnist.test.images

        # Calculate average logit for initialization of final bias
        img_avg = np.mean(self.train, axis=1)
        self.avg_logit = np.mean(np.log(img_avg / (1 - img_avg)))

        # For training
        self.batch_size = args.n_batch
        self.batches_in_train = self.train.shape[0] // self.batch_size
        self.batch_idx = 0

        # For validation and test - use batches to preserve memory
        self.n_IS = args.n_IS

        self.batch_size_val = args.n_batch_val
        self.batches_in_val = self.validation.shape[0] // self.batch_size_val
        self.batch_idx_val = 0

        self.batch_size_test = args.n_batch_test
        self.batches_in_test = self.test.shape[0] // self.batch_size_test
        self.batch_idx_test = 0

    def next_train_batch(self):
        """
        Training batches will reshuffle every epoch and involve dynamic 
        binarization
        """

        (batch, self.batch_idx) = self._next_batch(self.batch_idx, 
            self.batches_in_train, self.batch_size, self.train, 
            True, True, False)

        return batch

    def next_val_batch(self):
        """
        Validation batches will be used for ELBO estimates without importance
        sampling (could change)
        """

        (batch, self.batch_idx_val) = self._next_batch(self.batch_idx_val, 
            self.batches_in_val, self.batch_size_val, self.validation, 
            False, False, False)

        return batch

    def next_test_batch(self):
        """
        Test batches are same as validation but with added binarization
        """

        (batch, self.batch_idx_test) = self._next_batch(self.batch_idx_test,
            self.batches_in_test, self.batch_size_test, self.test, 
            False, True, True)

        return batch

    def _next_batch(self, batch_idx, batches_in_set, batch_size, data, 
        reshuffle, binarize, tile):
        """
        Generic function to get next batch.

        Args
            batch_idx: Batch counter
            batches_in_set: Number of batches in the set
            batch_size: Batch size
            data: Dataset to pull from
            reshuffle: If true, reshuffle each epoch
            binarize: If true, perform dynamic binarization
            tile: If true, tile the inputs

        Returns
            batch: Batch of data
            batch_idx: Updated batch index
        """

        if batch_idx >= batches_in_set:
            if reshuffle:
                np.random.shuffle(data)
            batch_idx = 0

        start_idx = batch_idx*batch_size
        end_idx = (batch_idx+1)*batch_size
        batch_raw = data[start_idx:end_idx, :]

        if binarize:
            batch_raw = np.random.binomial(1, batch_raw).astype(np.float32)

        if tile:
            batch = np.tile(batch_raw.reshape(
                [batch_size, 28, 28, 1]), 
                [self.n_IS, 1, 1, 1])
        else:
            batch = batch_raw.reshape([batch_size, 28, 28, 1])

        batch_idx += 1

        return (batch, batch_idx)
