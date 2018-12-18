# MNIST Tests

This folder contains tests comparing a base Variational Auto-Encoder (VAE) to a Hamiltonian Variational Auto-Encoder (HVAE) on the MNIST handwritten digits dataset.

## Running the tests

The full suite of tests to replicate the results of the paper are available in `tests.sh`. 
These test the VAE and HVAE with various settings and random seeds.
It is not recommended to run all of the tests at once (this will take a long time), but rather to run them individually.

## Viewing the Test Results

Note that when a test is started, it is assigned a unique identifier of the form <YYYY-MM-DD_hh_mm_ss>.
This identifier will be used to index the test results.
Methods for viewing the tests results are listed below.

### Reading and Parsing the Log File

After running a test, some information about the test will be appended to the file `log.txt`.
Upon running several tests, this file will become harder to read.
It is possible to parse this file into a .csv using the following python code:
```python
from utils import parse_log
parse_log()
```

The csv file will then be available as `log.csv`.

### Viewing the Plots

Each test also produces two plots into the directory `plots`.
The first, under file name `<test_ID>_loss.png`, shows the training and validation (negative) ELBO at each epoch.
The second, under file name `<test_ID>_samples.png`, gives samples from the learned generative model.

### Loading Saved Checkpoints

The learned models for each test are also saved in TensorFlow checkpoint files.
For each test, there is a TensorFlow checkpoint saved in `checkpoints/<test_ID>/`.
The method for saving and loading models with the `tf.train.Saver()` object in TensorFlow is demonstrated in the `main.py` file.

### Information in Pickle Files

Only three things are saved in the pickle file, located in `pickle/<test_ID>.p`.
The first is the history of training ELBO at each epoch.
The second is the history of validation ELBO at each epoch.
The third is the list of arguments for the given experiments (in the `args` structure).

## Short Description of Code Files in This Directory

* `adamax.py`: This file only contains the ADAMAX Optimizer.
* `data.py`: This has methods for handling MNIST batches.
	- Note that `tf.contrib.learn.datasets.load_dataset()` is deprecated. You will have to update at some point.
* `main.py`: This is the main file for running the experiments. You can see which parameters can be sent from the command line here.
* `models.py`: This contains the VAE and HVAE models as classes. HVAE inherits from VAE.
* `plotting.py`: This contains two methods: one for plotting the training curves, and one for plotting samples from the generative network.
* `training.py`: This file contains methods for doing one epoch of training, performing a validation step, and evaluating the model over the test set.
* `utils.py`: This file helps with setting GPUs and parsing the log file, along with containing a method to delete unwanted tests.

## Acknowledgement

I would like to thank the creators of the Sylvester Normalizing Flows code (available [here](https://github.com/riannevdberg/sylvester-flows)) for inspiring the format and structure of this code.
