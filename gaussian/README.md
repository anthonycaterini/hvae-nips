# Gaussian Tests

The test is run using the following command:
	`python3 run_test.py`
Note that this may take a long time to run and you may want to separate the tests by dimension.
Test configuration is given in `conf.py`.

Afterwards, the results can be plotted and summarized using the `plot_results.py` script which saves the two figures.

## List of files

* `conf.py`: The configuration file specifying the parameters of the test
* `experiment_classes.py`: The classes for each of the experiment types
* `plot_results_script.py`: Parses the results of the tests and makes plots
* `plot_utils.py`: Helper script for plotting
* `run_test.py`: Script for actually running the test. Do this before plotting results