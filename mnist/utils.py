import tensorflow as tf
import GPUtil
import os
import shutil

def set_gpus(n_gpus):
    """
    Find GPUs to use, if possible. Return TF config
    """

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    try:
        device_IDs = GPUtil.getAvailable(order='load', limit=n_gpus)
    except FileNotFoundError:
        print('\n---- No GPUs on this machine ----\n')
        return (gpu_config, 0)

    if len(device_IDs) > 0:
        str_device_list = ','.join([str(x) for x in device_IDs])
        gpu_config.gpu_options.visible_device_list = str_device_list

        if len(device_IDs) < n_gpus:
            print('\n**** Note: {0} GPUs requested, but only {1} found ****'.
                format(n_gpus, len(device_IDs)))

        print('\n---- Running on GPU(s) {} ----\n'.format(str_device_list))

    else:
        print('\n---- No GPUs available! ----\n')

    return (gpu_config, len(device_IDs))

def delete_test(id_str, check_dir='checkpoints', pickle_dir='pickle', 
    plot_dir='plots'):
    """
    Function to delete the checkpoint, pickle file, and plots for test id_str

    Note - Should also manually remove the test from the log file
    """

    # Delete checkpoint first
    id_check_dir = os.path.join(check_dir, id_str)

    try:
        shutil.rmtree(id_check_dir)
    except OSError as e:
        print('Error: {} - {}.'.format(e.filename, e.strerror))

    # Now the pickle file
    pickle_file = os.path.join(pickle_dir, id_str + '.p')

    try:
        os.remove(pickle_file)
    except OSError as e:
        print('Error: {} - {}.'.format(e.filename, e.strerror))

    # Finally the plots
    plot_list = os.listdir(plot_dir)

    for f in plot_list:
        if id_str in f:
            try:
                os.remove(os.path.join(plot_dir, f))
            except OSError as e:
                print('Error: {} - {}.'.format(e.filename, e.strerror))

def parse_log():
    """ 
    This function turns the log file into a more readable CSV

    The log file is given by 'log.txt', and we output the parsed 
    result to 'log.csv'
    """
    log_txt = 'log.txt'
    log_csv = 'log.csv'

    csv_out = open(log_csv, 'w')
    csv_out.write('ID,Model,Net Size,ES Epochs,K,Tempering,Vary Epsilon,' + 
        'Warmup,Learn Rate,Seed,Total Epochs,Epoch Time,ELBO,ELBO Error,' + 
        'NLL,NLL Error\n')

    out_list = []

    with open(log_txt, 'r') as file_in:
        line = 'temp' # Initialize to some non-null dummy string

        # Loop terminates when file is read completely
        while line:

            line = file_in.readline().rstrip('\n')

            # Advance if the file is done or not at start of a test
            if line == '':
                continue

            if line[0] != '-': 
                continue

            csv_line_list = []

            # Begin parsing the test with the ID
            csv_line_list.append(line.split(' ')[3])

            # Next line is where most parsing occurs, from the Namespace
            namespace = file_in.readline().rstrip(')\n').lstrip('Namespace(')
            param_list = namespace.split(', ')
            param_tpls = [(x.split('=')[0], x.split('=')[1]) 
                            for x in param_list]

            # Find the parameters and add them to the csv line
            def get_from_tpl_list(param):
                return [x[1] for x in param_tpls if x[0] == param][0]

            log_param_names = ['model', 'net_size', 'early_stopping_epochs',
                'K', 'temp_method', 'vary_eps', 'n_warmup', 'learn_rate', 
                'seed']

            for p in log_param_names:

                try:
                    csv_line_list.append(get_from_tpl_list(p).strip("'"))
                except IndexError as e:
                    # Early versions of logging didn't have network size
                    # or varied leapfrog stepsizes per flow step
                    if p == 'net_size':
                        csv_line_list.append('large')
                    if p == 'vary_eps':
                        csv_line_list.append('false')

            # Total epochs info
            csv_line_list.append(file_in.readline().split(' ')[2])

            line = file_in.readline() # Skip this line

            # Training time
            csv_line_list.append(file_in.readline().split(' ')[5])

            line = file_in.readline() # Skip this line

            # ELBO information
            elbo_line = file_in.readline().rstrip('\n').split(' ')
            csv_line_list.append(elbo_line[2])
            csv_line_list.append(elbo_line[4])

            # NLL information
            nll_line = file_in.readline().rstrip('\n').split(' ')
            csv_line_list.append(nll_line[2])
            csv_line_list.append(nll_line[4])

            out_list.append(csv_line_list.copy())

            line = file_in.readline()

    # Now, write the line in CSV with newlines at the end
    for i in range(len(out_list)):

        csv_str = ','.join(out_list[i]) + '\n'
        csv_out.write(csv_str)

    csv_out.close()
