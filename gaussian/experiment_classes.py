import numpy as np
import tensorflow as tf
import time
import pickle
import pdb
import os

from conf import params

class BaseModel():
    """ Base model containing generic methods for running the tests """

    def __init__(self, var_names, var_inits, model_name, d):
        """ 
        Initialize model from variable names and initial values 
        
        Args
            var_names: TensorFlow variable names
            var_inits: Initial values for variables as list of numpy arrays
            method_name: Name of the model to test
            d: Dimensionality of the problem
        """

        # Variable initialization
        tf_vars = []
        for i in range(len(var_inits)):
            tf_vars.append(tf.Variable(var_inits[i], dtype=tf.float32))

        self.tf_vars = tf_vars
        self.var_inits = var_inits
        self.model_name = model_name

        # Data placeholder and summary statistics
        self.d = d
        self.tf_x = tf.placeholder(tf.float32, shape=[params['n_data'], d])
        self.x_bar = tf.reduce_mean(self.tf_x, 0)
        self.C_xx = tf.einsum('ij,ik->jk', self.tf_x, self.tf_x)
        self.tf_batch = tf.placeholder(tf.int32)
        self.std_norm = tf.distributions.Normal(
            loc=np.zeros(d, dtype=np.float32), 
            scale=np.ones(d, dtype=np.float32))

        # First two TF variables will be the same across methods
        self.delta = tf_vars[0]
        self.log_sigma = tf_vars[1]

        # ELBO and optimization step defined in specific classes
        self.elbo = None
        self.opt_step = None

        # Filepath to save results
        self.save_dir = os.path.join('save', str(self.d))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def train(self, sess, train_x, train_ind):
        """ 
        Train the model.

        Args:
            sess: TensorFlow session
            train_x: Numpy training data
            train_ind: Index of training run

        Returns:
            delta: Value of delta at end of training
            sigma: Value of sigma at end of training

        The history of the run, given by the hist_dict variable, is also 
        saved by pickle to the file 
            ./save/<d>/<self.method_name>_train_<train_ind>.p
        """      

        # Start by initializing parameter and ELBO history
        hist_length = int(params['n_iter']/params['save_every']) + 1
        hist_dict = {}

        for j in range(len(self.tf_vars)):
            initial = self.var_inits[j]

            # Careful with scalars vs. arrays
            if isinstance(initial, np.ndarray):
                hist_array = np.zeros([hist_length] + list(initial.shape))
                hist_array[0, :] = initial
            else:
                hist_array = np.zeros(hist_length)
                hist_array[0] = initial

            hist_dict[self.tf_vars[j].name] = hist_array

        feed_dict = {self.tf_batch: params['n_batch'], self.tf_x: train_x}
        hist_dict['elbo'] = np.zeros(hist_length)
        hist_dict['elbo'][0] = sess.run(self.elbo, feed_dict)

        # Now move to training loop
        t0 = time.time()

        for i in range(params['n_iter']):
            sess.run(self.opt_step, feed_dict)

            # Record the information more often than we print
            if (i+1) % params['save_every'] == 0:
                save_idx = int((i+1) / params['save_every'])
                np_vars = sess.run(self.tf_vars + [self.elbo], feed_dict)

                # Record the updated data in the history
                for j in range(len(self.tf_vars)):
                    value = np_vars[j]

                    if isinstance(value, np.ndarray):
                        hist_dict[self.tf_vars[j].name][save_idx, :] = value
                    else:
                        hist_dict[self.tf_vars[j].name][save_idx] = value

                hist_dict['elbo'] = np_vars[-1]

                # Assume params['print_every'] divides params['save_every']
                if (i+1) % params['print_every'] == 0:
                    print(('{0}, d: {1:d}, Iter: {2:d}-{3:d}, s/iter:'
                        + ' {4:.3e}, ELBO: {5:.3e}').format(
                        self.model_name,
                        self.d,
                        train_ind+1,
                        i+1,
                        (time.time()-t0) / params['print_every'],
                        np_vars[-1]
                        )
                    )
                    t0 = time.time()

        # Save the data
        save_file = os.path.join(self.save_dir, 
            '{0}_train_{1:d}.p'.format(self.model_name, train_ind))
        pickle.dump(hist_dict, open(save_file, 'wb'))

        (delta, log_sigma) = sess.run((self.delta, self.log_sigma), feed_dict)
        sigma = np.exp(log_sigma)

        return (delta, sigma)

    def _get_yk_z_sig(self, z_in):
        """ 
        Method needed to calculate NLLs common to all models 

        We want to calculate, for each z in the batch, the sum of
            (x_k - mu_X - z)^T * Sigma_X^{-1} * (x_k - mu_X - z)
        over k = 1, ..., N, where:
            - Sigma_X is the estimated model covariance matrix
            - mu_X is the estimated model offset
            - N is the number of datapoints

        Returns:
            yk_z_sig: Tensorflow vector of length self.tf_batch
        """
        yk_z_sig = tf.zeros([self.tf_batch])

        for k in range(params['n_data']):
            x_k = self.tf_x[k ,:]
            yk_z_sig += tf.reduce_sum(
                (x_k-self.delta-z_in)**2 * tf.exp(-2*self.log_sigma), 1)

        return yk_z_sig


class HVAE(BaseModel):
    """ Specific implementation of HVAE """

    def __init__(self, var_names, var_inits, model_name, d, K):
        """ Initialize model including ELBO calculation """

        super().__init__(var_names, var_inits, model_name, d)

        self.K = K
        self.logit_eps = self.tf_vars[2]

        # If there are only three variables, it means we are not tempering
        if len(var_names) == len(var_inits) == 3:
            self.tempering = False
        else:
            self.tempering = True
            self.log_T_0 = self.tf_vars[-1]

        # Get the initial and evolved samples from the variational prior
        (z_0, p_0, z_K, p_K) = self._his()

        # Calculate the optimization objective (ELBO)
        self.elbo = self._get_elbo(z_K, p_K)
        self.opt_step = tf.train.RMSPropOptimizer(
            params['rms_eta']).minimize(-self.elbo, var_list=self.tf_vars)

    def _his(self):
        """ 
        Perform the HIS step to evolve samples

        Returns:
            z_0: Initial position
            p_0: Initial momentum
            z_K: Final position
            p_K: Final momentum
        """
        z_graph = {}
        p_graph = {}

        # Sample initial values with reparametrization if necessary
        z_0 = self.std_norm.sample(self.tf_batch)
        gamma_0 = self.std_norm.sample(self.tf_batch)

        if not self.tempering:
            p_0 = gamma_0
        else:
            p_0 = (1 + tf.exp(self.log_T_0))*gamma_0

        z_graph[0] = z_0
        p_graph[0] = p_0

        # Initialize temperature, define step size
        if not self.tempering:
            T_km1 = 1. # Keep the temperature at 1 throughout if not tempering
            T_k = 1.
        else:
            T_km1 = 1 + tf.exp(self.log_T_0)

        epsilon = params['max_eps'] / (1 + tf.exp(-self.logit_eps))
        var_x = tf.exp(2 * self.log_sigma)

        # Now perform K alternating steps of leapfrog and cooling
        for k in range(1, self.K+1):
                
            # First perform a leapfrog step
            z_in = z_graph[k-1]
            p_in = p_graph[k-1]

            p_half = p_in - 1/2*epsilon*self._dU_dz(z_in, var_x)
            z_k = z_in + epsilon*p_half
            p_temp = p_half - 1/2*epsilon*self._dU_dz(z_k, var_x)

            # Then do tempering
            if self.tempering: 
                T_k = 1 + tf.exp(self.log_T_0)*(1 - k**2/self.K**2)

            p_k = T_k/T_km1 * p_temp

            # End with updating the graph and the previous temperature
            z_graph[k] = z_k
            p_graph[k] = p_k
            T_km1 = T_k

        # Extract final (z_K, p_K)
        z_K = z_graph[self.K]
        p_K = p_graph[self.K]

        return (z_0, p_0, z_K, p_K)

    def _dU_dz(self, z_in, var_x):
        """ Calculate the gradient of the potential wrt z_in """
        grad_U = (z_in 
            + params['n_data']*(z_in + self.delta - self.x_bar)/var_x)
        return grad_U

    def _get_elbo(self, z_K, p_K):
        """ 
        Calculate the ELBO for HVAE 

        Args:
            z_K: Final position after HIS evolution
            p_K: Final momentum after HIS evolution

        Returns:
            elbo: The ELBO objective as a tensorflow object
        """
        var_inv_vec = tf.exp(-2*self.log_sigma)
        var_inv_mat = tf.diag(var_inv_vec)
        trace_term = tf.trace(tf.matmul(var_inv_mat, self.C_xx))

        z_sigX_z = tf.reduce_sum((z_K + self.delta) * var_inv_vec * 
            (z_K + self.delta - 2*self.x_bar), 1)
        z_T_z = tf.reduce_sum(z_K*z_K, 1)
        p_T_p = tf.reduce_sum(p_K*p_K, 1)

        Nd2_log2pi = params['n_data']*self.d/2*np.log(2*np.pi)

        elbo = (- params['n_data']*tf.reduce_sum(self.log_sigma) 
            - trace_term/2 - params['n_data']/2*tf.reduce_mean(z_sigX_z)
            - 1/2*(tf.reduce_mean(z_T_z) + tf.reduce_mean(p_T_p))
            + self.d - Nd2_log2pi)

        return elbo


class NF(BaseModel):
    """ Specific implementation of the Normalizing Flow """

    def __init__(self, var_names, var_inits, model_name, d, K):
        """ Initialize model including ELBO calculation """

        super().__init__(var_names, var_inits, model_name, d)

        self.K = K
        self.u_pre_reparam = self.tf_vars[2]
        self.w = self.tf_vars[3]
        self.b = self.tf_vars[4]

        w_T_u_pre_reparam = tf.reduce_sum(self.u_pre_reparam * self.w)
        self.u = (self.u_pre_reparam + (-1 + tf.nn.softplus(w_T_u_pre_reparam) 
            - w_T_u_pre_reparam)*self.w / tf.reduce_sum(self.w**2))

        # Get the initial and evolved samples from the variational prior
        (z_0, z_K, log_det_sum) = self._nf()

        # Calculate the optimization objective (ELBO)
        self.elbo = self._get_elbo(z_K, log_det_sum)
        self.opt_step = tf.train.RMSPropOptimizer(
            params['rms_eta']).minimize(-self.elbo, var_list=self.tf_vars)

    def _nf(self):
        """ 
        Perform the flow step to get the final samples

        Returns:
            z_0: Initial sample from variational prior
            z_K: Final sample after normalizing flow
            log_det_sum: Sum of log determinant terms at each flow step
        """
        z_graph = {}
        log_det_sum = tf.constant(0., dtype=tf.float32)

        # Begin with sampling from the variational prior
        z_0 = self.std_norm.sample(self.tf_batch)
        z_graph[0] = z_0

        # Need awkward operations to deal with broadcasting
        u_tiled = tf.tile(tf.reshape(self.u, [self.d, 1]), 
            [1, self.tf_batch])

        # Now perform the flow steps
        for i in range(1, self.K+1):

            # Evolution bit
            z_in = z_graph[i-1]

            w_T_z = tf.reduce_sum(self.w * z_in, 1)
            post_tanh = tf.tanh(w_T_z + self.b)
            u_tanh = tf.transpose(u_tiled * post_tanh)

            z_out = z_in + u_tanh
            z_graph[i] = z_out

            # log determinant terms 
            u_T_w = tf.reduce_sum(self.u * self.w)
            log_det_batch = tf.log(1 + (1 - post_tanh**2) * u_T_w)

            log_det_sum += tf.reduce_mean(log_det_batch)

        # Extract final z_K
        z_K = z_graph[self.K]

        return (z_0, z_K, log_det_sum)

    def _get_elbo(self, z_K, log_det_sum):
        """ 
        Calculate the ELBO for NF 

        Args:
            z_K: Final sample after NF evolution
            log_det_sum: Sum of log-determinant terms

        Returns:
            elbo: The ELBO objective as a tensorflow object
        """
        var_inv_vec = tf.exp(-2 * self.log_sigma)

        # Note that we say y = x - mu_X for ease of naming
        y_sig_y = tf.reduce_sum((self.tf_x - self.delta)**2 * var_inv_vec)
        y_bar_sig_z = tf.reduce_sum((self.x_bar - self.delta) 
                                    * var_inv_vec * z_K, 1)
        z_sig_z = tf.reduce_sum(z_K**2 * var_inv_vec, 1)
        z_T_z = tf.reduce_sum(z_K * z_K, 1)

        Nd2_log2pi = params['n_data']*self.d/2*np.log(2*np.pi)
        
        elbo = (- Nd2_log2pi - params['n_data']*tf.reduce_sum(self.log_sigma)
            - y_sig_y + params['n_data']*tf.reduce_mean(y_bar_sig_z)
            - params['n_data']/2.*tf.reduce_mean(z_sig_z) 
            - 1./2*tf.reduce_mean(z_T_z) + log_det_sum)

        return elbo


class VB(BaseModel):
    """ Specific implementation of Variational Bayes """

    def __init__(self, var_names, var_inits, model_name, d):
        """ Initialize model including ELBO calculation """

        super().__init__(var_names, var_inits, model_name, d)

        self.mu_z = self.tf_vars[2]
        self.log_sigma_z = self.tf_vars[3]

        # Calculate the optimization objective (ELBO)
        self.elbo = self._get_elbo()
        self.opt_step = tf.train.RMSPropOptimizer(
            params['rms_eta']).minimize(-self.elbo, var_list=self.tf_vars)

    def _get_elbo(self):
        """ 
        Calculate the ELBO for the VB example

        Returns:
            elbo: The ELBO objective as a tensorflow object
        """
        var_inv_vec = tf.exp(-2 * self.log_sigma)

        # Note that for this VB scheme the ELBO is completely deterministic
        y_sig_y = tf.reduce_sum((self.tf_x - self.delta)**2 * var_inv_vec)
        y_sig_mu = tf.reduce_sum((self.tf_x - self.delta) * var_inv_vec 
                                                        * self.mu_z)

        var_Z_over_var_X = tf.reduce_sum(tf.exp(2*self.log_sigma_z) 
                                        * var_inv_vec)

        mu_sig_mu = tf.reduce_sum(self.mu_z**2 * var_inv_vec)
        mu_T_mu = tf.reduce_sum(self.mu_z**2)

        Nd2_log2pi = params['n_data']*self.d/2*np.log(2*np.pi)

        elbo = (- Nd2_log2pi + tf.reduce_sum(self.log_sigma_z) 
            - params['n_data']*tf.reduce_sum(self.log_sigma) - 1/2*y_sig_y 
            + y_sig_mu - params['n_data']/2*(var_Z_over_var_X + mu_sig_mu)
            - 1/2*tf.reduce_sum(tf.exp(2*self.log_sigma_z)) - 1/2*mu_T_mu 
            - self.d/2
            )

        return elbo