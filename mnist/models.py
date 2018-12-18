"""
This script contains the VAE and HVAE classes. HVAE inherits from VAE
"""

from argparse import Namespace
import pdb
import tensorflow as tf
import numpy as np

class VAE():
    """
    Class for implementing the structure of the base VAE

    This class contains the following:
        - Inference and Generative Networks, including initializations
        - ELBO and NLL calculations
    """

    def __init__(self, args, avg_logit):
        
        self.z_dim = args.z_dim
        self.avg_logit = avg_logit

        self._set_net_params()

    def get_elbo(self, x, args):
        """
        Get average ELBO over a batch of points x

        Args
            x: Batch of input points - Tensorflow placeholder
            args: Command line arguments

        Returns
            elbo: Tensorflow float denoting average ELBO over batch
        """

        q_mu, q_sigma = self._inf_network(x)
        # The variational distribution is a Normal with mean and
        # standard deviation given by the inference network
        q_z = tf.distributions.Normal(loc=q_mu, scale=q_sigma)

        assert (q_z.reparameterization_type
            == tf.distributions.FULLY_REPARAMETERIZED)

        # The likelihood is Bernoulli-distributed with logits given by the
        # generative network, prior is standard multivariate normal
        z = q_z.sample()
        p_x_given_z_logits = self._gen_network(z)
        p_x_given_z = tf.distributions.Bernoulli(logits=p_x_given_z_logits)
        p_z = tf.distributions.Normal(
            loc=np.zeros(self.z_dim, dtype=np.float32),
            scale=np.ones(self.z_dim, dtype=np.float32))

        # Build the evidence lower bound (ELBO) or the negative loss
        kl = tf.reduce_sum(tf.distributions.kl_divergence(q_z, p_z), 1)
        expected_log_likelihood = tf.reduce_sum(p_x_given_z.log_prob(x),
                                              [1, 2, 3])
        elbo = tf.reduce_mean(expected_log_likelihood - kl)

        return elbo

    def get_nll(self, x, args):
        """
        Get average NLL over a batch of points x

        Args
            x: Batch of input points - Tensorflow placeholder
            args: Command line arguments

        Returns
            nll: Tensorflow float denoting average NLL over batch
        """

        q_mu, q_sigma = self._inf_network(x)
        q_z = tf.distributions.Normal(loc=q_mu, scale=q_sigma)

        z = q_z.sample()
        p_x_given_z_logits = self._gen_network(z)
        p_x_given_z = tf.distributions.Bernoulli(logits=p_x_given_z_logits)
        expected_log_likelihood = tf.reduce_sum(p_x_given_z.log_prob(x),
                                              [1, 2, 3])

        # Calculate log p(z) - log q(z | x) for NLL
        prior_minus_gen = (-1./2*tf.reduce_sum(z*z, axis=1)
            + tf.reduce_sum(tf.log(q_sigma), axis=1)
            + 1./2*tf.reduce_sum(((z - q_mu)/q_sigma)**2, axis=1))
        nll_samples = expected_log_likelihood + prior_minus_gen

        # Expect a huge, tiled vector for the NLL calculation. Need to 
        # reshape this to perform log-sum-exp over importance samples
        nll_samples_reshaped = tf.reshape(nll_samples, 
            [args.n_IS, args.n_batch_test])
        nll_lse = tf.reduce_logsumexp(nll_samples_reshaped, axis=0)
        nll = np.log(args.n_IS) - tf.reduce_mean(nll_lse)

        return nll

    def get_samples(self, args):
        """
        Get a tensorflow object representing samples from the gen. model
        """

        prior = tf.distributions.Normal(
            loc=np.zeros(self.z_dim, dtype=np.float32),
            scale=np.ones(self.z_dim, dtype=np.float32))
        z_0 = prior.sample(args.n_gen_samples)

        mnist_h = 28
        mnist_w = 28

        samples = tf.reshape(tf.sigmoid(self._gen_network(z_0)),
            (args.n_gen_samples, mnist_h, mnist_w))
        
        return samples

    def _set_net_params(self):
        """ 
        Initialize network parameters

        This function makes a dictionary of variables for the inference
        and generative networks -- with mostly hard-coded network 
        structures -- and sets the class variables i_params and g_params
        corresponding to the variable dictionaries for the inference and
        generative networks, respectively.
        """

        # Inference network first: convolutional parts, then fully-connected
        # parts, and then the biases
        i_wt_varshapes = [
            [5, 5, 1, 16], 
            [5, 5, 16, 32], 
            [5, 5, 32, 32],
            [512, 450], 
            [450, self.z_dim]
        ]
        i_wt_indims = [
            [28, 28, 1],
            [14, 14, 16],
            [7, 7, 32], 
            [512], 
            [450], 
            [self.z_dim]
        ]
        i_wt_params = self._bulk_init_wts(i_wt_varshapes, i_wt_indims, 
            'i_')

        i_bias_varshapes = [16, 32, 32, 450, self.z_dim, self.z_dim]
        i_bias_inits = np.zeros(len(i_bias_varshapes))
        i_bias_params = self._bulk_init_biases(i_bias_varshapes, 
            i_bias_inits, 'i_')

        self.i_params = {**i_wt_params, **i_bias_params}

        # Generative network next
        g_wt_varshapes = [
            [self.z_dim, 450], 
            [450, 512], 
            [5, 5, 32, 32], 
            [5, 5, 16, 32], 
            [5, 5, 1, 16]
        ]
        g_wt_indims = list(reversed(i_wt_indims))
        g_wt_params = self._bulk_init_wts(g_wt_varshapes, g_wt_indims, 
            'g_')

        g_bias_varshapes = [450, 512, 32, 16, 1]
        g_bias_inits = [0., 0., 0., 0., self.avg_logit]
        g_bias_params = self._bulk_init_biases(g_bias_varshapes, 
            g_bias_inits, 'g_')

        self.g_params = {**g_wt_params, **g_bias_params}

    def _bulk_init_wts(self, varshapes, indims, prefix):
        """
        Initialize dictionary of weight variables
        """
        weights = {}

        # Want to use Glorot initialization
        for i in range(len(varshapes)):

            s = varshapes[i]

            n_in = np.prod(indims[i])
            n_out = np.prod(indims[i+1])

            if i == len(varshapes)-1 and prefix == 'i_':
                # Initialize two variables in this case: one for sigma and
                # one for mu
                init_mu = np.random.normal(scale=1./np.sqrt(n_in+n_out),
                                        size=tuple(s)).astype(np.float32)
                init_sig = np.random.normal(scale=1./np.sqrt(n_in+n_out),
                                        size=tuple(s)).astype(np.float32)

                mu_name = 'W{0:d}_mu'.format(i+1)
                sig_name = 'W{0:d}_sig'.format(i+1)

                weights[mu_name] = tf.Variable(init_mu, name=prefix+mu_name)
                weights[sig_name] = tf.Variable(init_sig, 
                                                name=prefix+sig_name)

            else:
                init = np.random.normal(scale=1./np.sqrt(n_in+n_out), 
                                        size=tuple(s)).astype(np.float32)
                w_name = 'W{0:d}'.format(i+1)
                weights[w_name] = tf.Variable(init, name=prefix+w_name)

        return weights

    def _bulk_init_biases(self, varshapes, inits, prefix):
        """
        Initialize dictionary of bias variables
        """
        biases = {}

        # Simply use constant values
        for i in range(len(varshapes)):            

            init = tf.constant(inits[i], shape=[varshapes[i]], dtype=tf.float32)
            name = 'b{0:d}'.format(i+1)
            biases[name] = tf.Variable(init, name=prefix+name)

        return biases

    def _inf_network(self, x):
        """
        Inference network parametrizing a Gaussian.

        We use three convolutional layers, each with softplus 
        activations, and then a fully-connected layer feeding into a 
        hidden state, and then we fully-connect the hidden state to the 
        means and variances.

        Args
            x: Batch of MNIST images - Tensorflow placeholder

        Returns
            mu: Mean parameters of the Gaussian
            sigma: Standard deviation parameters of the Gaussian
        """

        # Load all params from the shared object
        i = Namespace(**self.i_params)

        # Convolutional layers - note that the feature maps in h3 will be 4x4
        h1 = tf.nn.softplus(self._conv2d(x, i.W1, 2) + i.b1)
        h2 = tf.nn.softplus(self._conv2d(h1, i.W2, 2) + i.b2)
        h3 = tf.nn.softplus(self._conv2d(h2, i.W3, 2) + i.b3)

        # First flatten h3 to send to fully-connected layers
        h3_flat = tf.contrib.layers.flatten(h3)

        h4 = tf.nn.softplus(tf.matmul(h3_flat, i.W4) + i.b4)

        # Fully-connected layers for mean and variance. Note sigma > 0.
        mu = tf.matmul(h4, i.W5_mu) + i.b5
        sigma = tf.nn.softplus(tf.matmul(h4, i.W5_sig) + i.b6)

        return mu, sigma

    def _gen_network(self, z_in):
        """
        Generative network parametrizing a product of Bernoullis.

        We will do the reverese of the inference network as mentioned 
        in Salimans et al. [2015]. We use a fully-connected layer first
        to put z into the correct shape, and then another layer with a 
        hidden state, and then perform 3 layers of transposed 
        convolutions to upsample the image. We return logits for the 
        associated Bernoulli distribution.

        Args:
            z_in: A batch of latent variables.

        Returns:
            bernoulli_logits: Logits for generative bernoulli model.
        """

        # Load all parameters for generative network
        g = Namespace(**self.g_params)
        W1 = self.g_params['W1']   # shape [latent_dim, hidden_dim]
        b1 = self.g_params['b1']
        W2 = self.g_params['W2']   # shape [hidden_dim, 4*4*32]
        b2 = self.g_params['b2']
        W3 = self.g_params['W3']   # shape [5, 5, 32, 32]
        b3 = self.g_params['b3']
        W4 = self.g_params['W4']   # shape [5, 5, 16, 32]
        b4 = self.g_params['b4']
        W5 = self.g_params['W5']   # shape [5, 5, 1, 16]
        b5 = self.g_params['b5']   

        # Need to back out batch size for deconvolution
        batch = tf.shape(z_in)[0]

        # Start with two fully-conmnected layers and then reshape
        h1 = tf.nn.softplus(tf.matmul(z_in, g.W1) + g.b1)
        h2_flat = tf.nn.softplus(tf.matmul(h1, g.W2) + g.b2)
        n_h2_maps = 32
        h2 = tf.reshape(h2_flat, [-1, 4, 4, n_h2_maps])

        # Three layers of tranposed convolution
        h3 = tf.nn.softplus(self._deconv2d(h2, g.W3, 
            tf.stack([batch, 7, 7, 32]), 2) + g.b3)
        h4 = tf.nn.softplus(self._deconv2d(h3, g.W4, 
            tf.stack([batch, 14, 14, 16]), 2) + g.b4)
        bernoulli_logits = self._deconv2d(h4, g.W5, 
            tf.stack([batch, 28, 28, 1]), 2) + g.b5

        return bernoulli_logits

    def _conv2d(self, x, W, stride=1):
        return tf.nn.conv2d(x, W, [1, stride, stride, 1], 'SAME')

    def _deconv2d(self, x, W, shape, stride=1):
        # We need to specify the output shape for the deconvolution operation
        return tf.nn.conv2d_transpose(x, W, shape, [1, stride, stride, 1])

class HVAE(VAE):
    """
    Class for implementing the structure of the base HVAE
    """
    
    def __init__(self, args, avg_logit):        

        # Need to specify Hamiltonian variables before the BaseVAE 
        # initialization so that they can be referenced in the ELBO
        # calculation
        self.K = args.K

        # Restrict epsilon to be between 0 and max_lf
        init_lf = args.init_lf * np.ones(args.z_dim)
        init_lf_reparam = np.log(init_lf / (args.max_lf-init_lf))

        if args.vary_eps == 'true':
            # If we vary epsilon across layers, we need K different
            # epsilon vectors
            init_lf_reparam = np.tile(init_lf_reparam, (self.K, 1))

        lf_reparam = tf.Variable(init_lf_reparam, dtype=tf.float32, 
            name='lf_eps_reparameterized')
        self.lf_eps = args.max_lf / (1+tf.exp(-lf_reparam))

        self.temp_method = args.temp_method

        if self.temp_method == 'free':
            # Variable tempering schedule to be learned
            # Want individual tempering params to be between 0 and 1
            init_alphas = args.init_alpha * np.ones(self.K)
            init_alphas_reparam = np.log(init_alphas / (1-init_alphas))
            alphas_reparam = tf.Variable(init_alphas_reparam, 
                dtype=tf.float32, name='alpha_reparameterized')

            self.alphas = tf.sigmoid(alphas_reparam)
            self.T_0 = tf.reduce_prod(self.alphas)**(-2)

        elif self.temp_method == 'fixed':
            # Fixed temperature schedule for varying initial temperature
            # Want initial temperature to be greater than 1 
            init_T_0 = args.init_T_0
            init_T_0_reparam = np.log(init_T_0 - 1)
            T_0_reparam = tf.Variable(init_T_0_reparam, 
                dtype=tf.float32, name='T_0_reparameterized')

            self.T_0 = 1 + tf.exp(T_0_reparam)

            k_vec = np.arange(1, self.K+1)
            k_m_1_vec = np.arange(0, self.K)

            temp_sched = (1-self.T_0)*k_vec**2/self.K**2 + self.T_0
            temp_sched_m_1 = (1-self.T_0)*k_m_1_vec**2/self.K**2 + self.T_0

            self.alphas = tf.sqrt(temp_sched / temp_sched_m_1)

        elif self.temp_method == 'none':
            # No tempering at all
            self.T_0 = 1.
            self.alphas = np.ones(self.K, dtype=np.float32)

        else:
            raise ValueError('Tempering method {0} not supported'.format(
                                                        temp_method))

        super().__init__(args, avg_logit)

    def get_elbo(self, x, args):
        """
        Get average ELBO over a batch of points x

        Args
            x: Batch of input points - Tensorflow placeholder
            args: Command line arguments

        Returns
            elbo: Tensorflow float denoting average ELBO over batch
        """

        q_mu, q_sigma = self._inf_network(x)

        # Sample from variaitonal prior and standard normal to init. flow
        std_norm = tf.distributions.Normal(loc=tf.fill(tf.shape(q_mu), 0.),
                                        scale=tf.fill(tf.shape(q_mu), 1.))
        z_0 = q_mu + q_sigma*std_norm.sample()
        p_0 = tf.sqrt(self.T_0) * std_norm.sample()

        # Flow evolution
        (z_K, p_K) = self._his(z_0, p_0, x, args)

        # Calculate expected log likelihood from generative network
        p_x_given_zK_logits = self._gen_network(z_K)
        p_x_given_zK = tf.distributions.Bernoulli(logits=p_x_given_zK_logits)
        expected_log_likelihood = tf.reduce_sum(p_x_given_zK.log_prob(x),
                                                [1, 2, 3])

        # The 'kl' term is the rest
        log_prob_zK = -1./2 * tf.reduce_sum(z_K*z_K, 1)
        log_prob_pK = -1./2 * tf.reduce_sum(p_K*p_K, 1)
        sum_log_sigma = tf.reduce_sum(tf.log(q_sigma), 1)

        neg_kl_term = log_prob_zK + log_prob_pK + sum_log_sigma + self.z_dim
        elbo = tf.reduce_mean(expected_log_likelihood + neg_kl_term)

        return elbo

    def get_nll(self, x, args):
        """
        Get average NLL over a batch of points x

        Args
            x: Batch of input points - Tensorflow placeholder
            args: Command line arguments

        Returns
            nll: Tensorflow float denoting average NLL over batch
        """

        q_mu, q_sigma = self._inf_network(x)

        # Sample from variaitonal prior and standard normal to init. flow
        std_norm = tf.distributions.Normal(loc=tf.fill(tf.shape(q_mu), 0.),
                                        scale=tf.fill(tf.shape(q_mu), 1.))
        z_0 = q_mu + q_sigma*std_norm.sample()
        p_0 = tf.sqrt(self.T_0) * std_norm.sample()

        # Flow evolution
        (z_K, p_K) = self._his(z_0, p_0, x, args)

        # Calculate expected log likelihood from generative network
        p_x_given_zK_logits = self._gen_network(z_K)
        p_x_given_zK = tf.distributions.Bernoulli(logits=p_x_given_zK_logits)
        expected_log_likelihood = tf.reduce_sum(p_x_given_zK.log_prob(x),
                                                [1, 2, 3])

        # The 'kl' term is the rest
        log_prob_zK = -1./2 * tf.reduce_sum(z_K*z_K, 1)
        log_prob_pK = -1./2 * tf.reduce_sum(p_K*p_K, 1)
        sum_log_sigma = tf.reduce_sum(tf.log(q_sigma), 1)

        # NLL requires explicit calculations with z_0 and p_0, whereas
        # in the ELBO these additional terms are integrated analytically
        log_prob_z0 = -1./2 * tf.reduce_sum(((z_0-q_mu)/q_sigma)**2, 1)
        log_prob_p0 = -1./(2*self.T_0) * tf.reduce_sum(p_0*p_0, 1)

        nll_samples = (expected_log_likelihood + log_prob_zK + log_prob_pK
            + sum_log_sigma - log_prob_z0 - log_prob_p0)

        # Expect a huge, tiled vector for the NLL calculation. Need to 
        # reshape this to perform log-sum-exp over importance samples
        nll_samples_reshaped = tf.reshape(nll_samples, 
            [args.n_IS, args.n_batch_test])
        nll_lse = tf.reduce_logsumexp(nll_samples_reshaped, axis=0)

        nll = np.log(args.n_IS) - tf.reduce_mean(nll_lse)

        return nll

    def _his(self, z_0, p_0, x, args):
        """ Evolve (z_0, p_0) according to HIS """
        z_graph = {}
        p_graph = {}
        z_graph[0] = z_0
        p_graph[0] = p_0

        # Alternate K steps of leapfrog and cooling
        for k in range(1, self.K+1):

            # Begin with leapfrog
            z_in = z_graph[k-1]
            p_in = p_graph[k-1]

            if args.vary_eps == 'true':
                lf_eps = self.lf_eps[k-1,:]
            else:
                lf_eps = self.lf_eps

            p_half = p_in - 1./2*lf_eps*self._dU_dz(z_in, x)
            z_k = z_in + lf_eps*p_half
            p_temp = p_half - 1./2*lf_eps*self._dU_dz(z_k, x)

            # Proceed to tempering
            p_k = self.alphas[k-1] * p_temp

            # Update the graph
            z_graph[k] = z_k
            p_graph[k] = p_k

        # Extract the final position and momentum
        z_K = z_graph[self.K]
        p_K = p_graph[self.K]

        return z_K, p_K

    def _dU_dz(self, z, x):
        """ Calculate the gradient of -log(p(x | z)) """
        net_out = self._gen_network(z)
        U = tf.reduce_sum(tf.nn.softplus(net_out) - x*net_out, (1,2,3))

        grad_U = tf.gradients(xs=z, ys=U)[0] + z

        return grad_U
