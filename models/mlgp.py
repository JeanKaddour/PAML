import tensorflow as tf
import numpy as np
from tensorflow.contrib import distributions as dist

from models.gpflowmod.svgp import SVGP
from gpflow import settings
from gpflow.params import Parameter
from gpflow.decors import params_as_tensors
from models.gpflowmod.conditionals import conditional, uncertain_conditional
from gpflow.conditionals import base_conditional
from utils.math_utils import block_diag
from gpflow.logdensities import multivariate_normal
from gpflow.mean_functions import Zero


class BASESVGP(SVGP):

    def __init__(self,
                 dim_in, dim_out,
                 kern, likelihood,
                 feat=None,
                 mean_function=None,
                 q_diag=False,
                 whiten=True,
                 Z=None,
                 num_data=None,
                 **kwargs):
        super(BASESVGP, self).__init__(
            dim_in=dim_in, dim_out=dim_out,
            kern=kern, likelihood=likelihood,
            feat=feat,
            mean_function=mean_function,
            q_diag=q_diag,
            whiten=whiten,
            Z=Z,
            num_data=num_data,

            **kwargs)

    @params_as_tensors
    def _build_predict_uncertain(self, Xnew_mu, Xnew_var,
                                 full_cov=False, full_output_cov=False, Luu=None):
        mu, var, inp_out_cov = uncertain_conditional(
            Xnew_mu=Xnew_mu, Xnew_var=Xnew_var, feat=self.feature, kern=self.kern,
            q_mu=self.q_mu, q_sqrt=self.q_sqrt, Luu=Luu,
            mean_function=self.mean_function,
            full_cov=full_cov, full_output_cov=full_output_cov, white=self.whiten)

        return mu, var, inp_out_cov

    @params_as_tensors
    def compute_Luu(self):
        Kuu = self.feature.Kuu(self.kern, jitter=settings.numerics.jitter_level)
        return tf.cholesky(Kuu)

    def get_model_param(self, session):
        lik_noise = self.likelihood.variance.read_value(session=session)
        kern_var = self.kern.variance.read_value(session=session)
        kern_ls = self.kern.lengthscales.read_value(session=session)
        return lik_noise, kern_var, kern_ls


class MLSVGP(BASESVGP):

    def __init__(self,
                 dim_in, dim_out,
                 kern, likelihood,
                 dim_h, num_h,
                 feat=None,
                 mean_function=None,
                 q_diag=False,
                 whiten=True,
                 Z=None,
                 num_data=None,
                 observed_config_space_dim=0,
                 latent_to_conf_space_kernel=None,
                 latent_to_conf_space_likelihood=None,
                 **kwargs):
        super(MLSVGP, self).__init__(
            dim_in=dim_in, dim_out=dim_out,
            kern=kern, likelihood=likelihood,
            feat=feat,
            mean_function=mean_function,
            q_diag=q_diag,
            whiten=whiten,
            Z=Z,
            num_data=num_data,
            **kwargs)

        self.dim_h = dim_h
        self.num_h = num_h
        self.observed_config_space = observed_config_space_dim
        self.mean_psi = Zero(observed_config_space_dim)
        self.configuration_kernel = latent_to_conf_space_kernel
        self.configuration_likelihood = latent_to_conf_space_likelihood

        # Initialize task variables
        H_mu = np.random.randn(num_h, dim_h)
        H_var = np.log(np.ones_like(H_mu) * 0.1)
        H_init = np.hstack([H_mu, H_var])
        self.H = Parameter(H_init, dtype=settings.float_type, name="H")

        # Create placeholders
        self.H_ids_ph = tf.placeholder(tf.int32, [None])
        self.H_unique_ph = tf.placeholder(tf.int32, [None])
        self.H_scale = tf.placeholder(settings.float_type, [])
        self.psi_ph = tf.placeholder(dtype=settings.float_type, shape=[None, self.observed_config_space])

    @params_as_tensors
    def build_likelihood(self):
        # Get prior KL.
        KL_U = self.build_prior_KL()

        H_sample, KL_H = self.sample_qH(self.H)

        KL_H = tf.reduce_sum(tf.gather(KL_H, self.H_unique_ph))
        KL_H *= self.H_scale

        H_sample_Yph = tf.gather(H_sample, self.H_ids_ph)
        H_sample_psi = tf.gather(H_sample, self.H_unique_ph)

        XH = tf.concat([self.X_mu_ph, H_sample_Yph], 1)

        # Get conditionals
        fmean, fvar = self._build_predict(XH, full_cov=False)

        # Get variational expectations.
        var_exp_Y = self.likelihood.variational_expectations(fmean, fvar, self.Y_ph)

        lik_term = (tf.reduce_sum(var_exp_Y) + self._build_likelihood_psi(H_sample_psi)) * self.data_scale

        likelihood = lik_term - KL_U - KL_H

        return likelihood

    @params_as_tensors
    def _build_predict(self, XHnew, full_cov=False, full_output_cov=False):
        mu, var = conditional(XHnew, self.feature, self.kern, self.q_mu, q_sqrt=self.q_sqrt, full_cov=full_cov,
                              white=self.whiten, full_output_cov=full_output_cov)
        return mu + self.mean_function(XHnew), var

    @params_as_tensors
    def _build_likelihood_psi(self, H_sample):
        r"""
        Construct configuration tensorflow function to compute the likelihood.

            \log p(Y | theta).

        """

        K = self.configuration_kernel.K(H_sample) + tf.eye(tf.shape(H_sample)[0],
                                                           dtype=settings.float_type) * self.configuration_likelihood.variance
        L = tf.cholesky(K)
        m = self.mean_psi(H_sample)
        logpdf = multivariate_normal(self.psi_ph, m, L)  # (R,) log-likelihoods for each independent dimension of Y

        return tf.reduce_sum(logpdf)

    @params_as_tensors
    def build_predict_psi(self, Hnew, full_cov=False):
        H_sample = tf.gather(self.H[:, :self.dim_h], self.H_unique_ph)
        y = self.psi_ph - self.mean_psi(H_sample)
        Kmn = self.configuration_kernel.K(H_sample, Hnew)
        Kmm_sigma = self.configuration_kernel.K(H_sample) + tf.eye(tf.shape(H_sample)[0],
                                                                   dtype=settings.float_type) * self.configuration_likelihood.variance
        Knn = self.configuration_kernel.K(Hnew) if full_cov else self.configuration_kernel.Kdiag(Hnew)
        f_mean, f_var = base_conditional(Kmn, Kmm_sigma, Knn, y, full_cov=full_cov,
                                         white=False)  # N x P, N x P or P x N x N
        return f_mean + self.mean_psi(Hnew), f_var

    @params_as_tensors
    def _build_predict_uncertain(self, Xnew_mu, Xnew_var,
                                 full_cov=False, full_output_cov=False, Luu=None):
        H = tf.gather(self.H, self.H_ids_ph)
        H_mu = H[:, :self.dim_h]
        H_var = tf.matrix_diag(tf.exp(H[:, self.dim_h:]))

        XH_mu = tf.concat([Xnew_mu, H_mu], 1)
        XH_var = block_diag(Xnew_var[0], H_var[0])[None, :, :]

        mu, var, inp_out_cov = uncertain_conditional(
            Xnew_mu=XH_mu, Xnew_var=XH_var, feat=self.feature, kern=self.kern,
            q_mu=self.q_mu, q_sqrt=self.q_sqrt, Luu=Luu,
            mean_function=self.mean_function,
            full_cov=full_cov, full_output_cov=full_output_cov, white=self.whiten)

        return mu, var, inp_out_cov[:, :-self.dim_h]

    @params_as_tensors
    def build_predict_uncertain(self, XH_mu, XH_var,
                                full_cov=False, full_output_cov=False, Luu=None):
        mu, var, inp_out_cov = uncertain_conditional(
            Xnew_mu=XH_mu, Xnew_var=XH_var, feat=self.feature, kern=self.kern,
            q_mu=self.q_mu, q_sqrt=self.q_sqrt, Luu=Luu,
            mean_function=self.mean_function,
            full_cov=full_cov, full_output_cov=full_output_cov, white=self.whiten)

        return mu, var, inp_out_cov[:, :-self.dim_h]

    @params_as_tensors
    def sample_qH(self, H):
        h_mu = H[:, :self.dim_h]
        h_var = tf.exp(H[:, self.dim_h:])
        qh = dist.Normal(h_mu, tf.sqrt(h_var))
        ph = dist.Normal(tf.zeros_like(h_mu), tf.ones_like(h_var))
        kl_h = dist.kl_divergence(qh, ph)
        h_sample = qh.sample()
        return h_sample, kl_h

    @params_as_tensors
    def compute_Luu(self):
        Kuu = self.feature.Kuu(self.kern, jitter=settings.numerics.jitter_level)
        return tf.cholesky(Kuu)

    def get_H_space(self, session):
        H = self.H.read_value(session=session)
        H_mu = H[:, :self.dim_h]
        H_var = np.exp(H[:, self.dim_h:])
        return H_mu, H_var

    def get_model_param(self, session):
        lik_noise = self.likelihood.variance.read_value(session=session)
        kern_var = self.kern.variance.read_value(session=session)
        kern_ls = self.kern.lengthscales.read_value(session=session)
        return lik_noise, kern_var, kern_ls

    def get_H_subset(self, session, end_task_id, start_task_id=0):
        H = self.H.read_value(session=session)[start_task_id:end_task_id]
        H[:, self.dim_h:] = np.exp(H[:, self.dim_h:])
        return H
