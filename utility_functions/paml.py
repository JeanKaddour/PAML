import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky


# We compute our proposed utility function as follows:
# 1. We treat the latent variables as components of a Gaussian Mixture model.
# 2. We compute the log-likelihood of a candidate embedding given the training task latent variables.

def generate_equal_weight_GMM(H_mu, H_var, covariance_type='diag'):
    n_components = len(H_mu)
    weights_init = len(H_mu) * [1. / len(H_mu)]
    GMM = GaussianMixture(n_components=n_components,
                          covariance_type=covariance_type, n_init=0, weights_init=None, means_init=None,
                          precisions_init=None, random_state=None, warm_start=True, verbose=0,
                          verbose_interval=10)
    GMM.weights_ = weights_init
    GMM.means_ = H_mu
    GMM.covariances_ = H_var
    GMM.precisions_cholesky_ = _compute_precision_cholesky(
        H_var, covariance_type)

    return GMM


def select_new_latent_point_PAML(H, X_pool):
    dim_h = int(H.shape[1] / 2)
    means = H[:, :dim_h]
    covariances = H[:, dim_h:]
    X_pool = X_pool[:, :dim_h]
    GMM = generate_equal_weight_GMM(means, covariances)
    scores = GMM.score_samples(X_pool).tolist()
    best_candidate_idx = np.argmin(scores)
    return X_pool[best_candidate_idx].reshape(1, -1)
