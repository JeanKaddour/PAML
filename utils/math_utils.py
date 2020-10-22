import numpy
import tensorflow as tf
from gpflow import settings
import math
import numpy as np


def gaussian_absolute_moment(pred_var, norm=1):
    factors = ((2 * pred_var ** 2) ** (norm / 2.) * math.gamma((1 + norm) / 2.)) / np.sqrt(np.pi)
    return factors


def mu_std(X):
    mu = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mu, std


def scale(X, x_min, x_max):
    nom = (X - X.min(axis=0)) * (x_max - x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom == 0] = 1
    return x_min + nom / denom


def covariance_scale(var, scale):
    scale = tf.matmul(scale, scale, transpose_a=True)
    return var * scale


def block_diag(M1, M2):
    D1 = tf.shape(input=M1)[0]
    D2 = tf.shape(input=M2)[0]
    UR = tf.zeros((D1, D2), dtype=settings.float_type)
    LL = tf.transpose(a=UR)
    U = tf.concat([M1, UR], 1)
    L = tf.concat([LL, M2], 1)
    return tf.concat([U, L], 0)


def vec_to_matsum(v, op):
    d = tf.shape(input=v)[1]
    v_tile = tf.tile(v[:, :, None], [1, 1, d])
    if op == "sum":
        v_sum = v_tile + v[:, None, :]
    else:
        v_sum = v_tile - v[:, None, :]
    return v_sum


def angular_transform(state_mu, state_var, dim_theta):
    n = tf.shape(input=state_mu)[0]
    d = tf.shape(input=state_mu)[1]

    d_diff = d - dim_theta
    new_d = 2 * dim_theta + d_diff
    theta_mu = state_mu[:, :dim_theta]
    theta_var = state_var[:, :dim_theta, :dim_theta]
    theta_var = tf.linalg.diag_part(theta_var)

    exp_theta_var = tf.exp(-theta_var / 2.)
    cos_theta_mu = tf.cos(theta_mu)
    sin_theta_mu = tf.sin(theta_mu)

    cos_mu = exp_theta_var * cos_theta_mu
    sin_mu = exp_theta_var * sin_theta_mu

    theta_mu_sum = vec_to_matsum(theta_mu, "sum")
    theta_mu_sub = vec_to_matsum(theta_mu, "sub")

    theta_var_sum = vec_to_matsum(theta_var, "sum")
    theta_var_sum = -theta_var_sum / 2.
    exp_theta_var_sum = tf.exp(theta_var_sum)
    exp_term_sum = tf.exp(theta_var_sum + theta_var) - exp_theta_var_sum
    exp_term_sub = tf.exp(theta_var_sum - theta_var) - exp_theta_var_sum

    U1 = exp_term_sum * tf.sin(theta_mu_sub)
    U2 = exp_term_sub * tf.sin(theta_mu_sum)
    U3 = exp_term_sum * tf.cos(theta_mu_sub)
    U4 = exp_term_sub * tf.cos(theta_mu_sum)

    cos_var = U3 + U4
    sin_var = U3 - U4
    cos_sin_cov = U1 + U2
    sin_cos_cov = tf.transpose(a=cos_sin_cov, perm=[0, 2, 1])

    new_theta_mu = tf.concat([cos_mu, sin_mu], 1)
    new_cos_var = tf.concat([cos_var, cos_sin_cov], 2)
    new_sin_var = tf.concat([sin_cos_cov, sin_var], 2)
    new_theta_var = tf.concat([new_cos_var, new_sin_var], 1)

    new_theta_var = new_theta_var / 2.

    cos_mu_diag = tf.linalg.diag(cos_mu)
    sin_mu_diag = -tf.linalg.diag(sin_mu)
    C = tf.concat([sin_mu_diag, cos_mu_diag], 2)
    C = tf.concat([C, tf.zeros(
        (n, d_diff, 2 * dim_theta), dtype=state_mu.dtype)], 1)

    inp_out_cov = tf.matmul(state_var, C)
    new_old_cov = inp_out_cov[:, dim_theta:]
    old_var = state_var[:, dim_theta:, dim_theta:] / 2.

    lower = tf.concat([new_old_cov, old_var], 2)
    right = tf.concat(
        [tf.transpose(a=new_old_cov, perm=[0, 2, 1]), old_var], 1)

    zeros = tf.zeros((n, new_d, 2 * dim_theta), dtype=state_mu.dtype)
    lower = tf.concat([tf.transpose(a=zeros, perm=[0, 2, 1]), lower], 1)
    right = tf.concat([zeros, right], 2)

    zeros = tf.zeros((n, new_d, new_d), dtype=state_mu.dtype)
    new_theta_var = tf.concat([new_theta_var, zeros[:, :d_diff, :2 * dim_theta]], 1)
    new_theta_var = tf.concat([new_theta_var, zeros[:, :, :d_diff]], 2)

    new_state_mu = tf.concat([new_theta_mu, state_mu[:, dim_theta:]], 1)
    new_state_var = new_theta_var + lower + right

    old_diff_cov = state_var[:, :, -d_diff:]
    inp_out_cov = tf.concat([inp_out_cov, old_diff_cov], 2)

    return new_state_mu, new_state_var, inp_out_cov


def normLP(x, norm, axis=1):
    if norm == 1:
        return numpy.sum(numpy.abs(x), axis=axis)
    elif norm == 2:
        return numpy.sqrt(numpy.sum(numpy.square(x), axis=axis))
    else:
        raise Exception('Yet unsupported norm %d!' % norm)


def sample_from_multidim_interval_uniformly(bounds, size):
    samples = []
    for _ in range(size):
        parameters = []
        for bound in bounds:
            parameters.append(np.random.uniform(bound[0], bound[1]))
        samples.append(parameters)
    return np.array(samples).reshape(size, bounds.shape[0])
