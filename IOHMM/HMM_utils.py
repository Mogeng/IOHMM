from __future__ import division

import warnings


import numpy as np
from scipy.misc import logsumexp

warnings.simplefilter("ignore")


def cal_log_alpha(log_prob_initial, log_prob_transition, log_Ey, log_state={}):

    # return the alpha matrix which should be t * k
    # initial should be k * 1
    # transition prob shoud be (t - 1) * k * k
    # emission prob should be t * k * q
    # Ey should be t * k
    assert log_prob_initial.ndim == 1
    assert log_prob_transition.ndim == 3
    assert log_Ey.ndim == 2
    t = log_Ey.shape[0]
    k = log_Ey.shape[1]
    log_alpha = np.zeros((t, k))
    if 0 in log_state:
        log_alpha[0, :] = log_state[0] + log_Ey[0, :]
    else:
        log_alpha[0, :] = log_prob_initial + log_Ey[0, :]
    for i in range(1, t):
        if i in log_state:
            log_alpha[i, :] = logsumexp(log_alpha[i - 1, :]) + log_state[i] + log_Ey[i, :]
        else:
            log_alpha[i, :] = logsumexp(log_prob_transition[i - 1, :, :].T +
                                        log_alpha[i - 1, :], axis=1) + log_Ey[i, :]
    assert log_alpha.ndim == 2
    return log_alpha


def cal_log_beta(log_prob_transition, log_Ey, log_state={}):

    # return the alpha matrix which should be t * k
    # pi should be k * 1
    # transition prob shoud be (t - 1) * k * k
    # emission prob should be t * k * q
    # y should be t * k
    assert len(log_prob_transition.shape) == 3
    assert len(log_Ey.shape) == 2
    t = log_Ey.shape[0]
    k = log_Ey.shape[1]
    log_beta = np.zeros((t, k))
    for i in range(t - 2, -1, -1):
        if i + 1 in log_state:
            log_beta[i, :] = logsumexp(log_state[i + 1] + log_beta[i + 1, :] + log_Ey[i + 1, :])
        else:
            log_beta[i, :] = logsumexp(log_prob_transition[i, :, :] +
                                       (log_beta[i + 1, :] + log_Ey[i + 1, :]), axis=1)
    assert len(log_beta.shape) == 2
    return log_beta


def cal_log_likelihood(log_alpha):
    return logsumexp(log_alpha[-1, :])


def cal_log_gamma(log_alpha, log_beta, log_likelihood, log_state={}):
    log_gamma = log_alpha + log_beta - log_likelihood
    for i in log_state:
        log_gamma[i, :] = log_state[i]
    return log_gamma


def cal_log_epsilon(log_prob_transition, log_Ey, log_alpha, log_beta, log_likelihood, log_state={}):
    # epsilon should be t - 1 * k * k
    k = log_Ey.shape[1]
    if log_prob_transition.shape[0] == 0:
        return np.zeros((0, k, k))
    else:
        log_p = log_prob_transition
        for i in log_state:
            log_p[i - 1, :, :] = log_state[i]
        log_epsilon = np.tile((log_Ey + log_beta)[1:, np.newaxis, :], [1, k, 1]) + \
            np.tile(log_alpha[:-1, :, np.newaxis], [1, 1, k]) + log_p - log_likelihood
        for i in log_state:
            if i + 1 in log_state:
                log_epsilon[i, :, :] = np.add.outer(log_state[i], log_state[i + 1])
        return log_epsilon


def cal_HMM(log_prob_initial, log_prob_transition, log_Ey, log_state={}):
    # assume here log_state is a dictionary
    log_alpha = cal_log_alpha(log_prob_initial, log_prob_transition, log_Ey, log_state)
    log_beta = cal_log_beta(log_prob_transition, log_Ey, log_state)
    log_likelihood = cal_log_likelihood(log_alpha)
    log_gamma = cal_log_gamma(log_alpha, log_beta, log_likelihood, log_state)
    log_epsilon = cal_log_epsilon(log_prob_transition, log_Ey, log_alpha,
                                  log_beta, log_likelihood, log_state)
    return log_gamma, log_epsilon, log_likelihood
