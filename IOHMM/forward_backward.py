'''
The forward backward algorithm of hidden markov model (HMM) .
Mainly used in the E-step of IOHMM given the
(1) initial probabilities, (2) transition probabilities, and (3) emission probabilities.

A feature of this implementation is that it is vectorized to the greatest extent
that we use numpy matrix operation as much as possible.
We have only one for loop in forward/backward calculation,
which is necessary due to dynamic programming (DP).

Another feature of this implementation is that it is calculated at the log level,
so that it is more robust to long sequences.
'''
from __future__ import division

from builtins import range
import warnings


import numpy as np
from scipy.special import logsumexp

warnings.simplefilter("ignore")


def forward_backward(log_prob_initial, log_prob_transition, log_Ey, log_state={}):
    """
    The forward_backward algorithm.
    Parameters
    ----------
    log_prob_initial : array-like of shape (k, )
        where k is the number of states of the HMM
        The log of the probability of initial state at timestamp 0.
        log_prob_initial_{i} is the log of the probability of being in state i
        at timestamp 0.
    log_prob_transition : array-like of shape (t-1, k, k)
        where t is the number of timestamps (length) of the sequence.
        log_prob_transition_{t, i, j} is the log of the probability of transferring
        to state j from state i at timestamp t.
    log_Ey : array-like of shape (t, k)
        log_Ey_{t, i} is the log of the probability of observing emission variables
        from state i at timestamp t.
    log_state: dict(int -> array-like of shape (k, ))
        timestamp i is a key of log_state if we know the state of that timestamp.
        Mostly used in semi-supervised and supervised IOHMM.
        log_state[t][i] is 0 and log_state[t][~i] is -np.Infinity
        if we know the state is i at timestamp t.
    Returns
    -------
    (1) posterior state log probability of each timestamp.
    (2) posterior "transition" log probability of each timestamp.
    (3) log likelihood of the sequence.
    see https://en.wikipedia.org/wiki/Forward-backward_algorithm for details.
    """
    log_alpha = forward(log_prob_initial, log_prob_transition, log_Ey, log_state)
    log_beta = backward(log_prob_transition, log_Ey, log_state)
    log_likelihood = cal_log_likelihood(log_alpha)
    log_gamma = cal_log_gamma(log_alpha, log_beta, log_likelihood, log_state)
    log_epsilon = cal_log_epsilon(log_prob_transition, log_Ey, log_alpha,
                                  log_beta, log_likelihood, log_state)
    return log_gamma, log_epsilon, log_likelihood


def forward(log_prob_initial, log_prob_transition, log_Ey, log_state={}):
    """
    The forward function to calculate log of forward variable alpha.
    Parameters
    ----------
    log_prob_initial : array-like of shape (k, )
        where k is the number of states of the HMM
        The log of the probability of initial state at timestamp 0.
        log_prob_initial_{i} is the log of the probability of being in state i
        at timestamp 0.
    log_prob_transition : array-like of shape (t-1, k, k)
        where t is the number of timestamps (length) of the sequence.
        log_prob_transition_{t, i, j} is the log of the probability of transferring
        to state j from state i at timestamp t.
    log_Ey : array-like of shape (t, k)
        log_Ey_{t, i} is the log of the probability of observing emission variables
        from state i at timestamp t.
    log_state: dict(int -> array-like of shape (k, ))
        timestamp i is a key of log_state if we know the state of that timestamp.
        Mostly used in semi-supervised and supervised IOHMM.
        log_state[t][i] is 0 and log_state[t][~i] is -np.Infinity
        if we know the state is i at timestamp t.
    Returns
    -------
    log_alpha : array-like of shape (t, k)
        log of forward variable alpha.
        see https://en.wikipedia.org/wiki/Forward-backward_algorithm for details.
    """
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
    assert log_alpha.shape == (t, k)
    return log_alpha


def backward(log_prob_transition, log_Ey, log_state={}):
    """
    The function to calculate log of backward variable beta.
    Parameters
    ----------
    log_prob_transition : array-like of shape (t-1, k, k)
        where t is the number of timestamps (length) of the sequence.
        log_prob_transition_{t, i, j} is the log of the probability of transferring
        to state j from state i at timestamp t.
    log_Ey : array-like of shape (t, k)
        log_Ey_{t, i} is the log of the probability of observing emission variables
        from state i at timestamp t.
    log_state: dict(int -> array-like of shape (k, ))
        timestamp i is a key of log_state if we know the state of that timestamp.
        Mostly used in semi-supervised and supervised IOHMM.
        log_state[t][i] is 0 and log_state[t][~i] is -np.Infinity
        if we know the state is i at timestamp t.
    Returns
    -------
    log_beta : array-like of shape (t, k)
        log of backward variable beta.
        see https://en.wikipedia.org/wiki/Forward-backward_algorithm for details.
    """
    assert log_prob_transition.ndim == 3
    assert log_Ey.ndim == 2
    t = log_Ey.shape[0]
    k = log_Ey.shape[1]
    log_beta = np.zeros((t, k))
    for i in range(t - 2, -1, -1):
        if i + 1 in log_state:
            log_beta[i, :] = logsumexp(log_state[i + 1] + log_beta[i + 1, :] + log_Ey[i + 1, :])
        else:
            log_beta[i, :] = logsumexp(log_prob_transition[i, :, :] +
                                       (log_beta[i + 1, :] + log_Ey[i + 1, :]), axis=1)
    assert log_beta.shape == (t, k)
    return log_beta


def cal_log_likelihood(log_alpha):
    """
    The function to calculate the log likelihood of the sequence.
    Parameters
    ----------
    log_alpha : array-like of shape (t, k)
        log of forward variable alpha.
    Returns
    -------
    log_likelihood : float
        The log likelihood of the sequence.
        see https://en.wikipedia.org/wiki/Forward-backward_algorithm for details.
    """
    return logsumexp(log_alpha[-1, :])


def cal_log_gamma(log_alpha, log_beta, log_likelihood, log_state={}):
    """
    The function to calculate the log of the posterior probability of each state
    at each timestamp.
    Parameters
    ----------
    log_alpha : array-like of shape (t, k)
        log of forward variable alpha.
    log_alpha : array-like of shape (t, k)
        log of backward variable beta.
    log_likelihood : float
        log likelihood of the sequence
    log_state: dict(int -> array-like of shape (k, ))
        timestamp i is a key of log_state if we know the state of that timestamp.
        Mostly used in semi-supervised and supervised IOHMM.
        log_state[t][i] is 0 and log_state[t][~i] is -np.Infinity
        if we know the state is i at timestamp t.
    Returns
    -------
    log_gamma : array-like of shape (t, k)
        the log of the posterior probability of each state.
        log_gamma_{t, i} is the posterior log of the probability of
        being in state i at stimestamp t.
        see https://en.wikipedia.org/wiki/Forward-backward_algorithm for details.
    """
    log_gamma = log_alpha + log_beta - log_likelihood
    for i in log_state:
        log_gamma[i, :] = log_state[i]
    return log_gamma


def cal_log_epsilon(log_prob_transition, log_Ey, log_alpha, log_beta, log_likelihood, log_state={}):
    """
    The function to calculate the log of the posterior joint probability
    of two consecutive timestamps
    Parameters
    ----------
    log_prob_transition : array-like of shape (t-1, k, k)
        where t is the number of timestamps (length) of the sequence.
        log_prob_transition_{t, i, j} is the log of the probability of transferring
        to state j from state i at timestamp t.
    log_Ey : array-like of shape (t, k)
        log_Ey_{t, i} is the log of the probability of observing emission variables
        from state i at timestamp t.
    log_alpha : array-like of shape (t, k)
        log of forward variable alpha.
    log_alpha : array-like of shape (t, k)
        log of backward variable beta.
    log_likelihood : float
        log likelihood of the sequence
    log_state: dict(int -> array-like of shape (k, ))
        timestamp i is a key of log_state if we know the state of that timestamp.
        Mostly used in semi-supervised and supervised IOHMM.
        log_state[t][i] is 0 and log_state[t][~i] is -np.Infinity
        if we know the state is i at timestamp t.
    Returns
    -------
    log_epsilon : array-like of shape (t-1, k, k)
        the log of the posterior probability of two consecutive timestamps.
        log_gamma_{t, i, j} is the posterior log of the probability of
        being in state i at timestamp t and
        being in state j at timestamp t+1.
        see https://en.wikipedia.org/wiki/Forward-backward_algorithm for details.
    """
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
