from __future__ import  division
import numpy as np
from scipy.misc import logsumexp
import warnings
warnings.simplefilter("ignore")

def calLogAlpha(log_prob_initial, log_prob_transition, log_Ey):

    ## return the alpha matrix which should be t * k
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
    log_alpha[0, :] = log_prob_initial + log_Ey[0,:]
    for i in range(1, t):
        log_alpha[i, :] = logsumexp(log_prob_transition[i-1,:,:].T + log_alpha[i-1,:], axis = 1) + log_Ey[i,:]
    assert log_alpha.ndim == 2
    return log_alpha

def calLogBeta(log_prob_transition, log_Ey):

    ## return the alpha matrix which should be t * k
    # pi should be k * 1
    # transition prob shoud be (t - 1) * k * k
    # emission prob should be t * k * q
    # y should be t * k
    assert len(log_prob_transition.shape) == 3
    assert len(log_Ey.shape) == 2
    t = log_Ey.shape[0]
    k = log_Ey.shape[1]
    log_beta = np.zeros((t, k))
    for i in range(t-2, -1, -1):
        log_beta[i, :] = logsumexp(log_prob_transition[i, :, :] + (log_beta[i+1, :] + log_Ey[i+1,:]), axis = 1)
    assert len(log_beta.shape) == 2
    return log_beta

def calLogLikelihood(log_alpha):
    return logsumexp(log_alpha[-1,:])

def calLogGamma(log_alpha, log_beta, ll):
    return log_alpha + log_beta - ll

def calGamma(log_alpha, log_beta, ll):
    return np.exp(calLogGamma(log_alpha, log_beta, ll))

def calLogEpsilon(log_prob_transition, log_Ey, log_alpha, log_beta, ll):
    # epsilon should be t - 1 * k * k
    k = log_Ey.shape[1]
    return np.tile((log_Ey + log_beta)[1:,np.newaxis,:], [1,k,1]) + np.tile(log_alpha[:-1,:,np.newaxis], [1,1,k]) + log_prob_transition - ll

def calEpsilon(log_prob_transition, log_Ey, log_alpha, log_beta, ll):
    return np.exp(calLogEpsilon(log_prob_transition, log_Ey, log_alpha, log_beta, ll))

def calHMM(log_prob_initial, log_prob_transition, log_Ey):

    log_alpha = calLogAlpha(log_prob_initial, log_prob_transition, log_Ey)
    log_beta = calLogBeta(log_prob_transition, log_Ey)
    ll = calLogLikelihood(log_alpha)
    log_gamma = calLogGamma(log_alpha, log_beta, ll)
    log_epsilon = calLogEpsilon(log_prob_transition, log_Ey, log_alpha, log_beta, ll)
    return log_gamma, log_epsilon, ll



