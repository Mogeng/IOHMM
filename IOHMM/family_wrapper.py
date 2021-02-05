'''
The Wrapper of statsmodels one parameter exponential family distributions used by GLM,
with the added functionality for log likelihood per sample. Loglikelihood per sample is
going to be used in IOHMM to estimate emission probability.
'''
from __future__ import division

from past.utils import old_div
from builtins import object
import numpy as np
from scipy import special
from statsmodels.genmod.families.family import (Poisson,
                                                Gaussian,
                                                Gamma,
                                                Binomial,
                                                InverseGaussian,
                                                NegativeBinomial)
import statsmodels.genmod.families.links as L

EPS = np.finfo(float).eps


class FamilyWrapper(object):
    """
    The parent class for the wrapper of one-parameter exponential families,
    with function for per sample loglikelihood.
    Parameters
    ----------
    link : a link function instance
        Link is the linear transformation function.
        See the individual families for available links.
    variance : a variance function
        Measures the variance as a function of the mean probabilities.
        See the individual families for the default variance function.
    Attributes
    ----------
    family : a statsmodels corresponding family object
    --------
    """

    def __init__(self, link, variance):
        raise NotImplementedError

    def loglike_per_sample(self, endog, mu, scale=1.):
        """
        The function to calculate log-likelihood per sample
        in terms of the fitted mean response.
        Parameters
        ----------
        endog : array-like.
            Endogenous response variable
            For binomial family, it could be of shape (n, ) or (n, k).
            where n is the number of samples and k is number of classes.abs
            For other families, it should be of shape (n, ).
        mu : array-like
            should be of shape (n, )
            Fitted mean response variable
        scale : float, optional
            The scale parameter, defaults to 1.
        Returns
        -------
        log_p : array-like
            The value of the loglikelihood function evaluated per sample.
            The shape should be (n, )
        """
        if scale > EPS:
            return self.family.loglike_obs(endog, mu, scale=scale)
        else:
            log_p = np.zeros(endog.shape[0])
            log_p[~np.isclose(endog, mu)] = - np.Infinity
            return log_p


class PoissonWrapper(FamilyWrapper):
    """
    The wrapper for Poisson exponential family.
    Subclass of FamilyWrapper
    Parameters
    ----------
    link : a link instance, optional
        The default link for the Poisson family is the log link. Available
        links are log, identity, and sqrt. See statsmodels.family.links for
        more information.
    Attributes
    ----------
    family : a statsmodels Possion family object
    --------
    """

    def __init__(self, link=L.log):
        # For now the statsmodels 0.8.0 still takes a link as an argument
        # will follow the changes in statsmodels whenever it happens
        self.family = Poisson(link=link)


class GaussianWrapper(FamilyWrapper):
    """
    The wrapper of Gaussian exponential family distribution,
    with function for per sample probability.
    Parameters
    ----------
    link : a link instance, optional
        The default link for the Gaussian family is the identity link.
        Available links are log, identity, and inverse.
        See statsmodels.family.links for more information.
    Attributes
    ----------
    family : a statsmodel Gaussian family object
    --------
    """

    def __init__(self, link=L.identity):
        self.family = Gaussian(link=link)



class GammaWrapper(FamilyWrapper):
    """
    The wrapper of Gaussian exponential family distribution,
    with function for per sample probability.
    Parameters
    ----------
    link : a link instance, optional
        The default link for the Gaussian family is the identity link.
        Available links are log, identity, and inverse.
        See statsmodels.family.links for more information.
    Attributes
    ----------
    family : a statsmodel Gaussian family object
    --------
    """

    def __init__(self, link=L.inverse_power):
        self.family = Gamma(link=link)


class BinomialWrapper(FamilyWrapper):
    """
    The wrapper of Binomial exponential family distribution,
    with function for per sample probability.
    Parameters
    ----------
    link : a link instance, optional
        The default link for the Binomial family is the logit link.
        Available links are logit, probit, cauchy, log, and cloglog.
        See statsmodels.family.links for more information.
    Attributes
    ----------
    family : a statsmodel Binomial family object
    --------
    """

    def __init__(self, link=L.logit):  # , n=1.):
        # TODO: it *should* work for a constant n>1 actually, if data_weights
        # is equal to n
        self.family = Binomial(link=link)


class InverseGaussianWrapper(FamilyWrapper):
    """
    The wrapper of InverseGaussian exponential family distribution,
    with function for per sample probability.
    Parameters
    ----------
    link : a link instance, optional
        The default link for the InverseGaussian family is the identity link.
        Available links are inverse_squared, inverse, log, and identity.
        See statsmodels.family.links for more information.
    Attributes
    ----------
    family : a statsmodel InverseGaussian family object
    --------
    """

    def __init__(self, link=L.inverse_squared):
        self.family = InverseGaussian(link=link)


class NegativeBinomialWrapper(FamilyWrapper):
    """
    The wrapper of NegativeBinomial exponential family distribution,
    with function for per sample probability.
    Parameters
    ----------
    link : a link instance, optional
        The default link for the NegativeBinomial family is the identity link.
        Available links are log, cloglog, identity, nbinom and power.
        See statsmodels.family.links for more information.
    Attributes
    ----------
    family : a statsmodel NegativeBinomial family object
    --------
    """

    def __init__(self, link=L.log, alpha=1.):
        # make it at least float
        self.family = NegativeBinomial(link=link, alpha=alpha)
