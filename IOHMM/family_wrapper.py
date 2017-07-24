'''
The extension of statsmodels one parameter exponential family distributions used by GLM,
with the added functionality for log likelihood per sample.
'''
# original code from
# https://github.com/statsmodels/statsmodels/blob/master/statsmodels/genmod/families/family.py
# Need to wait since the statsmodel formal package is not up to date as in github
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
    The parent class for forwarding one-parameter exponential families,
    with function for per sample probability.
    Parameters
    ----------
    link : a link function instance
        Link is the linear transformation function.
        See the individual families for available links.
    variance : a variance function
        Measures the variance as a function of the mean probabilities.
        See the individual families for the default variance function.
    See Also
    --------
    :ref:`links`
    """

    def __init__(self, link, variance):
        raise NotImplementedError

    def loglike_per_sample(self, endog, mu, scale=1.):
        """
        The log-likelihood function in terms of the fitted mean response.
        Parameters
        ----------
        endog : array-like
            Endogenous response variable
        mu : array-like
            Fitted mean response variable
        scale : float, optional
            The scale parameter, defaults to 1.
        Returns
        -------
        llf : array-like
            The value of the loglikelihood function evaluated per sample.
            The shape should be (n_samples, )
        """
        raise NotImplementedError


class PoissonWrapper(FamilyWrapper):
    """
    Poisson exponential family with function for per sample probability.
    Parameters
    ----------
    link : a link instance, optional
        The default link for the Poisson family is the log link. Available
        links are log, identity, and sqrt. See statsmodels.family.links for
        more information.
    Attributes
    ----------
    Poisson.link : a link instance
        The link function of the Poisson instance.
    Poisson.variance : varfuncs instance
        `variance` is an instance of
        statsmodels.genmod.families.family.varfuncs.mu
    See also
    --------
    statsmodels.genmod.families.family.Family
    :ref:`links`
    """

    def __init__(self, link=L.log):
        # For now the statsmodels 0.8.0 still takes a link as an argument
        # will follow the changes in statsmodels whenever it happens
        self.family = Poisson(link=link)

    def loglike_per_sample(self, endog, mu, scale=1.):
        r"""
        The log-likelihood function in terms of the fitted mean response.
        Parameters
        ----------
        endog : array-like
            Endogenous response variable
        mu : array-like
            Fitted mean response variable
        scale : float, optional
            Not used for in the Poisson loglike.
        Returns
        -------
        llf : array-like
            The value of the loglikelihood function evaluated per sample
            (endog,mu,freq_weights,scale) as defined below.
        Notes
        -----
        .. math::
           llf_i = scale * (Y_i * \log(\mu_i) - \mu_i -
                 \ln \Gamma(Y_i + 1))
        """
        return (endog * np.log(mu) - mu -
                special.gammaln(endog + 1)).reshape(-1,)


class GaussianWrapper(FamilyWrapper):
    """
    Gaussian exponential family distribution,
    with function for per sample probability.
    Parameters
    ----------
    link : a link instance, optional
        The default link for the Gaussian family is the identity link.
        Available links are log, identity, and inverse.
        See statsmodels.family.links for more information.
    Attributes
    ----------
    Gaussian.link : a link instance
        The link function of the Gaussian instance
    Gaussian.variance : varfunc instance
        `variance` is an instance of statsmodels.family.varfuncs.constant
    See also
    --------
    statsmodels.genmod.families.family.Family
    :ref:`links`
    """

    def __init__(self, link=L.identity):
        self.family = Gaussian(link=link)

    def loglike_per_sample(self, endog, mu, scale=1.):
        """
        The log-likelihood in terms of the fitted mean response.
        Parameters
        ----------
        endog : array-like
            Endogenous response variable
        mu : array-like
            Fitted mean response variable
        scale : float, optional
            Scales the loglikelihood function. The default is 1.
        Returns
        -------
        llf : array-like
            The value of the loglikelihood function evaluated per sample
            (endog,mu,freq_weights,scale) as defined below.

        Notes
        -----
        llf_i = - 1 / 2 * ((Y_i - mu_i)^2 / scale + log(2 * \pi * scale))
        """
        if scale > EPS:
            return ((endog * mu - mu**2 / 2.) / scale -
                    endog**2 / (2 * scale) - .5 * np.log(2 * np.pi * scale)).reshape(-1,)
        else:
            log_p = np.zeros(endog.shape[0])
            log_p[~np.isclose(endog, mu)] = - np.Infinity
            return log_p


class GammaWrapper(FamilyWrapper):
    """
    Gamma exponential family distribution.
    with function for per sample probability.
    Parameters
    ----------
    link : a link instance, optional
        The default link for the Gamma family is the inverse link.
        Available links are log, identity, and inverse.
        See statsmodels.family.links for more information.
    Attributes
    ----------
    Gamma.link : a link instance
        The link function of the Gamma instance
    Gamma.variance : varfunc instance
        `variance` is an instance of statsmodels.family.varfuncs.mu_squared
    See also
    --------
    statsmodels.genmod.families.family.Family
    :ref:`links`
    """

    def __init__(self, link=L.inverse_power):
        self.family = Gamma(link=link)

    def loglike_per_sample(self, endog, mu, scale=1.):
        """
        The log-likelihood function in terms of the fitted mean response.
        Parameters
        ----------
        endog : array-like
            Endogenous response variable
        mu : array-like
            Fitted mean response variable
        scale : float, optional
            The default is 1.
        Returns
        -------
        llf : array-like
            The value of the loglikelihood function evaluated per sample
            (endog,mu,freq_weights,scale) as defined below.
        Notes
        --------
        llf_i = -1 / scale * (Y_i / \mu_i+ \log(\mu_i)+
                 (scale -1) * \log(Y) + \log(scale) + scale *
                 \ln \Gamma(1 / scale))
        """
        if scale > EPS:
            endog_mu = self.family._clean(endog / mu)
            return (-(endog_mu - np.log(endog_mu) + scale *
                      np.log(endog) + np.log(scale) + scale *
                      special.gammaln(1. / scale)) / scale).reshape(-1,)
        else:
            log_p = np.zeros(endog.shape[0])
            log_p[~np.isclose(endog, mu)] = - np.Infinity
            return log_p


class BinomialWrapper(FamilyWrapper):
    """
    Binomial exponential family distribution.
    with function for per sample probability.
    Parameters
    ----------
    link : a link instance, optional
        The default link for the Binomial family is the logit link.
        Available links are logit, probit, cauchy, log, and cloglog.
        See statsmodels.family.links for more information.
    Attributes
    ----------
    Binomial.link : a link instance
        The link function of the Binomial instance
    Binomial.variance : varfunc instance
        `variance` is an instance of statsmodels.family.varfuncs.binary
    See also
    --------
    statsmodels.genmod.families.family.Family
    :ref:`links`
    Notes
    -----
    endog for Binomial can be specified in one of three ways.
    """

    def __init__(self, link=L.logit):  # , n=1.):
        # TODO: it *should* work for a constant n>1 actually, if data_weights
        # is equal to n
        self.family = Binomial(link=link)

    def loglike_per_sample(self, endog, mu, scale=1.):
        """
        The log-likelihood function in terms of the fitted mean response.
        Parameters
        ----------
        endog : array-like
            Endogenous response variable
        mu : array-like
            Fitted mean response variable
        scale : float, optional
            Not used for the Binomial GLM.
        Returns
        -------
        llf : array-like
            The value of the loglikelihood function evaluated per sample
            (endog,mu,freq_weights,scale) as defined below.
        Notes
        --------
        If the endogenous variable is binary:
        .. math::
         llf_i = (y_i * \log(\mu_i/(1-\mu_i)) + \log(1-\mu_i))
        If the endogenous variable is binomial:
        .. math::
           llf = (\ln \Gamma(n+1) -
                 \ln \Gamma(y_i + 1) - \ln \Gamma(n_i - y_i +1) + y_i *
                 \log(\mu_i / (n_i - \mu_i)) + n * \log(1 - \mu_i/n_i))
        where :math:`y_i = Y_i * n_i` with :math:`Y_i` and :math:`n_i` as
        defined in Binomial initialize.  This simply makes :math:`y_i` the
        original number of successes.
        """
        # special setup
        # see _Setup_binomial(self) in generalized_linear_model.py
        tmp = self.family.initialize(endog, 1)
        endog = tmp[0]
        if np.shape(self.family.n) == () and self.family.n == 1:
            return scale * (endog * np.log(mu / (1 - mu) + 1e-200) +
                            np.log(1 - mu)).reshape(-1,)
        else:
            y = endog * self.family.n  # convert back to successes
            return scale * (special.gammaln(self.family.n + 1) -
                            special.gammaln(y + 1) -
                            special.gammaln(self.family.n - y + 1) + y *
                            np.log(mu / (1 - mu)) + self.family.n *
                            np.log(1 - mu)).reshape(-1,)


class InverseGaussianWrapper(FamilyWrapper):
    """
    InverseGaussian exponential family.
    with function for per sample probability.
    Parameters
    ----------
    link : a link instance, optional
        The default link for the inverse Gaussian family is the
        inverse squared link.
        Available links are inverse_squared, inverse, log, and identity.
        See statsmodels.family.links for more information.
    Attributes
    ----------
    InverseGaussian.link : a link instance
        The link function of the inverse Gaussian instance
    InverseGaussian.variance : varfunc instance
        `variance` is an instance of statsmodels.family.varfuncs.mu_cubed
    See also
    --------
    statsmodels.genmod.families.family.Family
    :ref:`links`
    Notes
    -----
    The inverse Guassian distribution is sometimes referred to in the
    literature as the Wald distribution.
    """

    def __init__(self, link=L.inverse_squared):
        self.family = InverseGaussian(link=link)

    def loglike_per_sample(self, endog, mu, scale=1.):
        """
        The log-likelihood function in terms of the fitted mean response.
        Parameters
        ----------
        endog : array-like
            Endogenous response variable
        mu : array-like
            Fitted mean response variable
        scale : float, optional
            The default is 1.
        Returns
        -------
        llf : array-like
            The value of the loglikelihood function evaluated per sample
            (endog,mu,freq_weights,scale) as defined below.
        Notes
        -----
        llf_i = -1/2 * ((Y_i - \mu_i)^2 / (Y_i *
                 \mu_i^2 * scale) + \log(scale * Y_i^3) + \log(2 * \pi))
        """
        if scale > EPS:
            return -.5 * ((endog - mu)**2 / (endog * mu**2 * scale) +
                          np.log(scale * endog**3) + np.log(2 * np.pi)).reshape(-1,)
        else:
            log_p = np.zeros(endog.shape[0])
            log_p[~np.isclose(endog, mu)] = - np.Infinity
            return log_p


class NegativeBinomialWrapper(FamilyWrapper):
    """
    Negative Binomial exponential family.
    with function for per sample probability.
    Parameters
    ----------
    link : a link instance, optional
        The default link for the negative binomial family is the log link.
        Available links are log, cloglog, identity, nbinom and power.
        See statsmodels.family.links for more information.
    alpha : float, optional
        The ancillary parameter for the negative binomial distribution.
        For now `alpha` is assumed to be nonstochastic.  The default value
        is 1.  Permissible values are usually assumed to be between .01 and 2.
    Attributes
    ----------
    NegativeBinomial.link : a link instance
        The link function of the negative binomial instance
    NegativeBinomial.variance : varfunc instance
        `variance` is an instance of statsmodels.family.varfuncs.nbinom
    See also
    --------
    statsmodels.genmod.families.family.Family
    :ref:`links`
    Notes
    -----
    Power link functions are not yet supported.
    """

    def __init__(self, link=L.log, alpha=1.):
        # make it at least float
        self.family = NegativeBinomial(link=link, alpha=alpha)

    def loglike_per_sample(self, endog, mu, scale):
        """
        The log-likelihood function in terms of the fitted mean response.
        Parameters
        ----------
        endog : array-like
            Endogenous response variable
        mu : array-like
            The fitted mean response values
        scale : float
            The scale parameter
        Returns
        -------
        llf : array-like
            The value of the loglikelihood function evaluated per sample
            (endog,mu,freq_weights,scale) as defined below.
        Notes
        -----
        Defined as:
        .. math::
           llf = (Y_i * \log{(\alpha * \mu_i /
                 (1 + \alpha * \mu_i))} - \log{(1 + \alpha * \mu_i)}/
                 \alpha + Constant)
        where :math:`Constant` is defined as:
        .. math::
           Constant = \ln \Gamma{(Y_i + 1/ \alpha )} - \ln \Gamma(Y_i + 1) -
                      \ln \Gamma{(1/ \alpha )}
        """
        if scale > EPS:
            lin_pred = self.family._link(mu)
            constant = (special.gammaln(endog + 1 / self.family.alpha) -
                        special.gammaln(endog + 1) - special.gammaln(1 / self.family.alpha))
            exp_lin_pred = np.exp(lin_pred)
            return (endog * np.log(self.family.alpha * exp_lin_pred /
                                   (1 + self.family.alpha * exp_lin_pred)) -
                    np.log(1 + self.family.alpha * exp_lin_pred) /
                    self.family.alpha + constant).reshape(-1,)
        else:
            log_p = np.zeros(endog.shape[0])
            log_p[~np.isclose(endog, mu)] = - np.Infinity
            return log_p
