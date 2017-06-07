from __future__ import division
import numpy as np
import family
from sklearn import linear_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.optimize import newton_cg
from scipy import optimize
from scipy.misc import logsumexp
import statsmodels.regression.linear_model as lim
from statsmodels.tools.sm_exceptions import PerfectSeparationError
import sys
import warnings
warnings.simplefilter("ignore")


def _rescale_data(X, Y, sample_weight):
    """Rescale data so as to support sample_weight"""
    sqrtW = np.sqrt(sample_weight)
    newX = X * sqrtW.reshape(-1, 1)
    newY = Y * sqrtW.reshape(-1, 1)
    return newX, newY


def addIntercept(X):
    t = X.shape[0]
    X_with_bias = np.hstack((np.ones((t, 1)), X))
    return X_with_bias


class BaseModel(object):
    """
    A generic supervised model for data with input and output.
    BaseModel does nothing, but lays out the methods expected of any subclass.
    """

    def __init__(self, fam, solver, fit_intercept=True, est_sd=False,
                 penalty=None, reg=0, l1_ratio=0, tol=1e-4, max_iter=100):
        """
        Constructor
        Parameters
        ----------
        fam: family of the GLM, LM or MNL
        solver: family specific solver
        penalty: penalty to regularize the model
        reg: regularization strenth
        l1_ratio: if elastic net, the l1 reg ratio
        tol: tol in the optimization procedure
        max_iter: max_iter in the optimization procedure
        -------
        """
        self.fit_intercept = fit_intercept
        self.penalty = penalty
        self.reg = reg
        self.l1_ratio = l1_ratio
        self.fam = fam
        self.solver = solver
        self.tol = tol
        self.max_iter = max_iter
        self.est_sd = est_sd

    def fit(self, X, Y, sample_weight=None):
        """
        fit the weighted model
        Parameters
        ----------
        X : design matrix
        Y : response matrix
        sample_weight: sample weight vector
        """
        raise NotImplementedError

    def predict(self, X):
        """
        predict the Y value based on the model
        ----------
        X : design matrix
        Returns
        -------
        predicted value
        """
        return NotImplementedError

    def probability(self, X, Y):
        """
        Given a set of X and Y, calculate the probability of
        observing Y value
        """
        logP = self.log_probability(X, Y)
        if logP is not None:
            return np.exp(self.log_probability(X, Y))
        else:
            return None

    def log_probability(self, X, Y):
        """
        Given a set of X and Y, calculate the log probability of
        observing each of Y value given each X value

        should return a vector
        """
        return NotImplementedError

    def estimate_dispersion(self):
        raise NotImplementedError

    def estimate_sd(self):
        raise NotImplementedError

    def estimate_loglikelihood(self):
        raise NotImplementedError


class GLM(BaseModel):
    """
    A Generalized linear model for data with input and output.
    """
    def __init__(self, fam, solver='pinv', fit_intercept=True, est_sd=False, penalty=None,
                 reg=0, l1_ratio=0, tol=1e-4, max_iter=100):
        super(GLM, self).__init__(fam=fam, solver=solver, fit_intercept=fit_intercept,
                                  est_sd=est_sd, penalty=penalty, reg=reg, l1_ratio=l1_ratio,
                                  tol=tol, max_iter=max_iter)

    def fit(self, X, Y, sample_weight=None):
        """
        fit the weighted model
        Parameters
        ----------
        X : design matrix
        Y : response matrix
        sample_weight: sample weight vector

        """
        # family is the glm family with link, the family is the same as in the statsmodel
        if sample_weight is None:
            sample_weight = np.ones((X.shape[0],))
        assert X.shape[0] == sample_weight.shape[0]
        assert X.shape[0] == Y.shape[0]
        assert Y.ndim == 1 or Y.shape[1] == 1
        Y = Y.reshape(-1,)

        sum_w = np.sum(sample_weight)
        assert sum_w > 0
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.fit_intercept:
            X = addIntercept(X)
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.n_targets = 1
        # start fitting using irls
        mu = self.fam.starting_mu(Y)
        lin_pred = self.fam.predict(mu)
        dev = self.fam.deviance_weighted(Y, mu, sample_weight)
        if np.isnan(dev):
            raise ValueError("The first guess on the deviance function "
                             "returned a nan.  This could be a boundary "
                             " problem and should be reported.")

        # This special case is used to get the likelihood for a specific
        # params vector.

        for iteration in range(self.max_iter):
            weights = sample_weight * self.fam.weights(mu)
            wlsendog = lin_pred + self.fam.link.deriv(mu) * (Y-mu)
            if self.penalty is None:
                wls_results = lim.WLS(wlsendog, X, weights).fit(method=self.solver)

            if self.penalty == 'elasticnet':
                wls_results = lim.WLS(wlsendog, X, weights).fit_regularized(alpha=self.reg,
                                                                            L1_wt=self.l1_ratio)
            lin_pred = np.dot(X, wls_results.params)
            mu = self.fam.fitted(lin_pred)

            if Y.squeeze().ndim == 1 and np.allclose(mu - Y, 0):
                msg = "Perfect separation detected, results not available"
                raise PerfectSeparationError(msg)

            dev_new = self.fam.deviance_weighted(Y, mu, sample_weight)
            converged = np.fabs(dev - dev_new) <= self.tol
            dev = dev_new
            if converged:
                break

        self.converged = converged
        self.coef = wls_results.params
        self.dispersion = self.estimate_dispersion(X, Y, mu, sample_weight)
        if self.est_sd:
            self.sd = self.estimate_sd(X, Y, mu, sample_weight, weights)
        self.ll = self.estimate_loglikelihood(Y, mu, sample_weight)

    def predict(self, X):
        """
        predict the Y value based on the model
        ----------
        X : design matrix
        Returns
        -------
        predicted value
        """

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.fit_intercept:
            X = addIntercept(X)
        lin_pred = np.dot(X, self.coef)
        mu = self.fam.fitted(lin_pred)
        return mu

    def log_probability(self, X, Y):
        """
        Given a set of X and Y, calculate the probability of
        observing Y value
        """

        mu = self.predict(X)
        return self.fam.log_probability(Y.reshape(-1,), mu, scale=self.dispersion)

    def estimate_dispersion(self, X, Y, mu, w):
        if isinstance(self.fam, (family.Binomial, family.Poisson)):
            return 1
        else:
            resid = (Y - mu)
            return (resid ** 2 * w / self.fam.variance(mu)).sum() / np.sum(w)

    def estimate_sd(self, X, Y, mu, w, weights):
        if self.penalty is None and self.dispersion is not None:
            newX, newY = _rescale_data(X, Y, weights)
            wX, wY = _rescale_data(X, Y, w * weights)
            if X.shape[1] == 1:
                try:
                    cov = 1 / np.dot(newX.T, newX)
                    temp = np.dot(wX.T, wX)
                    sd = (np.sqrt(cov ** 2 * temp) * np.sqrt(self.dispersion)).reshape(-1,)
                except:
                    sd = None
            else:
                try:
                    cov = np.linalg.inv(np.dot(newX.T, newX))
                    temp = np.dot(cov, wX.T)
                    sd = np.sqrt(np.diag(np.dot(temp, temp.T))) * np.sqrt(self.dispersion)
                except:
                    sd = None
        else:
            sd = None
        return sd

    def estimate_loglikelihood(self, Y, mu, w):
        if self.dispersion is None:
            return None
        else:
            return self.fam.loglike_weighted(Y, mu, w, scale=self.dispersion)


class LM(BaseModel):
    """
    A Generalized linear model for data with input and output.
    """
    def __init__(self, solver='svd', fit_intercept=True, penalty=None, est_sd=False,
                 reg=0, l1_ratio=0, tol=1e-4, max_iter=100):
        super(LM, self).__init__(fam='LM', solver=solver, fit_intercept=fit_intercept,
                                 est_sd=est_sd, penalty=penalty,
                                 reg=reg, l1_ratio=l1_ratio, tol=tol, max_iter=max_iter)

    def fit(self, X, Y, sample_weight=None):
        """
        fit the weighted model
        Parameters
        ----------
        X : design matrix
        Y : response matrix
        sample_weight: sample weight vector

        """

        if sample_weight is None:
            sample_weight = np.ones((X.shape[0],))
        assert X.shape[0] == sample_weight.shape[0]
        assert X.shape[0] == Y.shape[0]

        sum_w = np.sum(sample_weight)
        assert sum_w > 0
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.fit_intercept:
            X = addIntercept(X)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.n_targets = Y.shape[1]
        newX, newY = _rescale_data(X, Y, sample_weight)
        if self.penalty is None:
            model = linear_model.LinearRegression(fit_intercept=False)
        if self.penalty == 'l1':
            model = linear_model.Lasso(fit_intercept=False, alpha=self.reg,
                                       tol=self.tol, max_iter=self.max_iter)
        if self.penalty == 'l2':
            model = linear_model.Ridge(fit_intercept=False, alpha=self.reg, tol=self.tol,
                                       max_iter=self.max_iter, solver=self.solver)
        if self.penalty == 'elasticnet':
            model = linear_model.ElasticNet(fit_intercept=False, alpha=self.reg,
                                            l1_ratio=self.l1_ratio, tol=self.tol,
                                            max_iter=self.max_iter)

        model.fit(newX, newY)
        self.coef = model.coef_.T
        if Y.shape[1] == 1:
            self.coef = self.coef.reshape(-1,)
        if self.penalty is not None:
            self.converged = model.n_iter_ < self.max_iter
        else:
            self.converged = None
        self.dispersion = self.estimate_dispersion(X, Y, sample_weight)
        if self.est_sd:
            self.sd = self.estimate_sd(X, Y, sample_weight)
        self.ll = self.estimate_loglikelihood(sample_weight)

    def predict(self, X):
        """
        predict the Y value based on the model
        ----------
        X : design matrix
        Returns
        -------
        predicted value
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.fit_intercept:
            X = addIntercept(X)
        mu = np.dot(X, self.coef)
        return mu

    def log_probability(self, X, Y):
        """
        Given a set of X and Y, calculate the probability of
        observing Y value
        """

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.fit_intercept:
            X = addIntercept(X)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        pred = np.dot(X, self.coef)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)

        if Y.shape[1] == 1:
            if self.dispersion > 0:
                logP = (Y * pred - pred**2/2)/self.dispersion - Y**2/(2 * self.dispersion) - \
                    .5*np.log(2 * np.pi * self.dispersion)
                logP = logP.reshape(-1,)
            else:
                logP = np.zeros((Y.shape[0],))
                logP[Y.reshape(-1,) != pred.reshape(-1, )] = -np.Infinity
                logP = logP.reshape(-1,)
        else:
            if np.linalg.det(self.dispersion) > 0:
                logP = -1/2*((Y.shape[1] * np.log(2 * np.pi) +
                             np.log(np.linalg.det(self.dispersion))) +
                             np.diag(np.dot(np.dot(Y - pred, np.linalg.inv(self.dispersion)),
                                                  (Y-pred).T)))
                logP = logP.reshape(-1,)
            else:
                if (np.diag(self.dispersion) > 0).all():
                    new_dispersion = np.diag(np.diag(self.dispersion))
                    logP = -1/2*((Y.shape[1] * np.log(2 * np.pi) +
                                 np.log(np.linalg.det(self.dispersion))) +
                                 np.diag(np.dot(np.dot(Y-pred, np.linalg.inv(new_dispersion)),
                                                      (Y-pred).T)))
                    logP = logP.reshape(-1,)

                else:
                    logP = np.zeros((Y.shape[0],))
                    logP[np.linalg.norm(Y-pred, axis=1) != 0] = -np.Infinity
                    logP = logP.reshape(-1,)
        return logP

    def estimate_dispersion(self, X, Y, sample_weight):
        newX, newY = _rescale_data(X, Y, sample_weight)
        newPred = np.dot(newX, self.coef)
        if newPred.ndim == 1:
            newPred = newPred.reshape(-1, 1)
        wresid = newY - newPred
        ssr = np.dot(wresid.T, wresid)
        sigma2 = ssr / np.sum(sample_weight)
        if sigma2.shape == (1, 1):
            sigma2 = sigma2[0, 0]
        return sigma2

    def estimate_sd(self, X, Y, sample_weight):
        newX, newY = _rescale_data(X, Y, sample_weight)
        if self.penalty is None:
            wX, wY = _rescale_data(X, Y, sample_weight ** 2)
            if newX.shape[1] == 1:
                try:
                    cov = 1 / np.dot(newX.T, newX)
                    temp = np.dot(wX.T, wX)
                    if newY.shape[1] == 1:
                        sd = np.sqrt(cov ** 2 * temp * self.dispersion).reshape(-1,)
                    else:
                        sd = np.sqrt(cov ** 2 * temp * np.diag(self.dispersion))
                except:
                    sd = None
            else:
                try:
                    cov = np.linalg.inv(np.dot(newX.T, newX))
                    temp = np.dot(cov, wX.T)
                    if newY.shape[1] == 1:
                        sd = np.sqrt(np.diag(np.dot(temp, temp.T)) * self.dispersion).reshape(-1,)
                    else:
                        sd = np.sqrt(np.outer(np.diag(np.dot(temp, temp.T)),
                                              np.diag(self.dispersion)))
                except:
                    sd = None
        else:
            sd = None
        return sd

    def estimate_loglikelihood(self, sample_weight):
        q = self.n_targets
        sum_w = np.sum(sample_weight)
        if q == 1:
            if self.dispersion > 0:
                ll = - q * sum_w / 2 * np.log(2 * np.pi) - \
                    sum_w / 2 * np.log(self.dispersion) - q * sum_w / 2
            else:
                ll = None
        else:
            if np.linalg.det(self.dispersion) > 0:
                ll = - q * sum_w / 2 * np.log(2 * np.pi) - \
                    sum_w / 2 * np.log(np.linalg.det(self.dispersion)) - q * sum_w / 2
            else:
                if (np.diag(self.dispersion) > 0).all():
                    ll = - q * sum_w / 2 * np.log(2 * np.pi) - \
                        np.sum(sum_w / 2 * np.log(np.diag(self.dispersion))) - q * sum_w / 2
                else:
                    ll = None
        return ll


class MNL(BaseModel):
    """
    A MNL for data with input and output.
    """
    def fit(self, X, Y, sample_weight=None):
        """
        fit the weighted model
        Parameters
        ----------
        X : design matrix
        Y : response matrix
        sample_weight: sample weight vector

        """
        raise NotImplementedError

    def predict_probability(self, X):
        """
        predict the Y value based on the model
        ----------
        X : design matrix
        Returns
        -------
        predicted value
        """
        return np.exp(self.predict_log_probability(X))

    def predict_log_probability(self, X):
        """
        predict the Y value based on the model
        ----------
        X : design matrix
        Returns
        -------
        predicted value
        """

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.fit_intercept:
            X = addIntercept(X)
        p = np.dot(X, self.coef)
        if p.ndim == 1:
            p = p.reshape(-1, 1)
        p -= logsumexp(p, axis=1)[:, np.newaxis]
        return p

    def predict(self, X):
        """
        predict the Y value based on the model
        ----------
        X : design matrix
        Returns
        -------
        predicted value
        """
        return NotImplementedError

    def log_probability(self, X, Y):
        """
        Given a set of X and Y, calculate the probability of
        observing Y value
        """
        return NotImplementedError

    def estimate_dispersion(self):
        return 1

    def estimate_sd(self, X, sample_weight):
        if self.penalty is None:
            o_normalized = np.dot(X, self.coef)
            if o_normalized.ndim == 1:
                o_normalized = o_normalized.reshape(-1, 1)
            o_normalized -= logsumexp(o_normalized, axis=1)[:, np.newaxis]
            o_normalized = np.exp(o_normalized)
            # calculate hessian
            p = self.n_features
            q = self.n_targets
            h = np.zeros((p*(q-1), p*(q-1)))
            for e in range(q-1):
                for f in range(q-1):
                    h[e*p: (e+1)*p, f*p: (f+1)*p] = -np.dot(np.dot(X.T, np.diag(
                        np.multiply(np.multiply(o_normalized[:, f+1],
                                                (e == f) - o_normalized[:, e+1]),
                                    sample_weight))), X)
            if np.sum(sample_weight) > 0:
                h = h / np.sum(sample_weight) * X.shape[0]
            if np.all(np.linalg.eigvals(-h) > 0) and np.linalg.cond(-h) < 1/sys.float_info.epsilon:
                sd = np.sqrt(np.diag(np.linalg.inv(-h))).reshape(p, q-1, order='F')
                sd = np.hstack((np.zeros((p, 1)), sd))
            else:
                sd = None
        else:
            sd = None
        return sd

    def estimate_loglikelihood(self, X, Y, sample_weight):
        return NotImplementedError


class MNLD(MNL):
    """
    A MNL for discrete data with input and output.
    """
    def __init__(self, solver='newton-cg', fit_intercept=True, est_sd=False, penalty=None,
                 reg=0, l1_ratio=0, tol=1e-4, max_iter=100):
        super(MNLD, self).__init__(fam='MNLD', solver=solver, fit_intercept=fit_intercept,
                                   est_sd=est_sd, penalty=penalty, reg=reg,
                                   l1_ratio=l1_ratio, tol=tol, max_iter=max_iter)

    def fit(self, X, Y, sample_weight=None):
        """
        fit the weighted model
        Parameters
        ----------
        X : design matrix
        Y : response matrix
        sample_weight: sample weight vector

        """

        if sample_weight is None:
            sample_weight = np.ones((X.shape[0],))
        assert Y.ndim == 1 or Y.shape[1] == 1
        assert X.shape[0] == Y.shape[0]
        assert X.shape[0] == sample_weight.shape[0]

        if self.reg == 0 or (self.penalty is None):
            penalty1 = 'l2'
            c = 1e200
        else:
            penalty1 = self.penalty
            c = 1/self.reg
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.fit_intercept:
            X = addIntercept(X)
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.n_targets = len(np.unique(Y))
        if self.n_targets < 2:
            raise ValueError('n_targets < 2')

        self.lb = LabelBinarizer().fit(Y)

        model = linear_model.LogisticRegression(fit_intercept=False, penalty=penalty1, C=c,
                                                multi_class='multinomial', solver=self.solver,
                                                tol=self.tol, max_iter=self.max_iter)

        model.fit(X, Y, sample_weight=sample_weight)
        w0 = model.coef_
        if self.n_targets == 2:
            w0 = np.vstack((np.zeros((1, self.n_features)), w0*2))
        w1 = w0.reshape(self.n_targets, -1)
        w1 = w1.T - w1.T[:, 0].reshape(-1, 1)
        self.coef = w1
        self.converged = model.n_iter_ < self.max_iter
        self.dispersion = self.estimate_dispersion()
        if self.est_sd:
            self.sd = self.estimate_sd(X, sample_weight)
        self.ll = self.estimate_loglikelihood(X, Y, sample_weight)

    def predict(self, X):
        """
        predict the Y value based on the model
        ----------
        X : design matrix
        Returns
        -------
        predicted value
        """
        index = np.argmax(self.predict_log_probability(X), axis=1)
        zero = np.zeros((X.shape[0], self.n_targets))
        zero[np.arange(X.shape[0]), index] = 1
        return self.lb.inverse_transform(zero)

    def log_probability(self, X, Y):
        """
        Given a set of X and Y, calculate the probability of
        observing Y value
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        assert Y.ndim == 1 or Y.shape[1] == 1

        assert X.shape[0] == Y.shape[0]
        p = self.predict_log_probability(X)
        Y_transformed = self.lb.transform(Y)
        if Y_transformed.shape[1] == 1:
            Y_aug = np.zeros((X.shape[0], 2))
            Y_aug[np.arange(X.shape[0]), Y_transformed.reshape(-1,)] = 1
        else:
            Y_aug = Y_transformed
        logP = np.sum(p*Y_aug, axis=1)

        return logP

    def estimate_loglikelihood(self, X, Y, sample_weight):
        o_normalized_log = np.dot(X, self.coef)
        if o_normalized_log.ndim == 1:
            o_normalized_log = o_normalized_log.reshape(-1, 1)
        o_normalized_log -= logsumexp(o_normalized_log, axis=1)[:, np.newaxis]
        Y_aug = self.lb.transform(Y)
        ll = (sample_weight[:, np.newaxis] * Y_aug * o_normalized_log).sum()
        return ll


class MNLP(MNL):
    """
    A MNL with probability response for data with input and output.
    """
    def __init__(self, solver='newton-cg', fit_intercept=True, est_sd=False, penalty=None,
                 reg=0, l1_ratio=0, tol=1e-4, max_iter=100):
        super(MNL, self).__init__(fam='MNLP', solver=solver, fit_intercept=fit_intercept,
                                  est_sd=est_sd, penalty=penalty, reg=reg,
                                  l1_ratio=l1_ratio, tol=tol, max_iter=max_iter)

    def fit(self, X, Y, sample_weight=None):
        """
        fit the weighted model
        Parameters
        ----------
        X : design matrix
        Y : response matrix
        sample_weight: sample weight vector

        """
        if sample_weight is None:
            sample_weight = np.ones((X.shape[0],))
        assert X.shape[0] == Y.shape[0]
        assert X.shape[0] == sample_weight.shape[0]
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.fit_intercept:
            X = addIntercept(X)
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.n_targets = Y.shape[1]
        if self.n_targets < 2:
            raise ValueError('n_targets < 2')
        w0 = np.zeros((self.n_targets*self.n_features, ))

        if self.solver == 'lbfgs':
            def func(x, *args):
                return _multinomial_loss_grad(x, *args)[0:2]
        else:
            def func(x, *args):
                return _multinomial_loss(x, *args)[0]

            def grad(x, *args):
                return _multinomial_loss_grad(x, *args)[1]
            hess = _multinomial_grad_hess

        if self.solver == 'lbfgs':
            try:
                w0, loss, info = optimize.fmin_l_bfgs_b(
                    func, w0, fprime=None,
                    args=(X, Y, self.reg, sample_weight),
                    iprint=0, pgtol=self.tol, maxiter=self.max_iter)
            except TypeError:
                # old scipy doesn't have maxiter
                w0, loss, info = optimize.fmin_l_bfgs_b(
                    func, w0, fprime=None,
                    args=(X, Y, self.reg, sample_weight),
                    iprint=0, pgtol=self.tol)
            if info["warnflag"] == 1:
                warnings.warn("lbfgs failed to converge. Increase the number "
                              "of iterations.")
            try:
                n_iter_i = info['nit'] - 1
            except:
                n_iter_i = info['funcalls'] - 1
        else:
            args = (X, Y, self.reg, sample_weight)
            w0, n_iter_i = newton_cg(hess, func, grad, w0, args=args,
                                     maxiter=self.max_iter, tol=self.tol)

        w1 = w0.reshape(self.n_targets, -1)
        w1 = w1.T - w1.T[:, 0].reshape(-1, 1)
        self.coef = w1
        self.converged = n_iter_i < self.max_iter
        self.dispersion = self.estimate_dispersion()
        if self.est_sd:
            self.sd = self.estimate_sd(X, sample_weight)
        self.ll = self.estimate_loglikelihood(X, Y, sample_weight)

    def predict(self, X):
        """
        predict the Y value based on the model
        ----------
        X : design matrix
        Returns
        -------
        predicted value
        """
        index = np.argmax(self.predict_log_probability(X), axis=1)
        return index

    def log_probability(self, X, Y):
        """
        Given a set of X and Y, calculate the probability of
        observing Y value
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        assert Y.ndim == 2

        assert X.shape[0] == Y.shape[0]
        p = self.predict_log_probability(X)
        logP = np.sum(p*Y, axis=1)
        return logP

    def estimate_loglikelihood(self, X, Y, sample_weight):
        o_normalized_log = np.dot(X, self.coef)
        if o_normalized_log.ndim == 1:
            o_normalized_log = o_normalized_log.reshape(-1, 1)
        o_normalized_log -= logsumexp(o_normalized_log, axis=1)[:, np.newaxis]
        ll = (sample_weight[:, np.newaxis] * Y * o_normalized_log).sum()
        return ll


def _multinomial_loss(w, X, Y, alpha, sample_weight):
    """Computes multinomial loss and class probabilities.
    Parameters
    ----------
    w : ndarray, shape (n_classes * n_features,) or
        (n_classes * (n_features + 1),)
        Coefficient vector.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.
    Y : ndarray, shape (n_samples, n_classes)
        Transformed labels according to the output of LabelBinarizer.
    alpha : float
        Regularization parameter. alpha is equal to 1 / C.
    sample_weight : array-like, shape (n_samples,) optional
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.
    Returns
    -------
    loss : float
        Multinomial loss.
    p : ndarray, shape (n_samples, n_classes)
        Estimated class probabilities.
    w : ndarray, shape (n_classes, n_features)
        Reshaped param vector excluding intercept terms.
    Reference
    ---------
    Bishop, C. M. (2006). Pattern recognition and machine learning.
    Springer. (Chapter 4.3.4)
    """
    n_classes = Y.shape[1]

    w = w.reshape(n_classes, -1)
    sample_weight = sample_weight[:, np.newaxis]

    p = np.dot(X, w.T)
    p -= logsumexp(p, axis=1)[:, np.newaxis]
    loss = -(sample_weight * Y * p).sum()
    loss += 0.5 * alpha * np.sum(w * w)
    p = np.exp(p, p)
    return loss, p, w


def _multinomial_loss_grad(w, X, Y, alpha, sample_weight):
    """Computes the multinomial loss, gradient and class probabilities.
    Parameters
    ----------
    w : ndarray, shape (n_classes * n_features,)
        Coefficient vector.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.
    Y : ndarray, shape (n_samples, n_classes)
        Transformed labels according to the output of LabelBinarizer.
    alpha : float
        Regularization parameter. alpha is equal to 1 / C.
    sample_weight : array-like, shape (n_samples,) optional
        Array of weights that are assigned to individual samples.
    Returns
    -------
    loss : float
        Multinomial loss.
    grad : ndarray, shape (n_classes * n_features,) or
        (n_classes * (n_features + 1),)
        Ravelled gradient of the multinomial loss.
    p : ndarray, shape (n_samples, n_classes)
        Estimated class probabilities
    Reference
    ---------
    Bishop, C. M. (2006). Pattern recognition and machine learning.
    Springer. (Chapter 4.3.4)
    """
    n_classes = Y.shape[1]
    n_features = X.shape[1]
    grad = np.zeros((n_classes, n_features))
    loss, p, w = _multinomial_loss(w, X, Y, alpha, sample_weight)
    sample_weight = sample_weight[:, np.newaxis]
    diff = sample_weight * (p - Y)
    grad[:, :n_features] = np.dot(diff.T, X)
    grad[:, :n_features] += alpha * w
    return loss, grad.ravel(), p


def _multinomial_grad_hess(w, X, Y, alpha, sample_weight):
    """
    Computes the gradient and the Hessian, in the case of a multinomial loss.
    Parameters
    ----------
    w : ndarray, shape (n_classes * n_features,)
        Coefficient vector.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.
    Y : ndarray, shape (n_samples, n_classes)
        Transformed labels according to the output of LabelBinarizer.
    alpha : float
        Regularization parameter. alpha is equal to 1 / C.
    sample_weight : array-like, shape (n_samples,) optional
        Array of weights that are assigned to individual samples.
    Returns
    -------
    grad : array, shape (n_classes * n_features,) or
        (n_classes * (n_features + 1),)
        Ravelled gradient of the multinomial loss.
    hessp : callable
        Function that takes in a vector input of shape (n_classes * n_features)
        or (n_classes * (n_features + 1)) and returns matrix-vector product
        with hessian.
    References
    ----------
    Barak A. Pearlmutter (1993). Fast Exact Multiplication by the Hessian.
        http://www.bcl.hamilton.ie/~barak/papers/nc-hessian.pdf
    """
    n_features = X.shape[1]
    n_classes = Y.shape[1]

    # `loss` is unused. Refactoring to avoid computing it does not
    # significantly speed up the computation and decreases readability
    loss, grad, p = _multinomial_loss_grad(w, X, Y, alpha, sample_weight)
    sample_weight = sample_weight[:, np.newaxis]

    # Hessian-vector product derived by applying the R-operator on the gradient
    # of the multinomial loss function.
    def hessp(v):
        v = v.reshape(n_classes, -1)
        # r_yhat holds the result of applying the R-operator on the multinomial
        # estimator.
        r_yhat = np.dot(X, v.T)
        r_yhat += (-p * r_yhat).sum(axis=1)[:, np.newaxis]
        r_yhat *= p
        r_yhat *= sample_weight
        hessProd = np.zeros((n_classes, n_features))
        hessProd[:, :n_features] = np.dot(r_yhat.T, X)
        hessProd[:, :n_features] += v * alpha
        return hessProd.ravel()

    return grad, hessp
