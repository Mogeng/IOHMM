'''
This is a unified interface of general/generalized linear models from
sklearn/statsmodels packages

Problems with sklearn:
1. No Generalized linear models available
2. Does not estimate standard deviation

Problems with statsmodels:
1. No working version of multivariate OLS with sample weights
2. MNLogit does not have sample weights

In this implementations,
we will mainly use statsmodels for
1. Generalized linear models with simple response
2. OLS (statsmodels 0.8.0 deprecated fit_regularized for WLS but I expect it to be back soon)

we will mainly use sklearn for
1. Multivariate OLS
2. Multinomial Logistic Regression with discrete output/probability outputs

'''

# //TODO check sum(sample_weight > 0)? Is it wrong to have the sample weight equal to zero?
# //TODO in future add arguments compatibility check

from __future__ import division

import cPickle as pickle
import logging
import numbers
import os


import numpy as np
from scipy.stats import multivariate_normal
from sklearn import linear_model
from sklearn.linear_model.base import _rescale_data
from sklearn.preprocessing import label_binarize
import statsmodels.api as sm
from statsmodels.tools import add_constant

EPS = np.finfo(float).eps


class BaseModel(object):
    """
    A generic supervised model for data with input and output.
    BaseModel does nothing, but lays out the methods expected of any subclass.
    """

    def __init__(self,
                 solver,
                 fit_intercept=True,
                 est_stderr=False,
                 tol=1e-4,
                 max_iter=100,
                 reg_method=None,
                 alpha=0,
                 l1_ratio=0,
                 coef=None,
                 stderr=None):
        """
        Constructor
        Parameters
        ----------
        solver: family specific solver
        fit_intercept: boolean indicating fit intercept or not
        est_stderr: boolean indicating calculte std.err of coefficients (usually expensive) or not
        tol: tolerence of fitting error
        max_iter: maximum iteraration of fitting
        reg_method: method to regularize the model, one of (None, elstic_net)
        alpha: regularization strength
        l1_ratio: if elastic_net, the l1 alpha ratio
        coef: the coefficients if loading from trained model
        stderr: the std.err of coefficients if loading from trained model
        -------
        """
        self.solver = solver
        self.fit_intercept = fit_intercept
        self.est_stderr = est_stderr
        self.tol = tol
        self.max_iter = max_iter
        self.reg_method = reg_method
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.coef = coef
        self.stderr = stderr

    def fit(self, X, Y, sample_weight=None):
        """
        fit the weighted model
        Parameters
        ----------
        X : design matrix, must be n*k, 2d
        Y : response matrix
        sample_weight: sample weight vector
        """

        def _estimate_dispersion(self):
            raise NotImplementedError

        def _estimate_stderr(self):
            raise NotImplementedError
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

    def loglike_per_sample(self, X, Y):
        """
        Given a set of X and Y, calculate the log probability of
        observing each of Y value given each X value

        should return a vector
        """
        return NotImplementedError

    def loglike(self, X, Y, sample_weight=None):
        if self.coef is None:
            logging.warning('No trained model, cannot calculate loglike.')
            return None
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        elif isinstance(sample_weight, numbers.Number):
            sample_weight = np.ones(X.shape[0]) * sample_weight
        return np.sum(sample_weight * self.loglike_per_sample(X, Y))

    def to_json(self, path):
        json_dict = {
            'data_type': self.__class__.__name__,
            'properties': {
                'solver': self.solver,
                'fit_intercept': self.fit_intercept,
                'est_stderr': self.est_stderr,
                'tol': self.tol,
                'max_iter': self.max_iter,
                'reg_method': self.reg_method,
                'alpha': self.alpha,
                'l1_ratio': self.l1_ratio,
                'coef': {
                    'data_type': 'numpy.ndarray',
                    'path': os.path.join(path, 'coef.npy')
                },
                'stderr': {
                    'data_type': 'numpy.ndarray',
                    'path': os.path.join(path, 'stderr.npy')
                }
            }
        }
        if not os.path.exists(os.path.dirname(json_dict['properties']['coef']['path'])):
            os.makedirs(os.path.dirname(json_dict['properties']['coef']['path']))
        np.save(json_dict['properties']['coef']['path'], self.coef)
        if not os.path.exists(os.path.dirname(json_dict['properties']['stderr']['path'])):
            os.makedirs(os.path.dirname(json_dict['properties']['stderr']['path']))
        np.save(json_dict['properties']['stderr']['path'], self.stderr)
        return json_dict

    @classmethod
    def _from_json(cls, json_dict, solver, fit_intercept, est_stderr,
                   tol, max_iter, reg_method, alpha, l1_ratio, coef, stderr):
        return cls(
            solver=solver, fit_intercept=fit_intercept, est_stderr=est_stderr,
            tol=tol, max_iter=max_iter,
            reg_method=reg_method, alpha=alpha, l1_ratio=l1_ratio,
            coef=coef, stderr=stderr)

    @classmethod
    def from_json(cls, json_dict):
        return cls._from_json(
            json_dict,
            solver=json_dict['properties']['solver'],
            fit_intercept=json_dict['properties']['fit_intercept'],
            est_stderr=json_dict['properties']['est_stderr'],
            tol=json_dict['properties']['tol'],
            max_iter=json_dict['properties']['max_iter'],
            reg_method=json_dict['properties']['reg_method'],
            alpha=json_dict['properties']['alpha'],
            l1_ratio=json_dict['properties']['l1_ratio'],
            coef=np.load(json_dict['properties']['coef']['path']),
            stderr=np.load(json_dict['properties']['stderr']['path']))


class GLM(BaseModel):
    """
    A Generalized linear model for data with input and output.
    """

    def __init__(self,
                 family,
                 solver='IRLS',
                 fit_intercept=True,
                 est_stderr=False,
                 tol=1e-4,
                 max_iter=100,
                 reg_method=None,
                 alpha=0,
                 l1_ratio=0,
                 coef=None,
                 stderr=None,
                 dispersion=None):
        """
        Constructor
        Parameters
        ----------
        family: family in the forwarding_family
        tol: tol in the optimization procedure
        max_iter: max_iter in the optimization procedure
        dispersion: dispersion/scale of the GLM
        -------
        """
        super(GLM, self).__init__(
            solver=solver, fit_intercept=fit_intercept, est_stderr=est_stderr,
            tol=tol, max_iter=max_iter,
            reg_method=reg_method, alpha=alpha, l1_ratio=l1_ratio,
            coef=coef, stderr=stderr)
        self.family = family
        self.dispersion = dispersion
        if self.coef is not None:
            dummy_X = dummy_Y = dummy_weight = np.zeros(1)
            self._model = sm.GLM(dummy_Y, dummy_X, family=self.family, freq_weights=dummy_weight)

    def fit(self, X, Y, sample_weight=None):
        """
        fit the weighted model
        Parameters
        ----------
        X : design matrix
        Y : response matrix
        sample_weight: sample weight vector
        """
        def _estimate_dispersion():
            # this is different from the implementations from statsmodels,
            # that we do not consider dof
            return self._model.scale

        def _estimate_stderr():
            # this is different from the implementations from statsmodels,
            # that we do not consider dof
            if self.reg_method is None or self.alpha < EPS:
                return fit_results.bse * np.sqrt(np.sum(sample_weight) / X.shape[0])
            return None

        if self.fit_intercept:
            X = add_constant(X, has_constant='add')
        if Y.ndim == 2 and Y.shape[1] == 1:
            Y = Y.reshape(-1,)
        assert Y.ndim == 1
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        elif isinstance(sample_weight, numbers.Number):
            sample_weight = np.ones(X.shape[0]) * sample_weight
        if np.sum(sample_weight) < EPS:
            logging.warning('Sum of sample weight is 0')
            return

        self._model = sm.GLM(Y, X, family=self.family, freq_weights=sample_weight)
        # dof in weighted regression does not make sense, hard code it to the total weights
        self._model.df_resid = np.sum(sample_weight)
        if self.reg_method is None or self.alpha < EPS:
            fit_results = self._model.fit(
                maxiter=self.max_iter, tol=self.tol, method=self.solver)
        else:
            fit_results = self._model.fit_regularized(
                method=self.reg_method, alpha=self.alpha,
                L1_wt=self.l1_ratio, maxiter=self.max_iter)
        self.coef = fit_results.params
        self.dispersion = _estimate_dispersion()
        if self.est_stderr:
            self.stderr = _estimate_stderr()

    def predict(self, X):
        """
        predict the Y value based on the model
        ----------
        X : design matrix
        Returns
        -------
        predicted value
        """
        if self.coef is None:
            logging.warning('No trained model, cannot predict.')
            return None
        if self.fit_intercept:
            X = add_constant(X, has_constant='add')
        return self._model.predict(self.coef, exog=X)

    def loglike_per_sample(self, X, Y):
        """
        Given a set of X and Y, calculate the probability of
        observing Y value
        """
        assert X.shape[0] == Y.shape[0]
        if self.coef is None:
            logging.warning('No trained model, cannot calculate loglike.')
            return None
        if Y.ndim == 2 and Y.shape[1] == 1:
            Y = Y.reshape(-1,)
        assert Y.ndim == 1
        mu = self.predict(X)
        return self.family.loglike_per_sample(Y, mu, scale=self.dispersion)

    def to_json(self, path):
        json_dict = super(GLM, self).to_json(path=path)
        json_dict['properties'].update(
            {
                'family': {
                    'data_type': self.family.__class__.__name__,
                    'path': os.path.join(path, 'family.p')
                },
                'dispersion': {
                    'data_type': 'numpy.ndarray',
                    'path': os.path.join(path, 'dispersion.npy')
                }
            })
        if not os.path.exists(os.path.dirname(json_dict['properties']['family']['path'])):
            os.makedirs(os.path.dirname(json_dict['properties']['family']['path']))
        pickle.dump(self.family, open(json_dict['properties']['family']['path'], 'wb'))
        if not os.path.exists(os.path.dirname(json_dict['properties']['dispersion']['path'])):
            os.makedirs(os.path.dirname(json_dict['properties']['dispersion']['path']))
        np.save(json_dict['properties']['dispersion']['path'], self.dispersion)
        return json_dict

    @classmethod
    def _from_json(cls, json_dict, solver, fit_intercept, est_stderr,
                   tol, max_iter, reg_method, alpha, l1_ratio, coef, stderr):
        return cls(
            solver=solver, fit_intercept=fit_intercept, est_stderr=est_stderr,
            reg_method=reg_method, alpha=alpha, l1_ratio=l1_ratio,
            coef=coef, stderr=stderr, tol=tol, max_iter=max_iter,
            family=pickle.load(open(json_dict['properties']['family']['path'])),
            dispersion=np.load(json_dict['properties']['dispersion']['path']))


class OLS(BaseModel):
    """
    A linear model for data with input and output.
    """

    def __init__(self, solver='svd', fit_intercept=True, est_stderr=False,
                 reg_method=None,  alpha=0, l1_ratio=0, tol=1e-4, max_iter=100,
                 coef=None, stderr=None,  dispersion=None, n_targets=None):
        super(OLS, self).__init__(
            solver=solver, fit_intercept=fit_intercept, est_stderr=est_stderr,
            tol=tol, max_iter=max_iter,
            reg_method=reg_method, alpha=alpha, l1_ratio=l1_ratio,
            coef=coef, stderr=stderr)
        self.dispersion = dispersion
        self.n_targets = n_targets
        self._pick_model()
        if self.coef is not None:
            self._model.coef_ = coef
            self._model.intercept_ = 0

    def _pick_model(self):
        if self.reg_method is None or self.alpha < EPS:
            self._model = linear_model.LinearRegression(
                fit_intercept=False)
        if self.reg_method == 'l1':
            self._model = linear_model.Lasso(
                fit_intercept=False, alpha=self.alpha,
                tol=self.tol, max_iter=self.max_iter)
        if self.reg_method == 'l2':
            self._model = linear_model.Ridge(
                fit_intercept=False, alpha=self.alpha, tol=self.tol,
                max_iter=self.max_iter, solver=self.solver)
        if self.reg_method == 'elastic_net':
            self._model = linear_model.ElasticNet(
                fit_intercept=False, alpha=self.alpha,
                l1_ratio=self.l1_ratio, tol=self.tol,
                max_iter=self.max_iter)

    def fit(self, X, Y, sample_weight=None):
        """
        fit the weighted model
        Parameters
        ----------
        X : design matrix
        Y : response matrix
        sample_weight: sample weight vector

        """
        def _estimate_dispersion():
            mu, wendog = _rescale_data(self.predict(X), Y, sample_weight)
            # this is due to sklearn inconsistency about lasso and linear regression, etc.
            wresid = mu - wendog
            return np.dot(wresid.T, wresid) / np.sum(sample_weight)

        def _estimate_stderr():
            # this is different from the implementations from statsmodels
            # that we do not consider dof
            # http://www.public.iastate.edu/~maitra/stat501/lectures/MultivariateRegression.pdf
            # https://stats.stackexchange.com/questions/52704/covariance-of-linear-
            # regression-coefficients-in-weighted-least-squares-method
            # http://pj.freefaculty.org/guides/stat/Regression/GLS/GLS-1-guide.pdf
            # https://stats.stackexchange.com/questions/27033/in-r-given-an-output-from-
            # optim-with-a-hessian-matrix-how-to-calculate-paramet
            # http://msekce.karlin.mff.cuni.cz/~vorisek/Seminar/0910l/jonas.pdf
            if self.reg_method is None or self.alpha < EPS:
                wexog, wendog = _rescale_data(X_train, Y, sample_weight)
                stderr = np.zeros((self.n_targets, X_train.shape[1]))
                try:
                    XWX_inverse_XW_sqrt = np.linalg.inv(np.dot(wexog.T, wexog)).dot(wexog.T)
                except:
                    return None
                sqrt_diag_XWX_inverse_XW_sqrt_W_XWX_inverse_XW_sqrt = np.sqrt(np.diag(
                    XWX_inverse_XW_sqrt.dot(np.diag(sample_weight)).dot(XWX_inverse_XW_sqrt.T)))
                for target in range(self.n_targets):
                    stderr[target, :] = (np.sqrt(self.dispersion[target, target]) *
                                         sqrt_diag_XWX_inverse_XW_sqrt_W_XWX_inverse_XW_sqrt)
                return stderr.reshape(self.coef.shape)
            return None

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        self.n_targets = Y.shape[1]

        if self.fit_intercept:
            X_train = add_constant(X, has_constant='add')
        else:
            X_train = X
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        elif isinstance(sample_weight, numbers.Number):
            sample_weight = np.ones(X.shape[0]) * sample_weight
        if np.sum(sample_weight) < EPS:
            logging.warning('Sum of sample weight is 0, not fitting model')
            return
        self._model.fit(X_train, Y, sample_weight)
        self.coef = self._model.coef_
        self.dispersion = _estimate_dispersion()
        if self.est_stderr:
            self.stderr = _estimate_stderr()

    def predict(self, X):
        """
        predict the Y value based on the model
        ----------
        X : design matrix
        Returns
        -------
        predicted value
        """
        if self.coef is None:
            logging.warning('No trained model, cannot predict.')
            return None
        if self.fit_intercept:
            X = add_constant(X, has_constant='add')
        return self._model.predict(X).reshape(-1, self.n_targets)

    def loglike_per_sample(self, X, Y):
        """
        Given a set of X and Y, calculate the probability of
        observing Y value
        """
        assert X.shape[0] == Y.shape[0]
        if self.coef is None:
            logging.warning('No trained model, cannot calculate loglike.')
            return None
        mu = self.predict(X)
        # https://stackoverflow.com/questions/13312498/how-to-find-degenerate-
        # rows-columns-in-a-covariance-matrix
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        zero_inds = np.where(np.diag(self.dispersion) < EPS)[0]
        log_p = np.zeros(Y.shape[0])
        log_p[~np.isclose(
            np.linalg.norm(
                Y[:, zero_inds] - mu[:, zero_inds], axis=1), 0)] = - np.Infinity
        non_zero_inds = np.setdiff1d(
            np.arange(Y.shape[1]), zero_inds, assume_unique=True)
        dispersion = self.dispersion[np.ix_(non_zero_inds, non_zero_inds)]
        if dispersion.shape[0] == 0:
            return log_p
        if np.linalg.det(dispersion) > EPS:
            # This is a harsh test, if the det is ensured to be > 0
            # all diagonal of dispersion will be > 0
            # for the zero parts:
            rv = multivariate_normal(cov=dispersion)
            log_p += rv.logpdf(Y[:, non_zero_inds] - mu[:, non_zero_inds])
            return log_p
        else:
            logging.error("""Dispersion matrix is singular, not able to calculate likelihood.
            Most like due to perfect correlations within dependent variables.
            Try another model specification.""")
            raise ValueError("""Dispersion matrix is singular, not able to calculate likelihood.
            Most like due to perfect correlations within dependent variables.
            Try another model specification.""")

    def to_json(self, path):
        json_dict = super(OLS, self).to_json(path=path)
        json_dict['properties'].update(
            {
                'dispersion': {
                    'data_type': 'numpy.ndarray',
                    'path': os.path.join(path, 'dispersion.npy')
                },
                'n_targets': self.n_targets
            })
        if not os.path.exists(os.path.dirname(json_dict['properties']['dispersion']['path'])):
            os.makedirs(os.path.dirname(json_dict['properties']['dispersion']['path']))
        np.save(json_dict['properties']['dispersion']['path'], self.dispersion)
        return json_dict

    @classmethod
    def _from_json(cls, json_dict, solver, fit_intercept, est_stderr,
                   tol, max_iter, reg_method, alpha, l1_ratio, coef, stderr):
        return cls(solver=solver, fit_intercept=fit_intercept, est_stderr=est_stderr,
                   reg_method=reg_method, alpha=alpha, l1_ratio=l1_ratio,
                   coef=coef, stderr=stderr,
                   tol=tol, max_iter=max_iter,
                   dispersion=np.load(json_dict['properties']['dispersion']['path']),
                   n_targets=json_dict['properties']['n_targets'])


class BaseMNL(BaseModel):
    """
    A MNL for data with input and output.
    """

    def __init__(self, solver='lbfgs', fit_intercept=True, est_stderr=False,
                 reg_method=None, alpha=0, l1_ratio=0,
                 tol=1e-4, max_iter=100,
                 coef=None, stderr=None,
                 classes=None, n_classes=None):
        super(BaseMNL, self).__init__(
            solver=solver, fit_intercept=fit_intercept, est_stderr=est_stderr,
            tol=tol, max_iter=max_iter,
            reg_method=reg_method, alpha=alpha, l1_ratio=l1_ratio,
            coef=coef, stderr=stderr)

        self.classes = classes
        self.n_classes = n_classes
        if self.coef is not None:
            if self.n_classes == 1:
                pass
            else:
                self._pick_model()
                self._model.coef_ = coef
                self._model.classes_ = classes
                self._model.intercept_ = 0

    def _pick_model(self):
        C = np.float64(1) / self.alpha
        if self.n_classes == 2:
            # perform logistic regression
            self._model = linear_model.LogisticRegression(
                fit_intercept=False, penalty=self.reg_method, C=C,
                solver=self.solver, tol=self.tol, max_iter=self.max_iter)

        else:
            # perform multinomial logistic regression
            self._model = linear_model.LogisticRegression(
                fit_intercept=False, penalty=self.reg_method, C=C,
                solver=self.solver, tol=self.tol, max_iter=self.max_iter,
                multi_class='multinomial')

    def fit(self, X, Y, sample_weight=None):
        """
        fit the weighted model
        Parameters
        ----------
        X : design matrix
        Y : response matrix
        sample_weight: sample weight vector

        """

        def _estimate_stderr():
            # http://mplab.ucsd.edu/tutorials/MultivariateLogisticRegression.pdf
            # https://github.com/cran/mlogit/blob/master/R/mlogit.methods.R
            # https://arxiv.org/pdf/1404.3177.pdf
            # https://stats.stackexchange.com/questions/283780/calculate-standard-
            # error-of-weighted-logistic-regression-coefficients
            # It seems that MNL with sample weights may not
            # be able to estimate stderr of coefficients
            # The reason is that
            # 1. the hessian is not scale-invariant to sample_weight
            # 2. There is no likelihood in weighted MNL
            # Two codes to calculate hessian:
            # 1. with sample weights:
            # https://github.com/scikit-learn/scikit-learn/
            # blob/ab93d657eb4268ac20c4db01c48065b5a1bfe80d/sklearn/linear_model/logistic.py
            # 2. without sample weights
            # http://www.statsmodels.org/dev/_modules/statsmodels/
            # discrete/discrete_model.html#MNLogit
            # now a placeholder
            return None

        if self.fit_intercept:
            X = add_constant(X, has_constant='add')
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        elif isinstance(sample_weight, numbers.Number):
            sample_weight = np.ones(X.shape[0]) * sample_weight
        if np.sum(sample_weight) < EPS:
            logging.warning('Sum of sample weight is 0, not fitting model')
            return

        X, Y, sample_weight = self._label_encoder(
            X, Y, sample_weight=sample_weight)
        assert Y.ndim == 1
        classes = np.unique(Y)
        self.n_classes = len(classes)

        # TODO
        if self.n_classes == 1:
            # no need to perform any model
            # self.coef is a all zeros array of shape (n_features,1)
            self.coef = np.zeros((X.shape[1], 1))
            self.classes = classes
        else:
            self._pick_model()
            self._model.fit(X, Y, sample_weight=sample_weight)
            # self.coef shape is wierd in sklearn, I will stick with it
            self.coef = self._model.coef_
            self.classes = self._model.classes_
            if self.est_stderr:
                self.stderr = _estimate_stderr()

    @staticmethod
    def _label_encoder(X, Y, sample_weight=None):
        return NotImplementedError

    def _label_decoder(self, Y):
        return NotImplementedError

    def predict_log_proba(self, X):
        """
        predict the Y value based on the model
        ----------
        X : design matrix
        Returns
        -------
        predicted value
        """
        if self.coef is None:
            logging.warning('No trained model, cannot predict.')
            return None
        if self.fit_intercept:
            X = add_constant(X, has_constant='add')
        if self.n_classes == 1:
            return np.zeros((X.shape[0], 1))

        return self._model.predict_log_proba(X)

    def predict(self, X):
        """
        predict the Y value based on the model
        ----------
        X : design matrix
        exclude_set : a set of excluded choices.
        Returns
        -------
        predicted value
        """
        if self.coef is None:
            logging.warning('No trained model, cannot predict.')
            return None
        if self.fit_intercept:
            X = add_constant(X, has_constant='add')
        if self.n_classes == 1:
            return self.classes[np.zeros(X.shape[0], dtype=np.int)]
        return self._model.predict(X)

    def loglike_per_sample(self, X, Y):
        """
        Given a set of X and Y, calculate the probability of
        observing Y value
        """
        assert X.shape[0] == Y.shape[0]
        if self.coef is None:
            logging.warning('No trained model, cannot calculate loglike.')
            return None
        Y = self._label_decoder(Y)
        assert X.shape[0] == Y.shape[0]
        assert Y.shape[1] == self.n_classes
        log_p = np.sum(self.predict_log_proba(X) * Y, axis=1)
        log_p[np.sum(Y, axis=1) < EPS] = -np.Infinity
        return log_p

    @classmethod
    def _from_json_MNL(cls, json_dict, solver, fit_intercept, est_stderr,
                       tol, max_iter, reg_method, alpha, l1_ratio, coef, stderr):
        return cls(
            solver=solver, fit_intercept=fit_intercept, est_stderr=est_stderr,
            reg_method=reg_method, alpha=alpha, l1_ratio=l1_ratio,
            coef=coef, stderr=stderr, tol=tol, max_iter=max_iter)

    @classmethod
    def _from_json(cls, json_dict, solver, fit_intercept, est_stderr,
                   tol, max_iter, reg_method, alpha, l1_ratio, coef, stderr):
        return cls._from_json_MNL(
            json_dict,
            solver=solver, fit_intercept=fit_intercept, est_stderr=est_stderr,
            reg_method=reg_method, alpha=alpha, l1_ratio=l1_ratio,
            coef=coef, stderr=stderr, tol=tol, max_iter=max_iter)


class DiscreteMNL(BaseMNL):
    """
    A MNL for discrete data with input and output.
    """

    def __init__(self, solver='lbfgs', fit_intercept=True, est_stderr=False,
                 reg_method=None, alpha=0, l1_ratio=0,
                 tol=1e-4, max_iter=100,
                 coef=None, stderr=None,
                 classes=None):
        n_classes = None if classes is None else classes.shape[0]
        super(DiscreteMNL, self).__init__(
            solver=solver, fit_intercept=fit_intercept, est_stderr=est_stderr,
            reg_method=reg_method, alpha=alpha, l1_ratio=l1_ratio,
            tol=tol, max_iter=max_iter,
            coef=coef, stderr=stderr,
            classes=classes, n_classes=n_classes)

    @staticmethod
    def _label_encoder(X, Y, sample_weight=None):
        if Y.ndim == 2 and Y.shape[1] == 1:
            Y = Y.reshape(-1,)
        return X, Y, sample_weight

    def _label_decoder(self, Y):
        # consider the case of outside labels
        if Y.ndim == 2 and Y.shape[1] == 1:
            Y = Y.reshape(-1,)
        assert Y.ndim == 1
        if self.n_classes == 1:
            return (Y == self.classes).reshape(-1, 1).astype(float)
        if self.n_classes == 2:
            # sklearn is stupid here
            label = np.zeros((Y.shape[0], self.n_classes))
            for clas_i, clas in enumerate(self.classes):
                label[:, clas_i] = (Y == clas).astype(float)
            return label
        return label_binarize(Y, self.classes)

    def to_json(self, path):
        json_dict = super(DiscreteMNL, self).to_json(path=path)
        json_dict['properties'].update(
            {
                'classes': {
                    'data_type': 'numpy.ndarray',
                    'path': os.path.join(path, 'classes.npy')
                }
            })
        if not os.path.exists(os.path.dirname(json_dict['properties']['classes']['path'])):
            os.makedirs(os.path.dirname(json_dict['properties']['classes']['path']))
        np.save(json_dict['properties']['classes']['path'], self.classes)
        return json_dict

    @classmethod
    def _from_json_MNL(cls, json_dict, solver, fit_intercept, est_stderr,
                       reg_method, alpha, l1_ratio, coef, stderr,
                       tol, max_iter):
        return cls(
            solver=solver, fit_intercept=fit_intercept, est_stderr=est_stderr,
            reg_method=reg_method, alpha=alpha, l1_ratio=l1_ratio,
            coef=coef, stderr=stderr,
            tol=tol, max_iter=max_iter,
            classes=np.load(json_dict['properties']['classes']['path']))


class CrossEntropyMNL(BaseMNL):
    """
    A MNL with probability response for data with input and output.
    """

    def __init__(self, solver='lbfgs', fit_intercept=True, est_stderr=False,
                 reg_method=None, alpha=0, l1_ratio=0,
                 tol=1e-4, max_iter=100,
                 coef=None, stderr=None,
                 n_classes=None):
        classes = None if n_classes is None else np.arange(n_classes)
        super(CrossEntropyMNL, self).__init__(
            solver=solver, fit_intercept=fit_intercept, est_stderr=est_stderr,
            reg_method=reg_method, alpha=alpha, l1_ratio=l1_ratio,
            tol=tol, max_iter=max_iter,
            coef=coef, stderr=stderr,
            classes=classes, n_classes=n_classes)

    @staticmethod
    def _label_encoder(X, Y, sample_weight=None):
        # idea from https://stats.stackexchange.com/questions/90622/
        # regression-model-where-output-is-a-probability
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        elif isinstance(sample_weight, numbers.Number):
            sample_weight = np.ones(X.shape[0]) * sample_weight
        if np.sum(sample_weight) < EPS:
            logging.warning('Sum of sample weight is 0, not fitting model')
            return
        n_samples, n_classes = X.shape[0], Y.shape[1]
        X_repeated = np.repeat(X, n_classes, axis=0)
        Y_repeated = np.tile(np.arange(n_classes), n_samples)
        sample_weight_repeated = Y.reshape(-1, ) * np.repeat(sample_weight, n_classes)
        return X_repeated, Y_repeated, sample_weight_repeated

    def _label_decoder(self, Y):
        """
        Given a set of X and Y, calculate the probability of
        observing Y value
        """
        assert Y.ndim == 2
        assert Y.shape[1] == self.n_classes
        return Y

    def to_json(self, path):
        json_dict = super(CrossEntropyMNL, self).to_json(path=path)
        json_dict['properties'].update(
            {
                'n_classes': self.n_classes
            })
        return json_dict

    @classmethod
    def _from_json_MNL(cls, json_dict, solver, fit_intercept, est_stderr,
                       reg_method, alpha, l1_ratio, coef, stderr,
                       tol, max_iter):
        return cls(
            solver=solver, fit_intercept=fit_intercept, est_stderr=est_stderr,
            reg_method=reg_method, alpha=alpha, l1_ratio=l1_ratio,
            coef=coef, stderr=stderr,
            tol=tol, max_iter=max_iter,
            n_classes=json_dict['properties']['n_classes'])
