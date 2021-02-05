'''
This is a unified interface/wrapper of general/generalized linear models from
sklearn/statsmodels packages.

Problems with sklearn:
1. No Generalized linear models available.
2. Does not estimate standard error of coefficients.
3. Logistic regression does not handle 1 class case.
4. For 2 class logistic regression, the 'ovr' result is not same as 'multinomial' result.

Problems with statsmodels:
1. No working version of multivariate OLS with sample weights.
2. MNLogit does not support sample weights.

Problem with both:
1. No interface to calculate loglike_per_sample,
   which is need to calculate emission probability in IOHMM.
2. No json-serialization.


In this implementations,
we will mainly use statsmodels for
1. Generalized linear models with simple response

we will mainly use sklearn for
1. Univariate/Multivariate Ordinary least square (OLS) models,
2. Multinomial Logistic Regression with discrete output/probability outputs

Note:
1. If using customized arguments for constructor, you may encounter compalints
   from the statsmodels/sklearn on imcompatible arguments.
   This maybe especially true for the compatibility between solver and regularization method.

2. For the GLM, statsmodels is not great when fitting with regularizations
   (espicially l1, and elstic_net). In this case the coefficients might be np.nan.
   Try not using regularizations if you select GLM until statsmodels is stable on this.
'''

# //TODO in future add arguments compatibility check

from __future__ import division

from future import standard_library

from builtins import range
from builtins import object
import pickle as pickle
import logging
import numbers
import os


import numpy as np
from scipy.stats import multivariate_normal
from sklearn import linear_model
from sklearn.linear_model._base import _rescale_data
from sklearn.preprocessing import label_binarize
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson, Binomial
from statsmodels.tools import add_constant
standard_library.install_aliases()
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
        solver: specific solver for each linear model
        fit_intercept: boolean indicating fit intercept or not
        est_stderr: boolean indicating calculte std.err of coefficients (usually expensive) or not
        tol: tolerence of fitting error
        max_iter: maximum iteraration of fitting
        reg_method: method to regularize the model, one of (None, l1, l2, elstic_net).
                    Need to be compatible with the solver.
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
        Fit the weighted model
        Parameters
        ----------
        X : design matrix of shape (n_samples, n_features), 2d
        Y : observed response matrix of shape
            (n_samples, ) or (n_samples, k) based on specific model
        sample_weight: sample weight vector of shape (n_samples, ), or float, or None
        """
        raise NotImplementedError

    def _raise_error_if_model_not_trained(self):
        """
        Raise error if the model is not trained (thus has coef)
        ----------
        """
        if self.coef is None:
            raise ValueError('Model is not trained.')

    def _raise_error_if_sample_weight_sum_zero(self, sample_weight):
        """
        Raise error if the sum of sample_weight is 0
        ----------
        sample_weight: array of (n_samples, )
        """
        if np.sum(sample_weight) < EPS:
            raise ValueError('Sum of sample weight is 0.')

    def _transform_X(self, X):
        """
        Transform the design matrix X
        ----------
        X : design matrix of shape (n_samples, n_features), 2d
        Returns
        -------
        X : design matrix of shape (n_samples, n_features + 1) if fit intercept
        """
        if self.fit_intercept:
            X = add_constant(X, has_constant='add')
        return X

    def _transform_sample_weight(self, X, sample_weight=None):
        """
        Transform the sample weight from anyform to array
        ----------
        X : design matrix of shape (n_samples, n_features), 2d
        sample_weight: sample weight vector of shape (n_samples, ), or float, or None
        Returns
        -------
        sample_weight: array of (n_samples, )
        """
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        elif isinstance(sample_weight, numbers.Number):
            sample_weight = np.ones(X.shape[0]) * sample_weight
        assert X.shape[0] == sample_weight.shape[0]
        return sample_weight

    def _transform_X_sample_weight(self, X, sample_weight=None):
        """
        Transform the design matrix X and sample_weight to the form they can be used to fit
        ----------
        X : design matrix of shape (n_samples, n_features), 2d
        sample_weight: sample weight vector of shape (n_samples, ), or float, or None
        Returns
        -------
        X : design matrix of shape (n_samples, n_features + 1) if fit intercept
        sample_weight: array of (n_samples, )
        """
        X = self._transform_X(X)
        sample_weight = self._transform_sample_weight(X, sample_weight=sample_weight)
        return X, sample_weight

    def predict(self, X):
        """
        Predict the Y value based on the model
        ----------
        X : design matrix of shape (n_samples, n_features), 2d
        Returns
        -------
        predicted value: of shape (n_samples, ) or (n_samples, k) based on specific model
        """
        raise NotImplementedError

    def loglike_per_sample(self, X, Y):
        """
        Given a set of X and Y, calculate the log probability of
        observing each of Y_i value given each X_i value
        Parameters
        ----------
        X : design matrix of shape (n_samples, n_features), 2d
        Y : observed response matrix of shape
            (n_samples, ) or (n_samples, k) based on specific model
        Returns
        -------
        log_p: array of shape (n_samples, )
        """
        raise NotImplementedError

    def loglike(self, X, Y, sample_weight=None):
        """
        Given a set of X and Y, calculate the log probability of
        observing Y, considering the sample weight.
        Parameters
        ----------
        X : design matrix of shape (n_samples, n_features), 2d
        Y : observed response matrix of shape
            (n_samples, ) or (n_samples, k) based on specific model
        Returns
        -------
        log_likelihood: float
        """
        self._raise_error_if_model_not_trained()
        sample_weight = self._transform_sample_weight(X, sample_weight=sample_weight)
        return np.sum(sample_weight * self.loglike_per_sample(X, Y))

    def to_json(self, path):
        """
        Generate json object of the model
        Parameters
        ----------
        path : the path to save the model
        Returns
        -------
        json_dict: a dictionary containing the attributes of the model
        """
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
        """
        Helper function to construct the linear model used by from_json.
        This function is designed to be override by subclasses.
        Parameters
        ----------
        json_dict : the dictionary that specifies the model
        solver: specific solver for each linear model
        fit_intercept: boolean indicating fit intercept or not
        est_stderr: boolean indicating calculte std.err of coefficients (usually expensive) or not
        tol: tolerence of fitting error
        max_iter: maximum iteraration of fitting
        reg_method: method to regularize the model, one of (None, l1, l2, elstic_net).
        alpha: regularization strength
        l1_ratio: if elastic_net, the l1 alpha ratio
        coef: the coefficients
        stderr: the std.err of coefficients
        Returns
        -------
        linear model object: a linear model object specified by the json_dict and other arguments
        """
        return cls(
            solver=solver, fit_intercept=fit_intercept, est_stderr=est_stderr,
            tol=tol, max_iter=max_iter,
            reg_method=reg_method, alpha=alpha, l1_ratio=l1_ratio,
            coef=coef, stderr=stderr)

    @classmethod
    def from_json(cls, json_dict):
        """
        Construct a linear model from a saved dictionary.
        This function is NOT designed to be override by subclasses.
        Parameters
        ----------
        json_dict: a json dictionary containing the attributes of the linear model.
        Returns
        -------
        linear model: a linear model object specified by the json_dict
        """
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
            coef=np.load(json_dict['properties']['coef']['path'], allow_pickle=True),
            stderr=np.load(json_dict['properties']['stderr']['path'], allow_pickle=True))


class GLM(BaseModel):
    """
    A wrapper for Generalized linear models.
    fit_regularized only support Poisson and Binomial due to statsmodels,
    and it is not stable. Try not using regularizations in GLM.
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
        solver: solver for GLM, default 'IRLS', otherwise will use gradient.
        fit_intercept: boolean indicating fit intercept or not
        est_stderr: boolean indicating calculte std.err of coefficients (usually expensive) or not
        tol: tolerence of fitting error
        max_iter: maximum iteraration of fitting
        reg_method: TRY NOT USING REGULARIZATIONS FOR GLM.
                    method to regularize the model, one of (None, elstic_net).
                    Need to be compatible with the solver.
        alpha: regularization strength
        l1_ratio: if elastic_net, the l1 alpha ratio
        coef: the coefficients if loading from trained model
        stderr: the std.err of coefficients if loading from trained model

        family: statsmodels.genmod.families.family.Family
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
            self._model = sm.GLM(dummy_Y, dummy_X, family=family,
                                 freq_weights=dummy_weight)

    def fit(self, X, Y, sample_weight=None):
        """
        Fit the weighted model
        Parameters
        ----------
        X : design matrix of shape (n_samples, n_features), 2d
        Y : response matrix of shape (n_samples, ) or (n_samples, k) depending on family
        sample_weight: sample weight vector of shape (n_samples, ), or float, or None
        """
        def _estimate_dispersion():
            """
            Estimate dispersion/scale based on the fitted model
            Returns
            -------
            dispersion: float
            """
            if isinstance(self.family, (Binomial, Poisson)):
                return 1.
            return self._model.scale

        def _estimate_stderr():
            """
            Estimate standard deviation of the coefficients.
            Returns
            -------
            standard deviation of the coefficients: array with the same shape as coef
            Notes
            -------
            I think the stderr of statsmodels is wrong.
            It uses the WLS stderr as the std err of GLM, which does not make sense,
            because the variance in WLS is inverse proportional to the weights.

            Anyway I will leave it here, stderr is not important.
            """
            if self.reg_method is None or self.alpha < EPS:
                return fit_results.bse * np.sqrt(self.dispersion / self._model.scale)
            return None

        X, sample_weight = self._transform_X_sample_weight(X, sample_weight=sample_weight)
        self._raise_error_if_sample_weight_sum_zero(sample_weight)
        Y = self._transform_Y(Y)
        self._model = sm.GLM(Y, X, family=self.family, freq_weights=sample_weight)
        # dof in weighted regression does not make sense, hard code it to the total weights
        self._model.df_resid = np.sum(sample_weight)
        if self.reg_method is None or self.alpha < EPS:
            fit_results = self._model.fit(
                maxiter=self.max_iter, tol=self.tol, method=self.solver, wls_method='pinv')
        else:
            fit_results = self._model.fit_regularized(
                method=self.reg_method, alpha=self.alpha,
                L1_wt=self.l1_ratio, maxiter=self.max_iter)
        self.coef = fit_results.params
        self.dispersion = _estimate_dispersion()
        if self.est_stderr:
            self.stderr = _estimate_stderr()

    def _transform_Y(self, Y):
        """
        Transform the response Y
        ----------
        Y : response matrix of shape (n_samples, ) or (n_samples, k) depending on family
        Returns
        -------
        Y : response matrix of shape (n_samples, ) or (n_samples, k) depending on family
        """
        if Y.ndim == 2 and Y.shape[1] == 1:
            Y = Y.reshape(-1,)
        return Y

    def predict(self, X):
        """
        Predict the Y value based on the model
        ----------
        X : design matrix of shape (n_samples, n_features), 2d
        Returns
        -------
        predicted value, of shape (n_samples, ), 1d
        """
        self._raise_error_if_model_not_trained()
        X = self._transform_X(X)
        return self._model.predict(self.coef, exog=X)

    def loglike_per_sample(self, X, Y):
        """
        Given a set of X and Y, calculate the log probability of
        observing each of Y value given each X value
        Parameters
        ----------
        X : design matrix of shape (n_samples, n_features), 2d
        Y : response matrix of shape (n_samples, ) or (n_samples, k) depending on family
        Returns
        -------
        log_p: array of shape (n_samples, )
        """
        self._raise_error_if_model_not_trained()
        assert X.shape[0] == Y.shape[0]
        Y = self._transform_Y(Y)
        mu = self.predict(X)
        if isinstance(self.family, Binomial):
            endog, _ = self.family.initialize(Y, 1.0)
        else:
            endog = Y
        if self.dispersion > EPS:
            return self.family.loglike_obs(endog, mu, scale=self.dispersion)
        log_p = np.zeros(endog.shape[0])
        log_p[~np.isclose(endog, mu)] = - np.Infinity
        return log_p

    def to_json(self, path):
        """
        Generate json object of the model
        Parameters
        ----------
        path : the path to save the model
        Returns
        -------
        json_dict: a dictionary containing the attributes of the GLM
        """
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
        """
        Helper function to construct the GLM used by from_json.
        This function overrides the parent class.
        Parameters
        ----------
        json_dict : the dictionary that specifies the model
        solver: specific solver for GLM
        fit_intercept: boolean indicating fit intercept or not
        est_stderr: boolean indicating calculte std.err of coefficients (usually expensive) or not
        tol: tolerence of fitting error
        max_iter: maximum iteraration of fitting
        reg_method: method to regularize the model, one of (None, elstic_net).
        alpha: regularization strength
        l1_ratio: if elastic_net, the l1 alpha ratio
        coef: the coefficients
        stderr: the std.err of coefficients
        Returns
        -------
        GLM object: a GLM object specified by the json_dict and other arguments
        """
        with open(json_dict['properties']['family']['path'], 'rb') as f:
            return cls(
                solver=solver, fit_intercept=fit_intercept, est_stderr=est_stderr,
                reg_method=reg_method, alpha=alpha, l1_ratio=l1_ratio,
                coef=coef, stderr=stderr, tol=tol, max_iter=max_iter,
                family=pickle.load(f),
                dispersion=np.load(json_dict['properties']['dispersion']['path'],
                                   allow_pickle=True))


class OLS(BaseModel):
    """
    A wrapper for Univariate and Multivariate Ordinary Least Squares (OLS).
    """

    def __init__(self, solver='svd', fit_intercept=True, est_stderr=False,
                 reg_method=None,  alpha=0, l1_ratio=0, tol=1e-4, max_iter=100,
                 coef=None, stderr=None,  dispersion=None, n_targets=None):
        """
        Constructor
        Parameters
        ----------
        solver: specific solver for OLS, default 'svd', possible solvers are:
                {'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag'}.
        fit_intercept: boolean indicating fit intercept or not
        est_stderr: boolean indicating calculte std.err of coefficients (usually expensive) or not
        tol: tolerence of fitting error
        max_iter: maximum iteraration of fitting
        reg_method: method to regularize the model, one of (None, l1, l2, elstic_net).
                    Need to be compatible with the solver.
        alpha: regularization strength
        l1_ratio: if elastic_net, the l1 alpha ratio
        coef: the coefficients if loading from trained model
        stderr: the std.err of coefficients if loading from trained model

        n_targets: the number of dependent variables
        dispersion: dispersion/scale mareix of the OLS
        -------
        """
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
        """
        Helper function to select a proper sklearn linear regression model
        based on the regulariztaion specified by the user.
        """
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
        Fit the weighted model
        Parameters
        ----------
        X : design matrix of shape (n_samples, n_features), 2d
        Y : response matrix of shape (n_samples, n_targets), 2d
        sample_weight: sample weight vector of shape (n_samples, ), or float, or None
        """
        def _estimate_dispersion():
            """
            Estimate dispersion matrix based on the fitted model
            Returns
            -------
            dispersion matrix: array of shape (n_targets, n_targets), 2d
            """
            mu, wendog = _rescale_data(self.predict(X), Y, sample_weight)
            wresid = mu - wendog
            return np.dot(wresid.T, wresid) / np.sum(sample_weight)

        def _estimate_stderr():
            """
            Estimate standard deviation of the coefficients.
            Returns
            -------
            standard deviation of the coefficients: array with the same shape as coef
            Notes
            -------
            It is not the same stderr as Weighted Least Squares (WLS).
            WLS assumes sample weight is inversely proportional to the covariance.
            Useful links:
            http://www.public.iastate.edu/~maitra/stat501/lectures/MultivariateRegression.pdf
            https://stats.stackexchange.com/questions/52704/covariance-of-linear-
            regression-coefficients-in-weighted-least-squares-method
            http://pj.freefaculty.org/guides/stat/Regression/GLS/GLS-1-guide.pdf
            https://stats.stackexchange.com/questions/27033/in-r-given-an-output-from-
            optim-with-a-hessian-matrix-how-to-calculate-paramet
            http://msekce.karlin.mff.cuni.cz/~vorisek/Seminar/0910l/jonas.pdf
            """
            if self.reg_method is None or self.alpha < EPS:
                wexog, wendog = _rescale_data(X_train, Y, sample_weight)
                stderr = np.zeros((self.n_targets, X_train.shape[1]))
                try:
                    XWX_inverse_XW_sqrt = np.linalg.inv(np.dot(wexog.T, wexog)).dot(wexog.T)
                except np.linalg.linalg.LinAlgError:
                    logging.warning('Covariance matrix is singular, cannot estimate stderr.')
                    return None
                sqrt_diag_XWX_inverse_XW_sqrt_W_XWX_inverse_XW_sqrt = np.sqrt(np.diag(
                    XWX_inverse_XW_sqrt.dot(np.diag(sample_weight)).dot(XWX_inverse_XW_sqrt.T)))
                for target in range(self.n_targets):
                    stderr[target, :] = (np.sqrt(self.dispersion[target, target]) *
                                         sqrt_diag_XWX_inverse_XW_sqrt_W_XWX_inverse_XW_sqrt)
                return stderr.reshape(self.coef.shape)
            return None

        X_train, sample_weight = self._transform_X_sample_weight(X, sample_weight=sample_weight)
        self._raise_error_if_sample_weight_sum_zero(sample_weight)
        Y = self._transform_Y(Y)
        self.n_targets = Y.shape[1]
        self._model.fit(X_train, Y, sample_weight)
        self.coef = self._model.coef_
        self.dispersion = _estimate_dispersion()
        if self.est_stderr:
            self.stderr = _estimate_stderr()

    def _transform_Y(self, Y):
        """
        Transform the response Y
        ----------
        Y : response matrix of shape (n_samples, ) or (n_samples, n_targets) depending on family
        Returns
        -------
        Y : response matrix of shape (n_samples, n_targets)
        """
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        return Y

    def predict(self, X):
        """
        Predict the Y value based on the model
        ----------
        X : design matrix of shape (n_samples, n_features), 2d
        Returns
        -------
        predicted value, of shape (n_samples, n_targets), 2d
        """
        self._raise_error_if_model_not_trained()
        X = self._transform_X(X)
        return self._model.predict(X).reshape(-1, self.n_targets)

    def loglike_per_sample(self, X, Y):
        """
        Given a set of X and Y, calculate the log probability of
        observing each of Y value given each X value
        Parameters
        ----------
        X : design matrix of shape (n_samples, n_features), 2d
        Y : observed response matrix of shape (n_samples, n_targets), 2d
        Returns
        -------
        log_p: array of shape (n_samples, )
        """
        self._raise_error_if_model_not_trained()
        assert X.shape[0] == Y.shape[0]
        mu = self.predict(X)
        # https://stackoverflow.com/questions/13312498/how-to-find-degenerate-
        # rows-columns-in-a-covariance-matrix
        Y = self._transform_Y(Y)
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
        if np.linalg.cond(dispersion) < 1 / EPS:
            # This is a harsh test, if the det is ensured to be > 0
            # all diagonal of dispersion will be > 0
            # for the zero parts:
            rv = multivariate_normal(cov=dispersion)
            log_p += rv.logpdf(Y[:, non_zero_inds] - mu[:, non_zero_inds])
            return log_p
        else:
            raise ValueError(
                """
                    Dispersion matrix is singular, cannot calculate likelike_per_sample.
                    Most like due to perfect correlations among dependent variables.
                    Try another model specification.
                """
            )

    def to_json(self, path):
        """
        Generate json object of the model
        Parameters
        ----------
        path : the path to save the model
        Returns
        -------
        json_dict: a dictionary containing the attributes of the OLS
        """
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
        """
        Helper function to construct the OLS used by from_json.
        This function overrides the parent class.
        Parameters
        ----------
        json_dict : the dictionary that specifies the model
        solver: specific solver for OLS
        fit_intercept: boolean indicating fit intercept or not
        est_stderr: boolean indicating calculte std.err of coefficients (usually expensive) or not
        tol: tolerence of fitting error
        max_iter: maximum iteraration of fitting
        reg_method: method to regularize the model, one of (None, l1, l2, elstic_net).
                    Need to be compatible with the solver.
        alpha: regularization strength
        l1_ratio: if elastic_net, the l1 alpha ratio
        coef: the coefficients
        stderr: the std.err of coefficients
        Returns
        -------
        OLS object: an OLS object specified by the json_dict and other arguments
        """
        return cls(solver=solver, fit_intercept=fit_intercept, est_stderr=est_stderr,
                   reg_method=reg_method, alpha=alpha, l1_ratio=l1_ratio,
                   coef=coef, stderr=stderr,
                   tol=tol, max_iter=max_iter,
                   dispersion=np.load(json_dict['properties']['dispersion']['path'],
                                      allow_pickle=True),
                   n_targets=json_dict['properties']['n_targets'])


class BaseMNL(BaseModel):
    """
    A Base Multinomial Logistic regression model.
    BaseMNL does nothing, to be extended by
    (1) MNL with discrete output (DiscreteMNL) and.
    (2) MNL with probability output (CrossEntropyMNL).
    """

    def __init__(self, solver='lbfgs', fit_intercept=True, est_stderr=False,
                 reg_method='l2', alpha=0, l1_ratio=0,
                 tol=1e-4, max_iter=100,
                 coef=None, stderr=None,
                 classes=None, n_classes=None):
        """
        Constructor
        Parameters
        ----------
        solver: specific solver for each linear model, default 'lbfgs',
                possible solvers are {'newton-cg', 'lbfgs', 'liblinear', 'sag'}.
                Need to be consistent with the regularization method.
        fit_intercept: boolean indicating fit intercept or not
        est_stderr: boolean indicating calculte std.err of coefficients (usually expensive) or not
        tol: tolerence of fitting error
        max_iter: maximum iteraration of fitting
        reg_method: method to regularize the model, one of (l1, l2).
                    Need to be compatible with the solver.
        alpha: regularization strength
        l1_ratio: the l1 alpha ratio
        coef: the coefficients if loading from trained model
        stderr: the std.err of coefficients if loading from trained model

        classes: an array of class labels
        n_classes: the number of classes to be classified
        -------
        """
        super(BaseMNL, self).__init__(
            solver=solver, fit_intercept=fit_intercept, est_stderr=est_stderr,
            tol=tol, max_iter=max_iter,
            reg_method=reg_method, alpha=alpha, l1_ratio=l1_ratio,
            coef=coef, stderr=stderr)

        self.classes = classes
        self.n_classes = n_classes
        if self.coef is not None:
            if self.n_classes >= 2:
                self._pick_model()
                self._model.coef_ = coef
                self._model.classes_ = classes
                self._model.intercept_ = 0

    def _pick_model(self):
        """
        Helper function to select a proper sklearn logistic regression model
        based on the regulariztaion specified by the user.
        """
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
        Fit the weighted model
        Parameters
        ----------
        X : design matrix of shape (n_samples, n_features), 2d
        Y : response matrix of shape (n_samples, ) for DiscreteMNL and
            (n_samples, n_classes) for CrossEntropyMNL
        sample_weight: sample weight vector of shape (n_samples, ), or float, or None
        """
        def _estimate_stderr():
            """
            Estimate standard deviation of the coefficients.
            Returns
            -------
            None for now, since I am not sure if we can estimate the stderr
            under the case there is sample_weight since there is no likelihood,
            thus no hessian of the log likelihood.
            Notes
            -------
            http://mplab.ucsd.edu/tutorials/MultivariateLogisticRegression.pdf
            https://github.com/cran/mlogit/blob/master/R/mlogit.methods.R
            https://arxiv.org/pdf/1404.3177.pdf
            https://stats.stackexchange.com/questions/283780/calculate-standard-
            error-of-weighted-logistic-regression-coefficients

            Two codes to calculate hessian:
            1. with sample weights:
            https://github.com/scikit-learn/scikit-learn/
            blob/ab93d657eb4268ac20c4db01c48065b5a1bfe80d/sklearn/linear_model/logistic.py
            2. without sample weights
            http://www.statsmodels.org/dev/_modules/statsmodels/
            discrete/discrete_model.html#MNLogit
            """
            return None

        X, sample_weight = self._transform_X_sample_weight(X, sample_weight=sample_weight)
        self._raise_error_if_sample_weight_sum_zero(sample_weight)
        X, Y, sample_weight = self._label_encoder(
            X, Y, sample_weight)
        assert Y.ndim == 1
        classes = np.unique(Y)
        self.n_classes = len(classes)

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
    def _label_encoder(X, Y, sample_weight):
        """
        Convert input to proper format to be used by sklearn logistic regression.
        Mainly transforms Y to a 1d vector containing the class label for each sample.
        This function is designed to be override by subclasses.
        Parameters
        ----------
        X : design matrix of shape (n_samples, n_features), 2d
        Y : response matrix of shape (n_samples, ) for DiscreteMNL and
            (n_samples, n_classes) for CrossEntropyMNL
        sample_weight: sample weight vector of shape (n_samples, )
        Returns
        -------
        X_transformed : design matrix of shape (n, n_features), 2d
        Y_transformed : response matrix of shape (n, )
        sample_weight_transformed: sample weight vector of shape (n, )
        where n:
        is n_samples in the discrete case and
        is n_samples * n_classes in the cross entropy case
        """
        raise NotImplementedError

    def _label_decoder(self, Y):
        """
        Convert the response vector to probability matrix.
        This function is designed to be override by subclasses.
        Parameters
        ----------
        Y : response matrix of shape (n_samples, ) for DiscreteMNL and
            (n_samples, n_classes) for CrossEntropyMNL
        Returns
        -------
        Y_transformed : of shape (n_samples, n_classes).
        """
        raise NotImplementedError

    def predict_log_proba(self, X):
        """
        Predict the log probability of each class
        ----------
        X : design matrix of shape (n_samples, n_features), 2d
        Returns
        -------
        log probability matrix : of shape (n_samples, n_classes), 2d
        """
        self._raise_error_if_model_not_trained()
        X = self._transform_X(X)
        if self.n_classes == 1:
            return np.zeros((X.shape[0], 1))

        return self._model.predict_log_proba(X)

    def predict(self, X):
        """
        Predict the most likely class label for each sample
        ----------
        X : design matrix of shape (n_samples, n_features), 2d
        Returns
        -------
        labels : of shape (n_samples, ), 1d
        """
        self._raise_error_if_model_not_trained()
        X = self._transform_X(X)
        if self.n_classes == 1:
            return self.classes[np.zeros(X.shape[0], dtype=np.int)]
        return self._model.predict(X)

    def loglike_per_sample(self, X, Y):
        """
        Given a set of X and Y, calculate the log probability of
        observing each of Y value given each X value
        Parameters
        ----------
        X : design matrix of shape (n_samples, n_features), 2d
        Y : response matrix of shape (n_samples, ) for DiscreteMNL and
            (n_samples, n_classes) for CrossEntropyMNL
        Returns
        -------
        log_p: array of shape (n_samples, )
        """
        self._raise_error_if_model_not_trained()
        assert X.shape[0] == Y.shape[0]
        Y = self._label_decoder(Y)
        assert X.shape[0] == Y.shape[0]
        assert Y.shape[1] == self.n_classes
        log_p = np.sum(self.predict_log_proba(X) * Y, axis=1)
        log_p[np.sum(Y, axis=1) < EPS] = -np.Infinity
        return log_p

    @classmethod
    def _from_json_MNL(cls, json_dict, solver, fit_intercept, est_stderr,
                       tol, max_iter, reg_method, alpha, l1_ratio, coef, stderr):
        """
        Helper function within the BaseMNL class to construct the specific MNL used by _from_json.
        This function is designed to be override by subsubclasses.
        Parameters
        ----------
        json_dict : the dictionary that specifies the model
        solver: specific solver for each MNL
        fit_intercept: boolean indicating fit intercept or not
        est_stderr: boolean indicating calculte std.err of coefficients (usually expensive) or not
        tol: tolerence of fitting error
        max_iter: maximum iteraration of fitting
        reg_method: method to regularize the model, one of (l1, l2).
                    Need to be compatible with the solver.
        alpha: regularization strength
        l1_ratio: the l1 alpha ratio
        coef: the coefficients
        stderr: the std.err of coefficients
        Returns
        -------
        Discrete/CrossEntropyMNL object: a MNL object specified by the json_dict and other arguments
        """
        return cls(
            solver=solver, fit_intercept=fit_intercept, est_stderr=est_stderr,
            reg_method=reg_method, alpha=alpha, l1_ratio=l1_ratio,
            coef=coef, stderr=stderr, tol=tol, max_iter=max_iter)

    @classmethod
    def _from_json(cls, json_dict, solver, fit_intercept, est_stderr,
                   tol, max_iter, reg_method, alpha, l1_ratio, coef, stderr):
        """
        Helper function to construct the linear model used by from_json.
        This function overrides the parent class.
        Parameters
        ----------
        json_dict : the dictionary that specifies the model
        solver: specific solver for each MNL
        fit_intercept: boolean indicating fit intercept or not
        est_stderr: boolean indicating calculte std.err of coefficients (usually expensive) or not
        tol: tolerence of fitting error
        max_iter: maximum iteraration of fitting
        reg_method: method to regularize the model, one of (l1, l2).
                    Need to be compatible with the solver.
        alpha: regularization strength
        l1_ratio: the l1 alpha ratio
        coef: the coefficients
        stderr: the std.err of coefficients
        Returns
        -------
        Discrete/CrossEntropyMNL object: a MNL object specified by the json_dict and other arguments
        """
        return cls._from_json_MNL(
            json_dict,
            solver=solver, fit_intercept=fit_intercept, est_stderr=est_stderr,
            reg_method=reg_method, alpha=alpha, l1_ratio=l1_ratio,
            coef=coef, stderr=stderr, tol=tol, max_iter=max_iter)


class DiscreteMNL(BaseMNL):
    """
    A MNL for the case where responses are discrete labels.
    """

    def __init__(self, solver='lbfgs', fit_intercept=True, est_stderr=False,
                 reg_method='l2', alpha=0, l1_ratio=0,
                 tol=1e-4, max_iter=100,
                 coef=None, stderr=None,
                 classes=None):
        """
        Constructor
        Parameters
        ----------
        solver: specific solver for each linear model, default 'lbfgs',
                possible solvers are {'newton-cg', 'lbfgs', 'liblinear', 'sag'}.
                Need to be consistent with the regularization method.
        fit_intercept: boolean indicating fit intercept or not
        est_stderr: boolean indicating calculte std.err of coefficients (usually expensive) or not
        tol: tolerence of fitting error
        max_iter: maximum iteraration of fitting
        reg_method: method to regularize the model, one of (l1, l2).
                    Need to be compatible with the solver.
        alpha: regularization strength
        l1_ratio: the l1 alpha ratio
        coef: the coefficients if loading from trained model
        stderr: the std.err of coefficients if loading from trained model

        classes: class labels if loading from trained model
        -------
        """
        n_classes = None if classes is None else classes.shape[0]
        super(DiscreteMNL, self).__init__(
            solver=solver, fit_intercept=fit_intercept, est_stderr=est_stderr,
            reg_method=reg_method, alpha=alpha, l1_ratio=l1_ratio,
            tol=tol, max_iter=max_iter,
            coef=coef, stderr=stderr,
            classes=classes, n_classes=n_classes)

    @staticmethod
    def _label_encoder(X, Y, sample_weight):
        """
        Convert input to proper format to be used by sklearn logistic regression.
        Basically do nothing for the discrete case.
        This function overrides parent class.
        Parameters
        ----------
        X : design matrix of shape (n_samples, n_features), 2d
        Y : response matrix of shape (n_samples, )
        sample_weight: sample weight vector of shape (n_samples, )
        Returns
        -------
        X_transformed : design matrix of shape (n_samples, n_features), 2d
        Y_transformed : response matrix of shape (n_samples, )
        sample_weight_transformed: sample weight vector of shape (n_samples, )
        """
        if Y.ndim == 2 and Y.shape[1] == 1:
            Y = Y.reshape(-1,)
        return X, Y, sample_weight

    def _label_decoder(self, Y):
        """
        Convert the response vector to probability matrix.
        This function overrides parent classes.
        Parameters
        ----------
        Y : response matrix of shape (n_samples, )
        Returns
        -------
        Y_transformed : of shape (n_samples, n_classes).
        """
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
        """
        Generate json object of the model
        Parameters
        ----------
        path : the path to save the model
        Returns
        -------
        json_dict: a dictionary containing the attributes of the DiscreteMNL
        """
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
        """
        Helper function within the construct the DiscreteMNL used by _from_json.
        This function overrides parent class.
        Parameters
        ----------
        json_dict : the dictionary that specifies the model
        solver: specific solver for each linear model
        fit_intercept: boolean indicating fit intercept or not
        est_stderr: boolean indicating calculte std.err of coefficients (usually expensive) or not
        tol: tolerence of fitting error
        max_iter: maximum iteraration of fitting
        reg_method: method to regularize the model, one of (None, l1, l2, elstic_net).
                    Need to be compatible with the solver.
        alpha: regularization strength
        l1_ratio: the l1 alpha ratio
        coef: the coefficients
        stderr: the std.err of coefficients
        Returns
        -------
        DiscreteMNL object: a DiscreteMNL object specified by the json_dict and other arguments
        """
        return cls(
            solver=solver, fit_intercept=fit_intercept, est_stderr=est_stderr,
            reg_method=reg_method, alpha=alpha, l1_ratio=l1_ratio,
            coef=coef, stderr=stderr,
            tol=tol, max_iter=max_iter,
            classes=np.load(json_dict['properties']['classes']['path'], allow_pickle=True))


class CrossEntropyMNL(BaseMNL):
    """
    A MNL for the case where responses are probabilities sum to one.
    """

    def __init__(self, solver='lbfgs', fit_intercept=True, est_stderr=False,
                 reg_method='l2', alpha=0, l1_ratio=0,
                 tol=1e-4, max_iter=100,
                 coef=None, stderr=None,
                 n_classes=None):
        """
        Constructor
        Parameters
        ----------
        solver: specific solver for each linear model, default 'lbfgs',
                possible solvers are {'newton-cg', 'lbfgs', 'liblinear', 'sag'}.
                Need to be consistent with the regularization method.
        fit_intercept: boolean indicating fit intercept or not
        est_stderr: boolean indicating calculte std.err of coefficients (usually expensive) or not
        tol: tolerence of fitting error
        max_iter: maximum iteraration of fitting
        reg_method: method to regularize the model, one of (l1, l2).
                    Need to be compatible with the solver.
        alpha: regularization strength
        l1_ratio: the l1 alpha ratio
        coef: the coefficients if loading from trained model
        stderr: the std.err of coefficients if loading from trained model

        n_classes: number of classes to be classified
        -------
        """
        classes = None if n_classes is None else np.arange(n_classes)
        super(CrossEntropyMNL, self).__init__(
            solver=solver, fit_intercept=fit_intercept, est_stderr=est_stderr,
            reg_method=reg_method, alpha=alpha, l1_ratio=l1_ratio,
            tol=tol, max_iter=max_iter,
            coef=coef, stderr=stderr,
            classes=classes, n_classes=n_classes)

    @staticmethod
    def _label_encoder(X, Y, sample_weight):
        """
        Convert input to proper format to be used by sklearn logistic regression.
        Mainly transforms Y to a 1d vector containing the class label for each sample.
        This function overrides parent class.
        Parameters
        ----------
        X : design matrix of shape (n_samples, n_features), 2d
        Y : response matrix of shape (n_samples, n_classes)
        sample_weight: sample weight vector of shape (n_samples, )
        Returns
        -------
        X_repeated : design matrix of shape (n_samples * n_classes, n_features), 2d
        Y_repeated : response matrix of shape (n_samples * n_classes, )
        sample_weight_repeated: sample weight vector of shape (n_samples * n_classes, )
        Notes
        ----------
        idea from https://stats.stackexchange.com/questions/90622/
        regression-model-where-output-is-a-probability
        """
        n_samples, n_classes = X.shape[0], Y.shape[1]
        X_repeated = np.repeat(X, n_classes, axis=0)
        Y_repeated = np.tile(np.arange(n_classes), n_samples)
        sample_weight_repeated = Y.reshape(-1, ) * np.repeat(sample_weight, n_classes)
        return X_repeated, Y_repeated, sample_weight_repeated

    def _label_decoder(self, Y):
        """
        Convert the response vector to probability matrix.
        In CrossEntropyMNL, this function basically does nothing.
        This function overrides parent classes.
        Parameters
        ----------
        Y : response matrix of shape (n_samples, n_classes)
        Returns
        -------
        Y_transformed : of shape (n_samples, n_classes).
        """
        assert Y.ndim == 2
        assert Y.shape[1] == self.n_classes
        return Y

    def to_json(self, path):
        """
        Generate json object of the model
        Parameters
        ----------
        path : the path to save the model
        Returns
        -------
        json_dict: a dictionary containing the attributes of the CrossEntropyMNL
        """
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
        """
        Helper function within the construct the CrossEntropyMNL used by _from_json.
        This function overrides parent class.
        Parameters
        ----------
        json_dict : the dictionary that specifies the model
        solver: specific solver for each linear model
        fit_intercept: boolean indicating fit intercept or not
        est_stderr: boolean indicating calculte std.err of coefficients (usually expensive) or not
        tol: tolerence of fitting error
        max_iter: maximum iteraration of fitting
        reg_method: method to regularize the model, one of (l1, l2).
                    Need to be compatible with the solver.
        alpha: regularization strength
        l1_ratio: the l1 alpha ratio
        coef: the coefficients
        stderr: the std.err of coefficients
        Returns
        -------
        CrossEntropyMNL object:
            a CrossEntropyMNL object specified by the json_dict and other arguments
        """
        return cls(
            solver=solver, fit_intercept=fit_intercept, est_stderr=est_stderr,
            reg_method=reg_method, alpha=alpha, l1_ratio=l1_ratio,
            coef=coef, stderr=stderr,
            tol=tol, max_iter=max_iter,
            n_classes=json_dict['properties']['n_classes'])
