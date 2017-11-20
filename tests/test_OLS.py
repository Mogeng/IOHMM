from __future__ import print_function
from __future__ import division
# import json
from past.utils import old_div
import unittest


import numpy as np
import statsmodels.api as sm


from IOHMM import OLS

# //TODO sample weight all zero

# Corner cases
# General
# 1. sample_weight is all zero
# 2. sample_weight is all one
# 3. sample_weight is a scale of all one
# 4. sample_weight is mixed of 0 and 1
# 6. when number of data is 1/or very small, less than the number of features
# 7. standard dataset compare with sklearn/statsmodels
# 8. output dimensions
# 9. collinearty in X
# 10. to/from json
# MultivariateOLS
# 1. Y is not column/row independent
# Discrete/CrossEntropyMNL
# 1. number of class is 1
# 2. number of class is 2


class UnivariateOLSTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_longley = sm.datasets.longley.load()

    def test_ols(self):
        self.model = OLS(
            solver='pinv', fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.data_longley.exog, self.data_longley.endog)
        # coefficient
        self.assertEqual(self.model.coef.shape, (1, 7))
        np.testing.assert_array_almost_equal(
            self.model.coef,
            np.array([-3482258.63459582, 15.0618722713733, -0.358191792925910E-01,
                      -2.02022980381683, -1.03322686717359, -0.511041056535807E-01,
                      1829.15146461355]).reshape(1, -1),
            decimal=3)
        # std.err of coefficient (calibrated by df_resid)
        self.assertEqual(self.model.stderr.shape, (1, 7))
        np.testing.assert_array_almost_equal(
            old_div(self.model.stderr, np.sqrt(old_div(9., self.data_longley.exog.shape[0]))),
            np.array([890420.383607373, 84.9149257747669, 0.03349,
                      0.488399681651699, 0.214274163161675, 0.226073200069370,
                      455.478499142212]).reshape(1, -1),
            decimal=2)
        # scale
        self.assertEqual(self.model.dispersion.shape, (1, 1))
        np.testing.assert_array_almost_equal(
            old_div(self.model.dispersion, (old_div(9., self.data_longley.exog.shape[0]))),
            np.array([[92936.0061673238]]),
            decimal=3)
        # predict
        np.testing.assert_array_almost_equal(
            self.data_longley.endog.reshape(-1, 1) - self.model.predict(self.data_longley.exog),
            np.array([267.34003, -94.01394, 46.28717, -410.11462,
                      309.71459, -249.31122, -164.04896, -13.18036, 14.30477, 455.39409,
                      -17.26893, -39.05504, -155.54997, -85.67131, 341.93151,
                      -206.75783]).reshape(-1, 1),
            decimal=3)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.data_longley.exog, self.data_longley.endog),
            -109.61743480849013,
            places=3)

        # to_json
        json_dict = self.model.to_json('./tests/linear_models/OLS/UnivariateOLS/')
        self.assertEqual(json_dict['properties']['solver'], 'pinv')

        # from_json
        self.model_from_json = OLS.from_json(json_dict)
        np.testing.assert_array_almost_equal(
            self.model.coef,
            self.model_from_json.coef,
            decimal=3)
        np.testing.assert_array_almost_equal(
            self.model.stderr,
            self.model_from_json.stderr,
            decimal=3)
        self.assertEqual(
            self.model.dispersion,
            self.model_from_json.dispersion)

    def test_ols_l1_regularized(self):
        # sklearn elastic net and l1 does not take sample_weights, will not test
        pass

    def test_ols_l2_regularized(self):
        # there is a bug in sklearn with weights, it can only use list right now
        self.model = OLS(
            solver='auto', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0.1, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.data_longley.exog, self.data_longley.endog, sample_weight=0.5)

        # coefficient
        print(self.model.coef)
        print(self.model.dispersion)
        print(self.data_longley.endog.reshape(-1, 1) - self.model.predict(self.data_longley.exog))
        print(self.model.loglike(self.data_longley.exog, self.data_longley.endog))
        np.testing.assert_array_almost_equal(
            self.model.coef,
            np.array([-2.0172203, -52.14364269, 0.07089677, -0.42552125,
                      -0.57305292, -0.41272483, 48.32484052]).reshape(1, -1),
            decimal=3)
        # std.err of coefficient (calibrated by df_resid)
        self.assertTrue(self.model.stderr is None)
        # scale
        self.assertEqual(self.model.dispersion.shape, (1, 1))
        np.testing.assert_array_almost_equal(
            old_div(self.model.dispersion, (old_div(9., self.data_longley.exog.shape[0]))),
            np.array([[250870.081]]),
            decimal=3)
        # predict
        np.testing.assert_array_almost_equal(
            self.data_longley.endog.reshape(-1, 1) - self.model.predict(self.data_longley.exog),
            np.array([[280.31871146],
                      [-131.6981265],
                      [90.64414685],
                      [-400.10244445],
                      [-440.59604167],
                      [-543.88595187],
                      [200.70483416],
                      [215.88629903],
                      [74.9456573],
                      [913.85128645],
                      [424.15996133],
                      [-9.5797488],
                      [-360.96841852],
                      [27.214226],
                      [150.87705909],
                      [-492.17489392]]),
            decimal=3)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.data_longley.exog, self.data_longley.endog),
            -117.561627187,
            places=3)

        self.assertEqual(
            self.model.loglike_per_sample(self.data_longley.exog, self.data_longley.endog).shape,
            (16, ))

    def test_ols_elastic_net_regularized(self):
        # sklearn elastic net and l1 does not take sample_weights, will not test
        pass

    def test_ols_sample_weight_all_half(self):
        self.model = OLS(
            solver='pinv', fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.data_longley.exog, self.data_longley.endog, sample_weight=0.5)
        # coefficient
        np.testing.assert_array_almost_equal(
            self.model.coef,
            np.array((-3482258.63459582, 15.0618722713733, -0.358191792925910E-01,
                      -2.02022980381683, -1.03322686717359, -0.511041056535807E-01,
                      1829.15146461355)).reshape(1, -1),
            decimal=3)
        # std.err of coefficient (calibrated by df_resid)
        np.testing.assert_array_almost_equal(
            old_div(self.model.stderr, np.sqrt(old_div(9., self.data_longley.exog.shape[0]))),
            np.array((890420.383607373, 84.9149257747669, 0.334910077722432E-01,
                      0.488399681651699, 0.214274163161675, 0.226073200069370,
                      455.478499142212)).reshape(1, -1),
            decimal=1)
        # scale
        np.testing.assert_array_almost_equal(
            old_div(self.model.dispersion, (old_div(9., self.data_longley.exog.shape[0]))),
            np.array((92936.0061673238)))
        # predict
        np.testing.assert_array_almost_equal(
            self.data_longley.endog.reshape(-1, 1) - self.model.predict(self.data_longley.exog),
            np.array((267.34003, -94.01394, 46.28717, -410.11462,
                      309.71459, -249.31122, -164.04896, -13.18036, 14.30477, 455.39409,
                      -17.26893, -39.05504, -155.54997, -85.67131, 341.93151,
                      -206.75783)).reshape(-1, 1),
            decimal=3)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.data_longley.exog, self.data_longley.endog),
            -109.61743480849013,
            places=3)

        self.assertEqual(
            self.model.loglike_per_sample(self.data_longley.exog, self.data_longley.endog).shape,
            (16, ))

    def test_ols_sample_weight_all_zero(self):
        self.model = OLS(
            solver='pinv', fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.assertRaises(ValueError, self.model.fit,
                          self.data_longley.exog, self.data_longley.endog, 0)

    def test_ols_sample_weight_half_zero_half_one(self):
        self.model = OLS(
            solver='pinv', fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        len_half = 8
        self.model.fit(self.data_longley.exog, self.data_longley.endog,
                       sample_weight=np.array([1] * len_half +
                                              [0] * (self.data_longley.exog.shape[0] - len_half)))
        self.model_half = OLS(
            solver='pinv', fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model_half.fit(self.data_longley.exog[:len_half], self.data_longley.endog[:len_half])
        # coefficient
        np.testing.assert_array_almost_equal(
            self.model.coef,
            self.model_half.coef,
            decimal=3)
        # std.err
        np.testing.assert_array_almost_equal(
            self.model.stderr,
            self.model_half.stderr,
            decimal=3)

        # scale
        np.testing.assert_array_almost_equal(
            self.model.dispersion,
            self.model_half.dispersion,
            decimal=3)

    # corner cases
    def test_ols_one_data_point(self):
        self.model = OLS(
            solver='pinv', fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.data_longley.exog[0:1, :],
                       self.data_longley.endog[0:1, ], sample_weight=0.5)
        # coef
        self.assertEqual(self.model.coef.shape, (1, 7))
        # scale
        self.assertAlmostEqual(self.model.dispersion, np.array([[0]]), places=6)
        # loglike_per_sample
        np.testing.assert_array_equal(self.model.loglike_per_sample(
            self.data_longley.exog[0:1, :], self.data_longley.endog[0:1, ]), np.array([0]))
        np.testing.assert_array_almost_equal(self.model.loglike_per_sample(
            np.array(self.data_longley.exog[0:1, :].tolist() * 6),
            np.array([60323, 0, 60323, 60322, 60322, 60323])),
            np.array([0, -np.Infinity, 0, -np.Infinity, -np.Infinity, 0]), decimal=3)

    def test_ols_multicolinearty(self):
        self.model_col = OLS(
            solver='pinv', fit_intercept=False, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        X = np.hstack([self.data_longley.exog[:, 0:1], self.data_longley.exog[:, 0:1]])
        self.model_col.fit(X,
                           self.data_longley.endog, sample_weight=0.8)
        self.model = OLS(
            solver='pinv', fit_intercept=False, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.data_longley.exog[:, 0:1],
                       self.data_longley.endog, sample_weight=0.8)
        # coef
        np.testing.assert_array_almost_equal(
            self.model_col.coef, np.array([319.47969664, 319.47969664]).reshape(1, -1), decimal=3)
        # stderr
        self.assertEqual(self.model_col.stderr, None)
        # scale
        np.testing.assert_array_almost_equal(
            self.model_col.dispersion, self.model.dispersion, decimal=3)
        # loglike_per_sample
        np.testing.assert_array_almost_equal(
            self.model_col.loglike_per_sample(X, self.data_longley.endog),
            self.model.loglike_per_sample(self.data_longley.exog[:, 0:1],
                                          self.data_longley.endog), decimal=3)
        np.testing.assert_array_almost_equal(
            self.model_col.predict(X),
            self.model.predict(self.data_longley.exog[:, 0:1]), decimal=3)


class IndependentMultivariateOLSTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        cls.X = np.random.normal(size=(1000, 1))
        cls.Y = np.random.normal(size=(cls.X.shape[0], 2))

    def test_ols(self):
        self.model = OLS(
            solver='pinv', fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.X, self.Y)
        # coefficient
        print(self.model.coef)
        print(self.model.dispersion)
        print(self.model.stderr)
        print(self.Y - self.model.predict(self.X))
        print(self.model.loglike(self.X, self.Y))

        self.assertEqual(self.model.coef.shape, (2, 2))
        np.testing.assert_array_almost_equal(
            self.model.coef,
            np.array([[-0.02924966, -0.03484827],
                      [-0.00978688, 0.00336316]]).reshape(2, -1),
            decimal=3)
        # std.err of coefficient (calibrated by df_resid)
        self.assertEqual(self.model.stderr.shape, (2, 2))
        np.testing.assert_array_almost_equal(
            self.model.stderr,
            np.array([[0.03083908, 0.03121143],
                      [0.03002101, 0.03038348]]).reshape(2, -1),
            decimal=2)
        # scale
        self.assertEqual(self.model.dispersion.shape, (2, 2))
        np.testing.assert_array_almost_equal(
            self.model.dispersion,
            np.array([[0.94905363, 0.0164185],
                      [0.0164185, 0.89937019]]),
            decimal=3)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.X, self.Y),
            -2758.54387369,
            places=3)

        # to_json
        json_dict = self.model.to_json('./tests/linear_models/OLS/MultivariateOLS/')
        self.assertEqual(json_dict['properties']['solver'], 'pinv')

        # from_json
        self.model_from_json = OLS.from_json(json_dict)
        np.testing.assert_array_almost_equal(
            self.model.coef,
            self.model_from_json.coef,
            decimal=3)
        np.testing.assert_array_almost_equal(
            self.model.stderr,
            self.model_from_json.stderr,
            decimal=3)
        np.testing.assert_array_almost_equal(
            self.model.dispersion,
            self.model_from_json.dispersion,
            decimal=3)

    def test_ols_l2_regularized(self):
        self.model = OLS(
            solver='auto', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0.1, l1_ratio=1,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.X, self.Y)
        # coefficient
        print(self.model.coef)
        print(self.model.dispersion)
        print(self.model.loglike(self.X, self.Y))

        self.assertEqual(self.model.coef.shape, (2, 2))
        np.testing.assert_array_almost_equal(
            self.model.coef,
            np.array([[-0.0292465, -0.03484456],
                      [-0.00978591, 0.00336286]]).reshape(2, -1),
            decimal=3)
        # std.err of coefficient (calibrated by df_resid)
        self.assertTrue(self.model.stderr is None)
        # scale
        self.assertEqual(self.model.dispersion.shape, (2, 2))
        np.testing.assert_array_almost_equal(
            self.model.dispersion,
            np.array([[0.94905363, 0.0164185],
                      [0.0164185, 0.89937019]]),
            decimal=3)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.X, self.Y),
            -2758.5438737,
            places=3)

    def test_ols_l1_regularized(self):
        # sklearn l1 and elstic net does not support sample weight
        pass

    def test_ols_sample_weight_all_half(self):
        self.model = OLS(
            solver='pinv', fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.X, self.Y, sample_weight=0.5)
        # coefficient
        print(self.model.coef)
        print(self.model.dispersion)
        print(self.model.stderr)
        print(self.model.loglike(self.X, self.Y, sample_weight=0.5))

        self.assertEqual(self.model.coef.shape, (2, 2))
        np.testing.assert_array_almost_equal(
            self.model.coef,
            np.array([[-0.02924966, -0.03484827],
                      [-0.00978688, 0.00336316]]).reshape(2, -1),
            decimal=3)
        # std.err of coefficient (calibrated by df_resid)
        self.assertEqual(self.model.stderr.shape, (2, 2))
        np.testing.assert_array_almost_equal(
            self.model.stderr,
            np.array([[0.03083908, 0.03121143],
                      [0.03002101, 0.03038348]]).reshape(2, -1),
            decimal=2)
        # scale
        self.assertEqual(self.model.dispersion.shape, (2, 2))
        np.testing.assert_array_almost_equal(
            self.model.dispersion,
            np.array([[0.94905363, 0.0164185],
                      [0.0164185, 0.89937019]]),
            decimal=3)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.X, self.Y, 0.5),
            old_div(-2758.54387369, 2.),
            places=3)

        self.assertEqual(
            self.model.loglike_per_sample(self.X, self.Y).shape,
            (1000, ))

    def test_ols_sample_weight_all_zero(self):
        self.model = OLS(
            solver='pinv', fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.assertRaises(ValueError, self.model.fit, self.X, self.Y, 0)

    def test_ols_sample_weight_half_zero_half_one(self):
        self.model = OLS(
            solver='pinv', fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        len_half = 500
        self.model.fit(self.X, self.Y,
                       sample_weight=np.array([1] * len_half +
                                              [0] * (self.X.shape[0] - len_half)))
        self.model_half = OLS(
            solver='pinv', fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model_half.fit(self.X[:len_half], self.Y[:len_half])
        # coefficient
        np.testing.assert_array_almost_equal(
            self.model.coef,
            self.model_half.coef,
            decimal=3)
        # std.err
        np.testing.assert_array_almost_equal(
            self.model.stderr,
            self.model_half.stderr,
            decimal=3)

        # scale
        np.testing.assert_array_almost_equal(
            self.model.dispersion,
            self.model_half.dispersion,
            decimal=3)

    # corner cases
    def test_ols_one_data_point(self):
        self.model = OLS(
            solver='pinv', fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.X[0:1, :],
                       self.Y[0:1, ], sample_weight=0.5)
        # coef
        self.assertEqual(self.model.coef.shape, (2, 2))
        # scale
        np.testing.assert_array_almost_equal(
            self.model.dispersion, np.array([[0, 0], [0, 0]]), decimal=6)
        # loglike_per_sample
        np.testing.assert_array_equal(self.model.loglike_per_sample(
            self.X[0:1, :], self.Y[0:1, ]), np.array([0]))

        np.testing.assert_array_almost_equal(self.model.loglike_per_sample(
            np.array(self.X[0:1, :].tolist() * 6),
            np.array([self.Y[0, ], self.Y[1, ], self.Y[0, ],
                      self.Y[1, ], self.Y[1, ], self.Y[0, ]])),
            np.array([0, -np.Infinity, 0, -np.Infinity, -np.Infinity, 0]), decimal=3)

    def test_ols_multicolinearty(self):
        self.model_col = OLS(
            solver='pinv', fit_intercept=False, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        X = np.hstack([self.X[:, 0:1], self.X[:, 0:1]])
        self.model_col.fit(X,
                           self.Y, sample_weight=0.5)
        self.model = OLS(
            solver='pinv', fit_intercept=False, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.X[:, 0:1],
                       self.Y, sample_weight=0.5)
        # stderr
        self.assertEqual(self.model_col.stderr, None)
        # scale
        np.testing.assert_array_almost_equal(
            self.model_col.dispersion, self.model.dispersion, decimal=3)
        # loglike_per_sample
        np.testing.assert_array_almost_equal(
            self.model_col.loglike_per_sample(X, self.Y),
            self.model.loglike_per_sample(self.X[:, 0:1],
                                          self.Y), decimal=0)
        np.testing.assert_array_almost_equal(
            self.model_col.predict(X),
            self.model.predict(self.X[:, 0:1]), decimal=1)


class PerfectCorrelationMultivariateOLSTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        cls.data_longley = sm.datasets.longley.load()
        cls.X = cls.data_longley.exog
        cls.Y = np.hstack((cls.data_longley.endog.reshape(-1, 1),
                           cls.data_longley.endog.reshape(-1, 1)))

    def test_ols(self):
        self.model = OLS(
            solver='auto', fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.X, self.Y)
        # coefficient
        self.assertEqual(self.model.coef.shape, (2, 7))
        np.testing.assert_array_almost_equal(
            self.model.coef,
            np.array([[-3482258.63459582, 15.0618722713733, -0.358191792925910E-01,
                       -2.02022980381683, -1.03322686717359, -0.511041056535807E-01,
                       1829.15146461355],
                      [-3482258.63459582, 15.0618722713733, -0.358191792925910E-01,
                       -2.02022980381683, -1.03322686717359, -0.511041056535807E-01,
                       1829.15146461355]]).reshape(2, -1),
            decimal=3)
        # std.err of coefficient (calibrated by df_resid)
        self.assertEqual(self.model.stderr.shape, (2, 7))
        np.testing.assert_array_almost_equal(
            old_div(self.model.stderr, np.sqrt(old_div(9., self.data_longley.exog.shape[0]))),
            np.array([[890420.383607373, 84.9149257747669, 0.03349,
                       0.488399681651699, 0.214274163161675, 0.226073200069370,
                       455.478499142212],
                      [890420.383607373, 84.9149257747669, 0.03349,
                       0.488399681651699, 0.214274163161675, 0.226073200069370,
                       455.478499142212]]).reshape(2, -1),
            decimal=2)
        # scale
        self.assertEqual(self.model.dispersion.shape, (2, 2))
        np.testing.assert_array_almost_equal(
            old_div(self.model.dispersion, (old_div(9., self.data_longley.exog.shape[0]))),
            np.array([[92936.0061673238, 92936.0061673238],
                      [92936.0061673238, 92936.0061673238]]),
            decimal=3)
        # predict
        np.testing.assert_array_almost_equal(
            self.Y - self.model.predict(self.X),
            np.hstack((np.array([267.34003, -94.01394, 46.28717, -410.11462,
                                 309.71459, -249.31122, -164.04896, -13.18036, 14.30477, 455.39409,
                                 -17.26893, -39.05504, -155.54997, -85.67131, 341.93151,
                                 -206.75783]).reshape(-1, 1),
                       np.array([267.34003, -94.01394, 46.28717, -410.11462,
                                 309.71459, -249.31122, -164.04896, -13.18036, 14.30477, 455.39409,
                                 -17.26893, -39.05504, -155.54997, -85.67131, 341.93151,
                                 -206.75783]).reshape(-1, 1))),
            decimal=3)
        # loglike/_per_sample
        self.assertRaises(ValueError,
                          self.model.loglike_per_sample, self.X, self.Y)

    def test_ols_l1_regularized(self):
        # sklearn elastic net and l1 does not take sample_weights, will not test
        pass

    def test_ols_l2_regularized(self):
        # there is a bug in sklearn with weights, it can only use list right now
        self.model = OLS(
            solver='auto', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0.1, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.X, self.Y, sample_weight=0.5)

        # coefficient
        np.testing.assert_array_almost_equal(
            self.model.coef,
            np.array([[-2.0172203, -52.14364269, 0.07089677, -0.42552125,
                       -0.57305292, -0.41272483, 48.32484052],
                      [-2.0172203, -52.14364269, 0.07089677, -0.42552125,
                       -0.57305292, -0.41272483, 48.32484052]]).reshape(2, -1),
            decimal=3)
        # std.err of coefficient (calibrated by df_resid)
        self.assertTrue(self.model.stderr is None)
        # scale
        self.assertEqual(self.model.dispersion.shape, (2, 2))
        np.testing.assert_array_almost_equal(
            old_div(self.model.dispersion, (old_div(9., self.data_longley.exog.shape[0]))),
            np.array([[250870.081, 250870.081],
                      [250870.081, 250870.081]]),
            decimal=3)
        # predict
        res = np.array([[280.31871146],
                        [-131.6981265],
                        [90.64414685],
                        [-400.10244445],
                        [-440.59604167],
                        [-543.88595187],
                        [200.70483416],
                        [215.88629903],
                        [74.9456573],
                        [913.85128645],
                        [424.15996133],
                        [-9.5797488],
                        [-360.96841852],
                        [27.214226],
                        [150.87705909],
                        [-492.17489392]])
        np.testing.assert_array_almost_equal(
            self.Y - self.model.predict(self.X),
            np.hstack((res, res)),
            decimal=3)

        # loglike/_per_sample
        self.assertRaises(ValueError,
                          self.model.loglike, self.X, self.Y)

    def test_ols_elastic_net_regularized(self):
        # sklearn elastic net and l1 does not take sample_weights, will not test
        pass

    def test_ols_sample_weight_all_half(self):
        self.model = OLS(
            solver='pinv', fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.X, self.Y, sample_weight=0.5)
        # coefficient
        np.testing.assert_array_almost_equal(
            self.model.coef,
            np.array(((-3482258.63459582, 15.0618722713733, -0.358191792925910E-01,
                       -2.02022980381683, -1.03322686717359, -0.511041056535807E-01,
                       1829.15146461355),
                      (-3482258.63459582, 15.0618722713733, -0.358191792925910E-01,
                       -2.02022980381683, -1.03322686717359, -0.511041056535807E-01,
                       1829.15146461355))).reshape(2, -1),
            decimal=3)
        # std.err of coefficient (calibrated by df_resid)
        np.testing.assert_array_almost_equal(
            old_div(self.model.stderr, np.sqrt(old_div(9., self.data_longley.exog.shape[0]))),
            np.array(((890420.383607373, 84.9149257747669, 0.334910077722432E-01,
                       0.488399681651699, 0.214274163161675, 0.226073200069370,
                       455.478499142212),
                      (890420.383607373, 84.9149257747669, 0.334910077722432E-01,
                       0.488399681651699, 0.214274163161675, 0.226073200069370,
                       455.478499142212))).reshape(2, -1),
            decimal=1)
        # scale
        np.testing.assert_array_almost_equal(
            old_div(self.model.dispersion, (old_div(9., self.data_longley.exog.shape[0]))),
            np.array(((92936.0061673238, 92936.0061673238),
                      (92936.0061673238, 92936.0061673238))),
            decimal=3)
        # predict
        res = np.array((267.34003, -94.01394, 46.28717, -410.11462,
                        309.71459, -249.31122, -164.04896, -13.18036, 14.30477, 455.39409,
                        -17.26893, -39.05504, -155.54997, -85.67131, 341.93151,
                        -206.75783)).reshape(-1, 1)
        np.testing.assert_array_almost_equal(
            self.Y - self.model.predict(self.X),
            np.hstack((res, res)),
            decimal=3)
        # loglike/_per_sample
        self.assertRaises(ValueError,
                          self.model.loglike, self.X, self.Y)

    def test_ols_sample_weight_all_zero(self):
        self.model = OLS(
            solver='pinv', fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.assertRaises(ValueError, self.model.fit, self.X, self.Y, 0)

    def test_ols_sample_weight_half_zero_half_one(self):
        self.model = OLS(
            solver='pinv', fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        len_half = 8
        self.model.fit(self.X, self.Y,
                       sample_weight=np.array([1] * len_half +
                                              [0] * (self.data_longley.exog.shape[0] - len_half)))
        self.model_half = OLS(
            solver='pinv', fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model_half.fit(self.X[:len_half], self.Y[:len_half])
        # coefficient
        np.testing.assert_array_almost_equal(
            self.model.coef,
            self.model_half.coef,
            decimal=3)
        # std.err
        np.testing.assert_array_almost_equal(
            self.model.stderr,
            self.model_half.stderr,
            decimal=3)

        # scale
        np.testing.assert_array_almost_equal(
            self.model.dispersion,
            self.model_half.dispersion,
            decimal=3)

    # corner cases
    def test_ols_one_data_point(self):
        self.model = OLS(
            solver='pinv', fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.X[0:1, :],
                       self.Y[0:1, ], sample_weight=0.5)
        # coef
        self.assertEqual(self.model.coef.shape, (2, 7))
        # scale
        np.testing.assert_array_almost_equal(
            self.model.dispersion, np.array([[0, 0], [0, 0]]), decimal=6)
        # loglike_per_sample
        np.testing.assert_array_equal(self.model.loglike_per_sample(
            self.X[0:1, :], self.Y[0:1, ]), np.array([0]))
        np.testing.assert_array_almost_equal(self.model.loglike_per_sample(
            np.array(self.X[0:1, :].tolist() * 6),
            np.array([[60323, 60323], [0, 60323], [60323, 60323],
                      [60322, 60323], [60322, 60322], [60323, 60323]])),
            np.array([0, -np.Infinity, 0, -np.Infinity, -np.Infinity, 0]), decimal=3)

    def test_ols_multicolinearty(self):
        self.model_col = OLS(
            solver='pinv', fit_intercept=False, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        X = np.hstack([self.X[:, 0:1], self.X[:, 0:1]])
        self.model_col.fit(X,
                           self.Y, sample_weight=0.8)
        self.model = OLS(
            solver='pinv', fit_intercept=False, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.X[:, 0:1],
                       self.Y, sample_weight=0.8)
        # coef
        np.testing.assert_array_almost_equal(
            self.model_col.coef, np.array([[319.47969664, 319.47969664],
                                           [319.47969664, 319.47969664]]).reshape(2, -1), decimal=3)
        # stderr
        self.assertEqual(self.model_col.stderr, None)
        # scale
        np.testing.assert_array_almost_equal(
            self.model_col.dispersion, self.model.dispersion, decimal=3)
        # loglike_per_sample
        self.assertRaises(ValueError,
                          self.model_col.loglike, X, self.Y)
        np.testing.assert_array_almost_equal(
            self.model_col.predict(X),
            self.model.predict(self.X[:, 0:1]), decimal=3)
