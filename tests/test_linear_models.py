# import json
import unittest


import numpy as np
from sklearn.preprocessing import label_binarize
import statsmodels.api as sm


from IOHMM import UnivariateOLS, DiscreteMNL, CrossEntropyMNL

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
        self.model = UnivariateOLS(
            solver='pinv', fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.data_longley.exog, self.data_longley.endog)
        # coefficient
        np.testing.assert_array_almost_equal(
            self.model.coef,
            (-3482258.63459582, 15.0618722713733, -0.358191792925910E-01,
             -2.02022980381683, -1.03322686717359, -0.511041056535807E-01,
             1829.15146461355),
            decimal=3)
        # std.err of coefficient (calibrated by df_resid)
        np.testing.assert_array_almost_equal(
            self.model.stderr / np.sqrt(9. / self.data_longley.exog.shape[0]),
            (890420.383607373, 84.9149257747669, 0.03349,
             0.488399681651699, 0.214274163161675, 0.226073200069370,
             455.478499142212),
            decimal=2)
        # scale
        self.assertAlmostEqual(
            self.model.dispersion / (9. / self.data_longley.exog.shape[0]),
            92936.0061673238,
            places=3)
        # predict
        np.testing.assert_array_almost_equal(
            self.data_longley.endog - self.model.predict(self.data_longley.exog),
            np.array((267.34003, -94.01394, 46.28717, -410.11462,
                      309.71459, -249.31122, -164.04896, -13.18036, 14.30477, 455.39409,
                      -17.26893, -39.05504, -155.54997, -85.67131, 341.93151,
                      -206.75783)),
            decimal=3)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.data_longley.exog, self.data_longley.endog),
            -109.61743480849013,
            places=3)
        # to_json
        json_dict = self.model.to_json('./tests/linear_models/ols/')
        self.assertEqual(json_dict['properties']['solver'], 'pinv')

        # from_json
        self.model_from_json = UnivariateOLS.from_json(json_dict)
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

    def test_ols_regularized(self):
        # Wierdly, WLS does not have fit_regularized in this version
        # will keep this as a place holder
        pass

    def test_ols_sample_weight_all_half(self):
        self.model = UnivariateOLS(
            solver='pinv', fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.data_longley.exog, self.data_longley.endog, sample_weight=0.5)
        # coefficient
        np.testing.assert_array_almost_equal(
            self.model.coef,
            (-3482258.63459582, 15.0618722713733, -0.358191792925910E-01,
             -2.02022980381683, -1.03322686717359, -0.511041056535807E-01,
             1829.15146461355),
            decimal=3)
        # std.err of coefficient (calibrated by df_resid)
        np.testing.assert_array_almost_equal(
            self.model.stderr / np.sqrt(9. / self.data_longley.exog.shape[0]),
            (890420.383607373, 84.9149257747669, 0.334910077722432E-01,
             0.488399681651699, 0.214274163161675, 0.226073200069370,
             455.478499142212),
            decimal=1)
        # scale
        self.assertAlmostEqual(
            self.model.dispersion / (9. / self.data_longley.exog.shape[0]),
            92936.0061673238,
            places=3)
        # predict
        np.testing.assert_array_almost_equal(
            self.data_longley.endog - self.model.predict(self.data_longley.exog),
            np.array((267.34003, -94.01394, 46.28717, -410.11462,
                      309.71459, -249.31122, -164.04896, -13.18036, 14.30477, 455.39409,
                      -17.26893, -39.05504, -155.54997, -85.67131, 341.93151,
                      -206.75783)),
            decimal=3)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.data_longley.exog, self.data_longley.endog),
            -109.61743480849013,
            places=3)

    def test_ols_sample_weight_all_zero(self):
        self.model = UnivariateOLS(
            solver='pinv', fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.data_longley.exog, self.data_longley.endog, sample_weight=0)
        # coefficient
        self.assertTrue(self.model.coef is None)
        # std.err of coefficient (calibrated by df_resid)
        self.assertTrue(self.model.stderr is None)
        # scale
        self.assertTrue(self.model.dispersion is None)

    def test_ols_sample_weight_half_zero_half_one(self):
        self.model = UnivariateOLS(
            solver='pinv', fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        len_half = 8
        self.model.fit(self.data_longley.exog, self.data_longley.endog,
                       sample_weight=np.array([1] * len_half +
                                              [0] * (self.data_longley.exog.shape[0] - len_half)))
        self.model_half = UnivariateOLS(
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
        self.assertAlmostEqual(
            self.model.dispersion,
            self.model_half.dispersion,
            places=3)

    # corner cases
    def test_ols_one_data_point(self):
        self.model = UnivariateOLS(
            solver='pinv', fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.data_longley.exog[0:1, :],
                       self.data_longley.endog[0:1, ], sample_weight=0.5)
        # coef
        self.assertEqual(self.model.coef.shape, (7, ))
        # stderr
        np.testing.assert_array_equal(self.model.stderr, np.zeros(7))
        # scale
        self.assertEqual(self.model.dispersion, 0)
        # loglike_per_sample
        np.testing.assert_array_equal(self.model.loglike_per_sample(
            self.data_longley.exog[0:1, :], self.data_longley.endog[0:1, ]), np.array([0]))
        np.testing.assert_array_equal(self.model.loglike_per_sample(
            np.array(self.data_longley.exog[0:1, :].tolist() * 6),
            np.array([60323, 0, 60323, 60322, 60322, 60323])),
            np.array([0, -np.Infinity, 0, -np.Infinity, -np.Infinity, 0]))

    def test_ols_multicolinearty(self):
        self.model_col = UnivariateOLS(
            solver='pinv', fit_intercept=False, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        X = np.hstack([self.data_longley.exog[:, 0:1], self.data_longley.exog[:, 0:1]])
        self.model_col.fit(X,
                           self.data_longley.endog, sample_weight=0.5)
        self.model = UnivariateOLS(
            solver='pinv', fit_intercept=False, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.data_longley.exog[:, 0:1],
                       self.data_longley.endog, sample_weight=0.5)
        # coef
        np.testing.assert_array_almost_equal(
            self.model_col.coef, np.array([319.47969664, 319.47969664]), decimal=3)
        # stderr
        self.assertEqual(self.model_col.stderr, None)
        # scale
        self.assertAlmostEqual(self.model_col.dispersion, self.model.dispersion, places=3)
        # loglike_per_sample
        np.testing.assert_array_almost_equal(
            self.model_col.loglike_per_sample(X, self.data_longley.endog),
            self.model.loglike_per_sample(self.data_longley.exog[:, 0:1],
                                          self.data_longley.endog), decimal=3)
        np.testing.assert_array_almost_equal(
            self.model_col.predict(X),
            self.model.predict(self.data_longley.exog[:, 0:1]), decimal=3)


class DiscreteMNLUnaryTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_spector = sm.datasets.spector.load()
        cls.y = np.array(['foo'] * cls.data_spector.endog.shape[0])

    def test_lr(self):
        self.model = DiscreteMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, classes=None)
        self.model.fit(self.data_spector.exog, self.y)
        # coefficient
        np.testing.assert_array_equal(
            self.model.coef,
            np.zeros((4, 1)))

        # predict
        np.testing.assert_array_equal(
            self.model.predict(self.data_spector.exog),
            np.array(['foo'] * self.data_spector.endog.shape[0]))
        # loglike/_per_sample
        np.testing.assert_array_equal(
            self.model.loglike_per_sample(self.data_spector.exog,
                                          np.array(['bar'] * 16 + ['foo'] * 16)),
            np.array([-np.Infinity] * 16 + [0] * 16))

    def test_lr_sample_weight_all_half(self):
        self.model = DiscreteMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, classes=None)
        self.model.fit(self.data_spector.exog, self.y, sample_weight=.5)
        # coefficient
        np.testing.assert_array_equal(
            self.model.coef,
            np.zeros((4, 1)))
        # loglike/_per_sample
        self.assertEqual(
            self.model.loglike(self.data_spector.exog, self.y, sample_weight=.5), 0)

    # corner cases
    def test_lr_one_data_point(self):
        # with regularization
        self.model = DiscreteMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, classes=None)
        self.model.fit(self.data_spector.exog[4:5, :],
                       self.y[4:5, ], sample_weight=0.5)
        # coef
        np.testing.assert_array_equal(
            self.model.coef,
            np.zeros((4, 1)))
        # loglike_per_sample
        np.testing.assert_array_almost_equal(self.model.loglike_per_sample(
            self.data_spector.exog[4:6, :], np.array(['foo', 'foo'])),
            np.array([0, 0]), decimal=3)

        np.testing.assert_array_almost_equal(self.model.loglike_per_sample(
            self.data_spector.exog[4:6, :], np.array(['foo', 'bar'])),
            np.array([0, -np.Infinity]), decimal=3)


class DiscreteMNLBinaryTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_spector = sm.datasets.spector.load()

    def test_lr(self):
        self.model = DiscreteMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, classes=None)
        self.model.fit(self.data_spector.exog, self.data_spector.endog)
        # coefficient
        np.testing.assert_array_almost_equal(
            self.model.coef,
            np.array([[-13.021, 2.8261, .09515, 2.378]]),
            decimal=3)

        # predict
        np.testing.assert_array_almost_equal(
            self.model.predict(self.data_spector.exog),
            np.array((0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,
                      0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  1.,  0.,  1.,  1.,  0.,
                      1.,  0.,  1.,  1.,  1.,  0.)),
            decimal=3)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.data_spector.exog, self.data_spector.endog),
            -12.8896334653335,
            places=3)
        # to_json
        json_dict = self.model.to_json('./tests/linear_models/discretemnl_lr/')
        self.assertEqual(json_dict['properties']['solver'], 'lbfgs')

        # from_json
        self.model_from_json = DiscreteMNL.from_json(json_dict)
        np.testing.assert_array_almost_equal(
            self.model.coef,
            self.model_from_json.coef,
            decimal=3)
        np.testing.assert_array_almost_equal(
            self.model.classes, np.array([0, 1]), decimal=3)
        self.assertEqual(self.model.n_classes, 2)

    def test_lr_regularized(self):
        self.model = DiscreteMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=.01, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, classes=None)
        self.model.fit(self.data_spector.exog, self.data_spector.endog)
        # coefficient
        np.testing.assert_array_almost_equal(
            self.model.coef,
            np.array([[-10.66,   2.364,   0.064,   2.142]]),
            decimal=3)

        # predict
        np.testing.assert_array_almost_equal(
            self.model.predict(self.data_spector.exog),
            np.array((0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,
                      0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  1.,  0.,  1.,  1.,  0.,
                      1.,  0.,  1.,  1.,  1.,  0.)),
            decimal=3)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.data_spector.exog, self.data_spector.endog),
            -13.016861222748519,
            places=3)
        # to_json
        json_dict = self.model.to_json('./tests/linear_models/discretemnl_lr/')
        self.assertEqual(json_dict['properties']['solver'], 'lbfgs')
        self.assertEqual(json_dict['properties']['alpha'], .01)

        # from_json
        self.model_from_json = DiscreteMNL.from_json(json_dict)
        np.testing.assert_array_almost_equal(
            self.model.coef,
            self.model_from_json.coef,
            decimal=3)
        self.assertEqual(self.model_from_json.alpha, .01)

    def test_lr_sample_weight_all_half(self):
        self.model = DiscreteMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, classes=None)
        self.model.fit(self.data_spector.exog, self.data_spector.endog, sample_weight=.5)
        # coefficient
        np.testing.assert_array_almost_equal(
            self.model.coef,
            np.array([[-13.021, 2.8261, .09515, 2.378]]),
            decimal=3)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.data_spector.exog, self.data_spector.endog, sample_weight=.5),
            -12.8896334653335 / 2.,
            places=3)

    def test_lr_sample_weight_all_zero(self):
        self.model = DiscreteMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, classes=None)
        self.model.fit(self.data_spector.exog, self.data_spector.endog, sample_weight=0)
        # coefficient
        self.assertTrue(self.model.coef is None)
        self.assertTrue(self.model.loglike(self.data_spector.exog, self.data_spector.endog) is None)

    def test_lr_sample_weight_half_zero_half_one(self):
        self.model = DiscreteMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, classes=None)
        len_half = 8
        self.model.fit(self.data_spector.exog, self.data_spector.endog,
                       sample_weight=np.array([1] * len_half +
                                              [0] * (self.data_spector.exog.shape[0] - len_half)))
        self.model_half = DiscreteMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, classes=None)
        self.model_half.fit(self.data_spector.exog[:len_half], self.data_spector.endog[:len_half])
        # coefficient
        np.testing.assert_array_almost_equal(
            self.model.coef,
            self.model_half.coef,
            decimal=3)

    # corner cases
    def test_lr_two_data_point(self):
        # with regularization
        self.model = DiscreteMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=.01, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, classes=None)
        self.model.fit(self.data_spector.exog[4:6, :],
                       self.data_spector.endog[4:6, ], sample_weight=0.5)
        # coef
        self.assertEqual(self.model.coef.shape, (1, 4))
        # loglike_per_sample
        np.testing.assert_array_almost_equal(self.model.loglike_per_sample(
            self.data_spector.exog[4:6, :], self.data_spector.endog[4:6, ]),
            np.array([-0.226, -0.289]), decimal=3)
        # with no regularization
        self.model = DiscreteMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, classes=None)
        self.model.fit(self.data_spector.exog[4:6, :],
                       self.data_spector.endog[4:6, ], sample_weight=0.5)
        # coef
        self.assertEqual(self.model.coef.shape, (1, 4))
        # loglike_per_sample
        np.testing.assert_array_almost_equal(self.model.loglike_per_sample(
            self.data_spector.exog[4:6, :], self.data_spector.endog[4:6, ]),
            np.array([0, 0]), decimal=3)
        # class in reverse
        self.model = DiscreteMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, classes=None)
        self.model.fit(self.data_spector.exog[3:5, :],
                       self.data_spector.endog[3:5, ], sample_weight=0.5)
        # coef
        self.assertEqual(self.model.coef.shape, (1, 4))
        # loglike_per_sample
        np.testing.assert_array_almost_equal(self.model.loglike_per_sample(
            self.data_spector.exog[3:5, :], self.data_spector.endog[3:5, ]),
            np.array([0, 0]), decimal=3)
        print self.model.classes, 'class'
        np.testing.assert_array_almost_equal(self.model.loglike_per_sample(
            self.data_spector.exog[3:5, :], np.array([0, 2])),
            np.array([0, -np.Infinity]), decimal=3)

    def test_lr_multicolinearty(self):
        self.model_col = DiscreteMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, classes=None)
        X = np.hstack([self.data_spector.exog[:, 0:1], self.data_spector.exog[:, 0:1]])
        self.model_col.fit(X,
                           self.data_spector.endog, sample_weight=0.5)
        self.model = DiscreteMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, classes=None)
        self.model.fit(self.data_spector.exog[:, 0:1],
                       self.data_spector.endog, sample_weight=0.5)

        np.testing.assert_array_almost_equal(
            self.model_col.coef, np.array([[-9.703,  1.42002783,  1.42002783]]), decimal=3)
        # loglike_per_sample
        np.testing.assert_array_almost_equal(
            self.model_col.loglike_per_sample(X, self.data_spector.endog),
            self.model.loglike_per_sample(self.data_spector.exog[:, 0:1],
                                          self.data_spector.endog), decimal=3)
        np.testing.assert_array_almost_equal(
            self.model_col.predict(X),
            self.model.predict(self.data_spector.exog[:, 0:1]), decimal=3)


class DiscreteMNLMultinomialTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_anes96 = sm.datasets.anes96.load()

    def test_lr(self):
        self.model = DiscreteMNL(
            solver='newton-cg', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, classes=None)
        self.model.fit(self.data_anes96.exog, self.data_anes96.endog)
        # coefficient
        # predict
        self.assertEqual(
            np.sum(self.model.predict(self.data_anes96.exog) ==
                   self.data_anes96.endog), 372)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.data_anes96.exog, self.data_anes96.endog),
            -1461.9227472481984,
            places=3)
        # to_json
        json_dict = self.model.to_json('./tests/linear_models/discretemnl_mlr/')
        self.assertEqual(json_dict['properties']['solver'], 'newton-cg')

        # from_json
        self.model_from_json = DiscreteMNL.from_json(json_dict)
        np.testing.assert_array_almost_equal(
            self.model.coef,
            self.model_from_json.coef,
            decimal=3)
        np.testing.assert_array_almost_equal(
            self.model.classes, np.array(range(7)), decimal=3)
        self.assertEqual(self.model.n_classes, 7)

    def test_lr_regularized(self):
        self.model = DiscreteMNL(
            solver='newton-cg', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=10, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, classes=None)
        self.model.fit(self.data_anes96.exog, self.data_anes96.endog)
        # predict
        self.assertEqual(
            np.sum(self.model.predict(self.data_anes96.exog) ==
                   self.data_anes96.endog), 333)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.data_anes96.exog, self.data_anes96.endog),
            -1540.888456277886,
            places=3)

    def test_lr_sample_weight_all_half(self):
        self.model_half = DiscreteMNL(
            solver='newton-cg', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, classes=None)
        self.model_half.fit(self.data_anes96.exog, self.data_anes96.endog, sample_weight=.5)
        self.model = DiscreteMNL(
            solver='newton-cg', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, classes=None)
        self.model.fit(self.data_anes96.exog, self.data_anes96.endog)
        # coefficient
        np.testing.assert_array_almost_equal(self.model.coef, self.model_half.coef, decimal=3)
        # predict
        self.assertEqual(
            np.sum(self.model.predict(self.data_anes96.exog) ==
                   self.data_anes96.endog), 372)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.data_anes96.exog, self.data_anes96.endog, sample_weight=.5),
            -1461.92274725 / 2.,
            places=3)

    def test_lr_sample_weight_all_zero(self):
        self.model = DiscreteMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, classes=None)
        self.model.fit(self.data_anes96.exog, self.data_anes96.endog, sample_weight=0)
        # coefficient
        self.assertTrue(self.model.coef is None)
        self.assertTrue(self.model.loglike(self.data_anes96.exog, self.data_anes96.endog) is None)

    def test_lr_sample_weight_half_zero_half_one(self):
        self.model = DiscreteMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, classes=None)
        len_half = 500
        self.model.fit(self.data_anes96.exog, self.data_anes96.endog,
                       sample_weight=np.array([1] * len_half +
                                              [0] * (self.data_anes96.exog.shape[0] - len_half)))
        self.model_half = DiscreteMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, classes=None)
        self.model_half.fit(self.data_anes96.exog[:len_half], self.data_anes96.endog[:len_half])
        # coefficient
        np.testing.assert_array_almost_equal(
            self.model.coef,
            self.model_half.coef,
            decimal=3)

    # corner cases
    def test_lr_three_data_point(self):
        # with regularization
        self.model = DiscreteMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=.1, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, classes=None)
        self.model.fit(self.data_anes96.exog[6:9, :],
                       self.data_anes96.endog[6:9, ], sample_weight=0.5)
        # coef
        self.assertEqual(self.model.coef.shape, (3, 6))
        # loglike_per_sample
        np.testing.assert_array_almost_equal(self.model.loglike_per_sample(
            self.data_anes96.exog[6:9, :], np.array([1, 4, 3])),
            np.array([-0.015, -0.089, -0.095]), decimal=3)
        np.testing.assert_array_almost_equal(self.model.loglike_per_sample(
            self.data_anes96.exog[6:9, :], np.array([3, 1, 4])),
            np.array([-4.2, -5.046, -2.827]), decimal=3)
        np.testing.assert_array_almost_equal(self.model.loglike_per_sample(
            self.data_anes96.exog[6:9, :], np.array([3, 0, 5])),
            np.array([-4.2, -np.Infinity,  -np.Infinity]), decimal=3)

    def test_lr_multicolinearty(self):
        self.model_col = DiscreteMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, classes=None)
        X = np.hstack([self.data_anes96.exog[:, 0:1], self.data_anes96.exog[:, 0:1]])
        self.model_col.fit(X,
                           self.data_anes96.endog, sample_weight=0.5)
        self.model = DiscreteMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, classes=None)
        self.model.fit(self.data_anes96.exog[:, 0:1],
                       self.data_anes96.endog, sample_weight=0.5)
        # loglike_per_sample
        np.testing.assert_array_almost_equal(
            self.model_col.loglike_per_sample(X, self.data_anes96.endog),
            self.model.loglike_per_sample(self.data_anes96.exog[:, 0:1],
                                          self.data_anes96.endog), decimal=3)
        np.testing.assert_array_almost_equal(
            self.model_col.predict(X),
            self.model.predict(self.data_anes96.exog[:, 0:1]), decimal=3)


class CrossEntropyMNLUnaryTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_spector = sm.datasets.spector.load()
        cls.y = np.ones((cls.data_spector.endog.shape[0], 1))

    def test_label_encoder(self):
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = np.array([[1], [1], [1]])
        X_repeated, Y_repeated, sample_weight_repeated = \
            CrossEntropyMNL._label_encoder(x, y)
        np.testing.assert_array_equal(X_repeated, x)
        np.testing.assert_array_equal(
            Y_repeated, np.array([0, 0, 0]))
        np.testing.assert_array_equal(
            sample_weight_repeated,
            np.array([1, 1, 1]))
        # with sample_weight
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = np.array([[1], [1], [1]])
        sample_weight = np.array([0.25, 0.5, 0.25])
        X_repeated, Y_repeated, sample_weight_repeated = \
            CrossEntropyMNL._label_encoder(x, y, sample_weight)
        np.testing.assert_array_equal(X_repeated, x)
        np.testing.assert_array_equal(
            Y_repeated, np.array([0, 0, 0]))
        np.testing.assert_array_equal(
            sample_weight_repeated, sample_weight)

    def test_lr(self):
        self.model = CrossEntropyMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        self.model.fit(self.data_spector.exog, self.y)
        # coefficient
        np.testing.assert_array_equal(
            self.model.coef,
            np.zeros((4, 1)))

        # predict
        np.testing.assert_array_equal(
            self.model.predict(self.data_spector.exog),
            np.array([0] * self.data_spector.endog.shape[0]))
        # loglike/_per_sample
        np.testing.assert_array_equal(
            self.model.loglike_per_sample(self.data_spector.exog,
                                          np.array([1] * 16 + [0] * 16).reshape(-1, 1)),
            np.array([0] * 16 + [-np.Infinity] * 16))

    def test_lr_sample_weight_all_half(self):
        self.model = CrossEntropyMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        self.model.fit(self.data_spector.exog, self.y, sample_weight=.5)
        # coefficient
        np.testing.assert_array_equal(
            self.model.coef,
            np.zeros((4, 1)))
        # loglike/_per_sample
        self.assertEqual(
            self.model.loglike(self.data_spector.exog, self.y, sample_weight=.5), 0)

    # corner cases
    def test_lr_one_data_point(self):
        # with regularization
        self.model = CrossEntropyMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        self.model.fit(self.data_spector.exog[4:5, :],
                       self.y[4:5, ], sample_weight=0.5)
        # coef
        np.testing.assert_array_equal(
            self.model.coef,
            np.zeros((4, 1)))
        # loglike_per_sample
        np.testing.assert_array_almost_equal(self.model.loglike_per_sample(
            self.data_spector.exog[4:6, :], np.array([1, 0]).reshape(-1, 1)),
            np.array([0, -np.Infinity]), decimal=3)
        np.testing.assert_array_almost_equal(self.model.loglike_per_sample(
            self.data_spector.exog[4:6, :], np.array([1, 1]).reshape(-1, 1)),
            np.array([0, 0]), decimal=3)


class CrossEntropyMNLBinaryTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_spector = sm.datasets.spector.load()
        cls.y = np.array([
            [1, 0], [1, 0], [1, 0], [1, 0],
            [0, 1], [1, 0], [1, 0], [1, 0],
            [1, 0], [0, 1], [1, 0], [1, 0],
            [1, 0], [0, 1], [1, 0], [1, 0],
            [1, 0], [1, 0], [1, 0], [0, 1],
            [1, 0], [0, 1], [1, 0], [1, 0],
            [0, 1], [0, 1], [0, 1], [1, 0],
            [0, 1], [0, 1], [1, 0], [0, 1]])
        cls.y_disturbed = np.array([
            [0.99, 0.01], [0.99, 0.01], [0.99, 0.01], [0.99, 0.01],
            [0.01, 0.99], [0.99, 0.01], [0.99, 0.01], [0.99, 0.01],
            [0.99, 0.01], [0.01, 0.99], [0.99, 0.01], [0.99, 0.01],
            [0.99, 0.01], [0.01, 0.99], [0.99, 0.01], [0.99, 0.01],
            [0.99, 0.01], [0.99, 0.01], [0.99, 0.01], [0.01, 0.99],
            [0.99, 0.01], [0.01, 0.99], [0.99, 0.01], [0.99, 0.01],
            [0.01, 0.99], [0.01, 0.99], [0.01, 0.99], [0.99, 0.01],
            [0.01, 0.99], [0.01, 0.99], [0.99, 0.01], [0.01, 0.99]])

    def test_lr(self):
        self.model = CrossEntropyMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        self.model.fit(self.data_spector.exog, self.y)
        # coefficient
        np.testing.assert_array_almost_equal(
            self.model.coef,
            np.array([[-13.021, 2.8261, .09515, 2.378]]),
            decimal=3)

        # predict
        np.testing.assert_array_almost_equal(
            self.model.predict(self.data_spector.exog),
            np.array((0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,
                      0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  1.,  0.,  1.,  1.,  0.,
                      1.,  0.,  1.,  1.,  1.,  0.)),
            decimal=3)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.data_spector.exog, self.y),
            -12.8896334653335,
            places=3)
        # to_json
        json_dict = self.model.to_json('./tests/linear_models/crossentropymnl_lr/')
        self.assertEqual(json_dict['properties']['solver'], 'lbfgs')

        # from_json
        self.model_from_json = CrossEntropyMNL.from_json(json_dict)
        np.testing.assert_array_almost_equal(
            self.model.coef,
            self.model_from_json.coef,
            decimal=3)
        np.testing.assert_array_almost_equal(
            self.model.classes, np.array([0, 1]), decimal=3)
        self.assertEqual(self.model.n_classes, 2)

    def test_lr_disturbed(self):
        self.model = CrossEntropyMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        self.model.fit(self.data_spector.exog, self.y_disturbed)
        # coefficient
        np.testing.assert_array_almost_equal(
            self.model.coef,
            np.array([[-12.327,  2.686,  0.089,  2.258]]),
            decimal=3)

        # predict
        np.testing.assert_array_almost_equal(
            self.model.predict(self.data_spector.exog),
            np.array((0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,
                      0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  1.,  0.,  1.,  1.,  0.,
                      1.,  0.,  1.,  1.,  1.,  0.)),
            decimal=3)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.data_spector.exog, self.y_disturbed),
            -13.366314173353134,
            places=3)

    def test_lr_regularized(self):
        self.model = CrossEntropyMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=.01, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        self.model.fit(self.data_spector.exog, self.y)
        # coefficient
        np.testing.assert_array_almost_equal(
            self.model.coef,
            np.array([[-10.66,   2.364,   0.064,   2.142]]),
            decimal=3)

        # predict
        np.testing.assert_array_almost_equal(
            self.model.predict(self.data_spector.exog),
            np.array((0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,
                      0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  1.,  0.,  1.,  1.,  0.,
                      1.,  0.,  1.,  1.,  1.,  0.)),
            decimal=3)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.data_spector.exog, self.y),
            -13.016861222748515,
            places=3)

    def test_lr_sample_weight_all_half(self):
        self.model = CrossEntropyMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        self.model.fit(self.data_spector.exog, self.y, sample_weight=.5)
        # coefficient
        np.testing.assert_array_almost_equal(
            self.model.coef,
            np.array([[-13.021, 2.8261, .09515, 2.378]]),
            decimal=3)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.data_spector.exog, self.y, sample_weight=.5),
            -12.8896334653335 / 2.,
            places=3)

    def test_lr_disturbed_sample_weight_all_half(self):
        self.model = CrossEntropyMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        self.model.fit(self.data_spector.exog, self.y_disturbed, sample_weight=.5)
        # coefficient
        np.testing.assert_array_almost_equal(
            self.model.coef,
            np.array([[-12.327,  2.686,  0.089,  2.258]]),
            decimal=3)

        # predict
        np.testing.assert_array_almost_equal(
            self.model.predict(self.data_spector.exog),
            np.array((0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,
                      0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  1.,  0.,  1.,  1.,  0.,
                      1.,  0.,  1.,  1.,  1.,  0.)),
            decimal=3)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.data_spector.exog, self.y_disturbed, sample_weight=.5),
            -13.366314173353134 / 2.,
            places=3)

    def test_lr_sample_weight_all_zero(self):
        self.model = CrossEntropyMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        self.model.fit(self.data_spector.exog, self.y, sample_weight=0)
        # coefficient
        self.assertTrue(self.model.coef is None)
        self.assertTrue(self.model.loglike(self.data_spector.exog, self.y) is None)

    def test_lr_sample_weight_half_zero_half_one(self):
        self.model = CrossEntropyMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        len_half = 8
        self.model.fit(self.data_spector.exog, self.y,
                       sample_weight=np.array([1] * len_half +
                                              [0] * (self.y.shape[0] - len_half)))
        self.model_half = CrossEntropyMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        self.model_half.fit(self.data_spector.exog[:len_half], self.y[:len_half])
        # coefficient
        np.testing.assert_array_almost_equal(
            self.model.coef,
            self.model_half.coef,
            decimal=3)

    def test_lr_disturbed_sample_weight_half_zero_half_one(self):
        self.model = CrossEntropyMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        len_half = 8
        self.model.fit(self.data_spector.exog, self.y_disturbed,
                       sample_weight=np.array([1] * len_half +
                                              [0] * (self.y_disturbed.shape[0] - len_half)))
        self.model_half = CrossEntropyMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        self.model_half.fit(self.data_spector.exog[:len_half], self.y_disturbed[:len_half])
        # coefficient
        np.testing.assert_array_almost_equal(
            self.model.coef,
            self.model_half.coef,
            decimal=3)

    # corner cases
    def test_lr_two_data_point(self):
        # with regularization
        self.model = CrossEntropyMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=.1, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        self.model.fit(self.data_spector.exog[4:6, :],
                       self.y[4:6, ], sample_weight=0.5)
        # coef
        self.assertEqual(self.model.coef.shape, (1, 4))
        # loglike_per_sample
        np.testing.assert_array_almost_equal(self.model.loglike_per_sample(
            self.data_spector.exog[4:6, :], self.y[4:6, ]),
            np.array([-0.495, -0.661]), decimal=3)
        # loglike_per_sample
        np.testing.assert_array_almost_equal(self.model.loglike_per_sample(
            self.data_spector.exog[4:6, :],
            np.array([[0, 0], [1, 0]])),
            np.array([-np.Infinity, -0.661]), decimal=3)
        np.testing.assert_array_almost_equal(self.model.loglike_per_sample(
            self.data_spector.exog[4:6, :],
            np.array([[0, 0], [0, 1]])),
            np.array([-np.Infinity, -0.726]), decimal=3)

    def test_lr_disturbed_two_data_point(self):
        # with regularization
        self.model = CrossEntropyMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=.1, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        self.model.fit(self.data_spector.exog[4:6, :],
                       self.y_disturbed[4:6, ], sample_weight=0.5)
        # coef
        self.assertEqual(self.model.coef.shape, (1, 4))
        # loglike_per_sample
        np.testing.assert_array_almost_equal(self.model.loglike_per_sample(
            self.data_spector.exog[4:6, :], self.y_disturbed[4:6, ]),
            np.array([-0.503, -0.662]), decimal=3)
        # loglike_per_sample
        np.testing.assert_array_almost_equal(self.model.loglike_per_sample(
            self.data_spector.exog[4:6, :],
            np.array([[0, 0], [0.99, 0.01]])),
            np.array([-np.Infinity, -0.662]), decimal=3)
        np.testing.assert_array_almost_equal(self.model.loglike_per_sample(
            self.data_spector.exog[4:6, :],
            np.array([[0, 0], [0.01, 0.99]])),
            np.array([-np.Infinity, -0.725]), decimal=3)

    def test_lr_multicolinearty(self):
        self.model_col = CrossEntropyMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        X = np.hstack([self.data_spector.exog[:, 0:1], self.data_spector.exog[:, 0:1]])
        self.model_col.fit(X,
                           self.y, sample_weight=0.5)
        self.model = CrossEntropyMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        self.model.fit(self.data_spector.exog[:, 0:1],
                       self.y, sample_weight=0.5)

        np.testing.assert_array_almost_equal(
            self.model_col.coef, np.array([[-9.703,  1.42002783,  1.42002783]]), decimal=3)
        # loglike_per_sample
        np.testing.assert_array_almost_equal(
            self.model_col.loglike_per_sample(X, self.y),
            self.model.loglike_per_sample(self.data_spector.exog[:, 0:1],
                                          self.y), decimal=3)
        np.testing.assert_array_almost_equal(
            self.model_col.predict(X),
            self.model.predict(self.data_spector.exog[:, 0:1]), decimal=3)

    def test_lr_disturbed_multicolinearty(self):
        self.model_col = CrossEntropyMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        X = np.hstack([self.data_spector.exog[:, 0:1], self.data_spector.exog[:, 0:1]])
        self.model_col.fit(X,
                           self.y_disturbed, sample_weight=0.5)
        self.model = CrossEntropyMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        self.model.fit(self.data_spector.exog[:, 0:1],
                       self.y_disturbed, sample_weight=0.5)

        np.testing.assert_array_almost_equal(
            self.model_col.coef, np.array([[-9.359,  1.37,  1.37]]), decimal=3)
        # loglike_per_sample
        np.testing.assert_array_almost_equal(
            self.model_col.loglike_per_sample(X, self.y_disturbed),
            self.model.loglike_per_sample(self.data_spector.exog[:, 0:1],
                                          self.y_disturbed), decimal=3)
        np.testing.assert_array_almost_equal(
            self.model_col.predict(X),
            self.model.predict(self.data_spector.exog[:, 0:1]), decimal=3)


class CrossEntropyMNLMultinomialTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_anes96 = sm.datasets.anes96.load()
        cls.y = label_binarize(cls.data_anes96.endog, classes=range(7))
        cls.y_disturbed = (cls.y + 0.01) / 1.07

    def test_label_encoder(self):
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        X_repeated, Y_repeated, sample_weight_repeated = \
            CrossEntropyMNL._label_encoder(x, y)
        np.testing.assert_array_equal(
            X_repeated,
            np.array([
                [1, 2, 3], [1, 2, 3], [1, 2, 3],
                [4, 5, 6], [4, 5, 6], [4, 5, 6],
                [7, 8, 9], [7, 8, 9], [7, 8, 9]]))
        np.testing.assert_array_equal(
            Y_repeated,
            np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]))
        np.testing.assert_array_equal(
            sample_weight_repeated,
            np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]))
        # with sample_weight
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = np.array([[0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5]])
        sample_weight = np.array([0.25, 0.5, 0.25])
        X_repeated, Y_repeated, sample_weight_repeated = \
            CrossEntropyMNL._label_encoder(x, y, sample_weight)
        np.testing.assert_array_equal(
            X_repeated,
            np.array([
                [1, 2, 3], [1, 2, 3], [1, 2, 3],
                [4, 5, 6], [4, 5, 6], [4, 5, 6],
                [7, 8, 9], [7, 8, 9], [7, 8, 9]]))
        np.testing.assert_array_equal(
            Y_repeated,
            np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]))
        np.testing.assert_array_equal(
            sample_weight_repeated,
            np.array([0.125, 0.0625, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.0625, 0.125]))

    def test_lr(self):
        self.model = CrossEntropyMNL(
            solver='newton-cg', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=10, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        self.model.fit(self.data_anes96.exog, self.y)
        # coefficient
        # predict
        self.assertEqual(
            np.sum(self.model.predict(self.data_anes96.exog) ==
                   self.data_anes96.endog), 333)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.data_anes96.exog, self.y),
            -1540.888458338286,
            places=3)
        # to_json
        json_dict = self.model.to_json('./tests/linear_models/crossentropymnl_mlr/')
        self.assertEqual(json_dict['properties']['solver'], 'newton-cg')

        # from_json
        self.model_from_json = CrossEntropyMNL.from_json(json_dict)
        np.testing.assert_array_almost_equal(
            self.model.coef,
            self.model_from_json.coef,
            decimal=3)
        np.testing.assert_array_almost_equal(
            self.model.classes, np.array(range(7)), decimal=3)
        self.assertEqual(self.model.n_classes, 7)

    def test_lr_disturbed(self):
        self.model = CrossEntropyMNL(
            solver='newton-cg', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=10, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        self.model.fit(self.data_anes96.exog, self.y_disturbed)
        # coefficient
        # predict
        self.assertEqual(
            np.sum(self.model.predict(self.data_anes96.exog) ==
                   self.data_anes96.endog), 335)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.data_anes96.exog, self.y_disturbed),
            -1580.5280532302786,
            places=3)

    def test_lr_regularized(self):
        self.model = CrossEntropyMNL(
            solver='newton-cg', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=.5, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        self.model.fit(self.data_anes96.exog, self.y)
        # predict
        self.assertEqual(
            np.sum(self.model.predict(self.data_anes96.exog) ==
                   self.data_anes96.endog), 369)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.data_anes96.exog, self.y),
            -1466.9886103092626,
            places=3)

    def test_lr_disturbed_regularized(self):
        self.model = CrossEntropyMNL(
            solver='newton-cg', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=.5, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        self.model.fit(self.data_anes96.exog, self.y_disturbed)
        # predict
        self.assertEqual(
            np.sum(self.model.predict(self.data_anes96.exog) ==
                   self.data_anes96.endog), 366)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.data_anes96.exog, self.y_disturbed),
            -1519.9521131193064,
            places=3)

    def test_lr_sample_weight_all_half(self):
        self.model_half = CrossEntropyMNL(
            solver='newton-cg', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        self.model_half.fit(self.data_anes96.exog, self.y, sample_weight=.5)
        self.model = CrossEntropyMNL(
            solver='newton-cg', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        self.model.fit(self.data_anes96.exog, self.y)
        # coefficient
        np.testing.assert_array_almost_equal(self.model.coef, self.model_half.coef, decimal=3)
        # predict
        self.assertEqual(
            np.sum(self.model_half.predict(self.data_anes96.exog) ==
                   self.data_anes96.endog), 372)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.data_anes96.exog, self.y, sample_weight=.5),
            -1461.92274725 / 2.,
            places=3)

    def test_lr_disturbed_sample_weight_all_half(self):
        self.model_half = CrossEntropyMNL(
            solver='newton-cg', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        self.model_half.fit(self.data_anes96.exog, self.y_disturbed, sample_weight=.5)
        self.model = CrossEntropyMNL(
            solver='newton-cg', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        self.model.fit(self.data_anes96.exog, self.y_disturbed)
        # coefficient
        np.testing.assert_array_almost_equal(self.model.coef, self.model_half.coef, decimal=3)
        # predict
        self.assertEqual(
            np.sum(self.model_half.predict(self.data_anes96.exog) ==
                   self.data_anes96.endog), 367)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.data_anes96.exog, self.y_disturbed, sample_weight=.5),
            -1516.50148 / 2.,
            places=3)

    def test_lr_sample_weight_all_zero(self):
        self.model = DiscreteMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, classes=None)
        self.model.fit(self.data_anes96.exog, self.y_disturbed, sample_weight=0)
        # coefficient
        self.assertTrue(self.model.coef is None)
        self.assertTrue(self.model.loglike(self.data_anes96.exog, self.y_disturbed) is None)

    def test_lr_sample_weight_half_zero_half_one(self):
        self.model = CrossEntropyMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        len_half = 500
        self.model.fit(self.data_anes96.exog, self.y,
                       sample_weight=np.array([1] * len_half +
                                              [0] * (self.data_anes96.exog.shape[0] - len_half)))
        self.model_half = CrossEntropyMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        self.model_half.fit(self.data_anes96.exog[:len_half], self.y[:len_half])
        # coefficient
        np.testing.assert_array_almost_equal(
            self.model.coef,
            self.model_half.coef,
            decimal=3)

    def test_lr_disturbed_sample_weight_half_zero_half_one(self):
        self.model = CrossEntropyMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        len_half = 500
        self.model.fit(self.data_anes96.exog, self.y_disturbed,
                       sample_weight=np.array([1] * len_half +
                                              [0] * (self.data_anes96.exog.shape[0] - len_half)))
        self.model_half = CrossEntropyMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        self.model_half.fit(self.data_anes96.exog[:len_half], self.y_disturbed[:len_half])
        # coefficient
        np.testing.assert_array_almost_equal(
            self.model.coef,
            self.model_half.coef,
            decimal=3)

    # corner cases
    def test_lr_three_data_point(self):
        # with regularization
        self.model = CrossEntropyMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=.1, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        self.model.fit(self.data_anes96.exog[6:9, :],
                       self.y[6:9, ], sample_weight=0.5)
        # coef
        self.assertEqual(self.model.coef.shape, (7, 6))
        # loglike_per_sample
        np.testing.assert_array_almost_equal(self.model.loglike_per_sample(
            self.data_anes96.exog[6:9, :], self.y[6:9, ]),
            np.array([-0.015, -0.091, -0.095]), decimal=3)
        np.testing.assert_array_almost_equal(self.model.loglike_per_sample(
            self.data_anes96.exog[6:9, :], label_binarize([3, 1, 4], range(7))),
            np.array([-4.201, -5.094, -2.825]), decimal=3)
        np.testing.assert_array_almost_equal(self.model.loglike_per_sample(
            self.data_anes96.exog[6:9, :], label_binarize([3, 0, 5], range(7))),
            np.array([-4.201, -7.352, -8.957]), decimal=3)

    def test_lr_disturbed_three_data_point(self):
        # with regularization
        self.model = CrossEntropyMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=.1, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        self.model.fit(self.data_anes96.exog[6:9, :],
                       self.y_disturbed[6:9, ], sample_weight=0.5)
        # coef
        self.assertEqual(self.model.coef.shape, (7, 6))
        # loglike_per_sample
        np.testing.assert_array_almost_equal(self.model.loglike_per_sample(
            self.data_anes96.exog[6:9, :], self.y_disturbed[6:9, ]),
            np.array([-0.336, -0.389, -0.398]), decimal=3)
        np.testing.assert_array_almost_equal(self.model.loglike_per_sample(
            self.data_anes96.exog[6:9, :], label_binarize([3, 1, 4], range(7))),
            np.array([-3.415, -4.506, -2.367]), decimal=3)
        np.testing.assert_array_almost_equal(self.model.loglike_per_sample(
            self.data_anes96.exog[6:9, :], label_binarize([3, 0, 5], range(7))),
            np.array([-3.415, -4.492, -4.301]), decimal=3)

    def test_lr_multicolinearty(self):
        self.model_col = CrossEntropyMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        X = np.hstack([self.data_anes96.exog[:, 0:1], self.data_anes96.exog[:, 0:1]])
        self.model_col.fit(X,
                           self.y, sample_weight=0.5)
        self.model = CrossEntropyMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        self.model.fit(self.data_anes96.exog[:, 0:1],
                       self.y, sample_weight=0.5)
        # loglike_per_sample
        np.testing.assert_array_almost_equal(
            self.model_col.loglike_per_sample(X, self.y),
            self.model.loglike_per_sample(self.data_anes96.exog[:, 0:1],
                                          self.y), decimal=3)
        np.testing.assert_array_almost_equal(
            self.model_col.predict(X),
            self.model.predict(self.data_anes96.exog[:, 0:1]), decimal=3)

    def test_lr_disturbed_multicolinearty(self):
        self.model_col = CrossEntropyMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        X = np.hstack([self.data_anes96.exog[:, 0:1], self.data_anes96.exog[:, 0:1]])
        self.model_col.fit(X,
                           self.y_disturbed, sample_weight=0.5)
        self.model = CrossEntropyMNL(
            solver='lbfgs', fit_intercept=True, est_stderr=True,
            reg_method='l2',  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None, n_classes=None)
        self.model.fit(self.data_anes96.exog[:, 0:1],
                       self.y_disturbed, sample_weight=0.5)
        # loglike_per_sample
        np.testing.assert_array_almost_equal(
            self.model_col.loglike_per_sample(X, self.y_disturbed),
            self.model.loglike_per_sample(self.data_anes96.exog[:, 0:1],
                                          self.y_disturbed), decimal=3)
        np.testing.assert_array_almost_equal(
            self.model_col.predict(X),
            self.model.predict(self.data_anes96.exog[:, 0:1]), decimal=3)
