from __future__ import print_function
from __future__ import division
from past.utils import old_div
import unittest


import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.tests.results.results_glm import InvGauss


from IOHMM import GLM


class PoissonTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = sm.datasets.cpunish.load()
        cls.X = cls.data.exog
        cls.X[:, 3] = np.log(cls.X[:, 3])
        cls.Y = cls.data.endog

    def test_glm_IRLS(self):
        self.model = GLM(
            solver='IRLS', family=sm.families.Poisson(),
            fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.X, self.Y)
        # coefficient
        self.assertEqual(self.model.coef.shape, (7, ))
        np.testing.assert_array_almost_equal(
            self.model.coef,
            np.array((-6.801480e+00, 2.611017e-04, 7.781801e-02, -9.493111e-02, 2.969349e-01,
                      2.301183e+00, -1.872207e+01)),
            decimal=3)
        # std.err of coefficient (calibrated by df_resid)
        self.assertEqual(self.model.stderr.shape, (7, ))
        np.testing.assert_array_almost_equal(
            self.model.stderr,
            np.array((4.146850e+00, 5.187132e-05, 7.940193e-02, 2.291926e-02, 4.375164e-01,
                      4.283826e-01, 4.283961e+00)),
            decimal=2)
        # scale
        self.assertEqual(self.model.dispersion, 1)
        # predict
        np.testing.assert_array_almost_equal(
            self.model.predict(self.X),
            np.array([35.2263655,  8.1965744,  1.3118966,
                      3.6862982,  2.0823003,  1.0650316,  1.9260424,  2.4171405,
                      1.8473219,  2.8643241,  3.1211989,  3.3382067,  2.5269969,
                      0.8972542, 0.9793332,  0.5346209,  1.9790936]),
            decimal=3)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.X, self.Y),
            -31.92732869482515,
            places=3)

        self.assertAlmostEqual(
            self.model.loglike_per_sample(self.X, self.Y).sum(),
            -31.92732869482515,
            places=3)

        self.assertEqual(
            self.model.loglike_per_sample(self.X, self.Y).shape, (17,))
        # to_json
        json_dict = self.model.to_json('./tests/linear_models/GLM/Poisson/')
        self.assertEqual(json_dict['properties']['solver'], 'IRLS')

        # from_json
        self.model_from_json = GLM.from_json(json_dict)
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

        np.testing.assert_array_almost_equal(
            self.model_from_json.predict(self.X),
            np.array([35.2263655,  8.1965744,  1.3118966,
                      3.6862982,  2.0823003,  1.0650316,  1.9260424,  2.4171405,
                      1.8473219,  2.8643241,  3.1211989,  3.3382067,  2.5269969,
                      0.8972542, 0.9793332,  0.5346209,  1.9790936]),
            decimal=3)

    def test_glm_regularized(self):
        # there is a bug in sklearn with weights, it can only use list right now
        self.model = GLM(
            solver='auto', family=sm.families.Poisson(),
            fit_intercept=True, est_stderr=True,
            reg_method='elastic_net',  alpha=0.01, l1_ratio=0.5,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.X, self.Y, sample_weight=0.5)
        self.assertEqual(self.model.coef.shape, (7, ))
        np.testing.assert_array_almost_equal(
            self.model.coef,
            np.array((2.104e-01,   8.331e-05,  -2.736e-02,  -1.347e-01,  -4.327e-02,
                      3.241e+00,  -4.788e+00)),
            decimal=3)
        # std.err of coefficient (calibrated by df_resid)
        self.assertTrue(self.model.stderr is None)
        # scale
        self.assertEqual(self.model.dispersion, 1)
        # predict
        np.testing.assert_array_almost_equal(
            self.model.predict(self.X),
            np.array([23.949,  10.275,   1.12,   7.302,   2.707,   1.585,   0.776,
                      1.894,   3.242,   8.968,   2.265,   1.735,   1.152,   0.202,
                      2.412,   0.952,   3.488]),
            decimal=3)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.X, self.Y),
            -42.636883391983268,
            places=3)

        self.assertAlmostEqual(
            self.model.loglike_per_sample(self.X, self.Y).sum(),
            -42.636883391983268,
            places=3)

        self.assertEqual(
            self.model.loglike_per_sample(self.X, self.Y).shape, (17,))

    def test_glm_sample_weight_all_half(self):
        self.model = GLM(
            solver='IRLS', family=sm.families.Poisson(),
            fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.X, self.Y, sample_weight=0.5)
        # coefficient
        self.assertEqual(self.model.coef.shape, (7, ))
        np.testing.assert_array_almost_equal(
            self.model.coef,
            np.array((-6.801480e+00, 2.611017e-04, 7.781801e-02, -9.493111e-02, 2.969349e-01,
                      2.301183e+00, -1.872207e+01)),
            decimal=3)
        # std.err of coefficient (calibrated by df_resid)
        self.assertEqual(self.model.stderr.shape, (7, ))
        np.testing.assert_array_almost_equal(
            self.model.stderr,
            np.array((5.86e+00,   7.33e-05,   1.12e-01,   3.24e-02,   6.19e-01,
                      6.06e-01,   6.06e+00)),
            decimal=2)
        # scale
        self.assertEqual(self.model.dispersion, 1)
        # predict
        np.testing.assert_array_almost_equal(
            self.model.predict(self.X),
            np.array([35.2263655,  8.1965744,  1.3118966,
                      3.6862982,  2.0823003,  1.0650316,  1.9260424,  2.4171405,
                      1.8473219,  2.8643241,  3.1211989,  3.3382067,  2.5269969,
                      0.8972542, 0.9793332,  0.5346209,  1.9790936]),
            decimal=3)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.X, self.Y, sample_weight=0.5),
            old_div(-31.92732869482515, 2.),
            places=3)

        self.assertAlmostEqual(
            self.model.loglike_per_sample(self.X, self.Y).sum(),
            -31.92732869482515,
            places=3)

        self.assertEqual(
            self.model.loglike_per_sample(self.X, self.Y).shape, (17,))

    def test_glm_sample_weight_all_zero(self):
        self.model = GLM(
            solver='IRLS', family=sm.families.Poisson(),
            fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.assertRaises(ValueError, self.model.fit, self.X, self.Y, 0)

    def test_GLM_sample_weight_half_zero_half_one(self):
        self.model = GLM(
            solver='IRLS', family=sm.families.Poisson(),
            fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        len_half = 8
        self.model.fit(self.X, self.Y,
                       sample_weight=np.array([1] * len_half +
                                              [0] * (self.X.shape[0] - len_half)))
        self.model_half = GLM(
            solver='IRLS', family=sm.families.Poisson(),
            fit_intercept=True, est_stderr=True,
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
            decimal=2)

        # scale
        np.testing.assert_array_almost_equal(
            self.model.dispersion,
            self.model_half.dispersion,
            decimal=3)

    # corner cases
    def test_glm_one_data_point(self):
        self.model = GLM(
            solver='IRLS', family=sm.families.Poisson(),
            fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.X[0:1, :],
                       self.Y[0:1, ], sample_weight=0.5)
        # coef
        self.assertEqual(self.model.coef.shape, (7, ))
        # scale
        self.assertEqual(self.model.dispersion, 1)
        # loglike_per_sample
        np.testing.assert_array_almost_equal(self.model.loglike_per_sample(
            self.X[0:1, :], self.Y[0:1, ]), np.array([-2.72665]), decimal=3)
        np.testing.assert_array_almost_equal(self.model.loglike_per_sample(
            np.array(self.X[0:1, :].tolist() * 6),
            np.array([31, 32, 33, 34, 35, 36])),
            np.array([-3.154, -3.009, -2.894, -2.81, -2.754, -2.727]), decimal=3)

    def test_ols_multicolinearty(self):
        self.model_col = GLM(
            solver='irls', family=sm.families.Poisson(),
            fit_intercept=False, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        X = np.hstack([self.X[:, 0:1], 2 * self.X[:, 0:1]])
        self.model_col.fit(X,
                           self.Y, sample_weight=0.5)
        self.model = GLM(
            solver='IRLS', family=sm.families.Poisson(),
            fit_intercept=False, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.X[:, 0:1],
                       self.Y, sample_weight=0.5)
        # coef
        np.testing.assert_array_almost_equal(
            self.model_col.coef, np.array([8.000e-06, 1.6000e-05]), decimal=3)
        # stderr
        np.testing.assert_array_almost_equal(
            self.model_col.stderr, np.array([9.09531196e-07, 1.81906239e-06]), decimal=3)
        # scale
        np.testing.assert_array_almost_equal(
            self.model_col.dispersion, self.model.dispersion, decimal=3)
        # loglike_per_sample
        np.testing.assert_array_almost_equal(
            self.model_col.loglike_per_sample(X, self.Y),
            self.model.loglike_per_sample(self.X[:, 0:1],
                                          self.Y), decimal=3)
        np.testing.assert_array_almost_equal(
            self.model_col.predict(X),
            self.model.predict(self.X[:, 0:1]), decimal=3)


class GammaTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = sm.datasets.scotland.load()
        cls.X = cls.data.exog
        cls.Y = cls.data.endog

    def test_glm_IRLS(self):
        self.model = GLM(
            solver='IRLS', family=sm.families.Gamma(),
            fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.X, self.Y)
        # coefficient
        self.assertEqual(self.model.coef.shape, (8, ))
        np.testing.assert_array_almost_equal(
            self.model.coef,
            np.array((-1.776527e-02, 4.961768e-05, 2.034423e-03, -7.181429e-05, 1.118520e-04,
                      -1.467515e-07, -5.186831e-04, -2.42717498e-06)),
            decimal=3)
        # std.err of coefficient (calibrated by df_resid)
        self.assertEqual(self.model.stderr.shape, (8, ))
        np.testing.assert_array_almost_equal(
            self.model.stderr * np.sqrt(old_div(32., 24.)),
            np.array((1.147922e-02, 1.621577e-05, 5.320802e-04, 2.711664e-05, 4.057691e-05,
                      1.236569e-07, 2.402534e-04, 7.460253e-07)),
            decimal=2)
        # scale
        self.assertAlmostEqual(self.model.dispersion * 32. / 24., 0.003584283, places=6)
        # predict
        np.testing.assert_array_almost_equal(
            self.model.predict(self.X),
            np.array([57.80431482,  53.2733447, 50.56347993, 58.33003783,
                      70.46562169,  56.88801284,  66.81878401,  66.03410393,
                      57.92937473,  63.23216907,  53.9914785,  61.28993391,
                      64.81036393,  63.47546816,  60.69696114,  74.83508176,
                      56.56991106,  72.01804172,  64.35676519,  52.02445881,
                      64.24933079,  71.15070332,  45.73479688,  54.93318588,
                      66.98031261,  52.02479973,  56.18413736,  58.12267471,
                      67.37947398,  60.49162862,  73.82609217,  69.61515621]),
            decimal=3)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.X, self.Y),
            -82.47352,
            places=2)

        self.assertAlmostEqual(
            self.model.loglike_per_sample(self.X, self.Y).sum(),
            -82.47352,
            places=2)

        self.assertEqual(
            self.model.loglike_per_sample(self.X, self.Y).shape, (32,))
        # to_json
        json_dict = self.model.to_json('./tests/linear_models/GLM/Gamma/')
        self.assertEqual(json_dict['properties']['solver'], 'IRLS')

        # from_json
        self.model_from_json = GLM.from_json(json_dict)
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

        np.testing.assert_array_almost_equal(
            self.model_from_json.predict(self.X),
            np.array([57.80431482,  53.2733447, 50.56347993, 58.33003783,
                      70.46562169,  56.88801284,  66.81878401,  66.03410393,
                      57.92937473,  63.23216907,  53.9914785,  61.28993391,
                      64.81036393,  63.47546816,  60.69696114,  74.83508176,
                      56.56991106,  72.01804172,  64.35676519,  52.02445881,
                      64.24933079,  71.15070332,  45.73479688,  54.93318588,
                      66.98031261,  52.02479973,  56.18413736,  58.12267471,
                      67.37947398,  60.49162862,  73.82609217,  69.61515621]),
            decimal=3)

    def test_glm_regularized(self):
        pass

    def test_glm_sample_weight_all_half(self):
        self.model = GLM(
            solver='IRLS', family=sm.families.Gamma(),
            fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.X, self.Y, sample_weight=0.5)
        # coefficient
        self.assertEqual(self.model.coef.shape, (8, ))
        np.testing.assert_array_almost_equal(
            self.model.coef,
            np.array((-1.776527e-02, 4.961768e-05, 2.034423e-03, -7.181429e-05, 1.118520e-04,
                      -1.467515e-07, -5.186831e-04, -2.42717498e-06)),
            decimal=3)
        # std.err of coefficient (calibrated by df_resid)
        self.assertEqual(self.model.stderr.shape, (8, ))
        np.testing.assert_array_almost_equal(
            self.model.stderr * np.sqrt(32. / 24. / 2.),
            np.array((1.147922e-02, 1.621577e-05, 5.320802e-04, 2.711664e-05, 4.057691e-05,
                      1.236569e-07, 2.402534e-04, 7.460253e-07)),
            decimal=3)
        # scale
        self.assertAlmostEqual(self.model.dispersion * 32. / 24., 0.003584283, places=6)
        # predict
        np.testing.assert_array_almost_equal(
            self.model.predict(self.X),
            np.array([57.80431482,  53.2733447, 50.56347993, 58.33003783,
                      70.46562169,  56.88801284,  66.81878401,  66.03410393,
                      57.92937473,  63.23216907,  53.9914785,  61.28993391,
                      64.81036393,  63.47546816,  60.69696114,  74.83508176,
                      56.56991106,  72.01804172,  64.35676519,  52.02445881,
                      64.24933079,  71.15070332,  45.73479688,  54.93318588,
                      66.98031261,  52.02479973,  56.18413736,  58.12267471,
                      67.37947398,  60.49162862,  73.82609217,  69.61515621]),
            decimal=3)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.X, self.Y, sample_weight=0.5),
            old_div(-82.47352, 2.),
            places=2)

        self.assertAlmostEqual(
            self.model.loglike_per_sample(self.X, self.Y).sum(),
            -82.47352,
            places=2)

        self.assertEqual(
            self.model.loglike_per_sample(self.X, self.Y).shape, (32,))

    def test_glm_sample_weight_all_zero(self):
        self.model = GLM(
            solver='IRLS', family=sm.families.Gamma(),
            fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.assertRaises(ValueError, self.model.fit, self.X, self.Y, 0)

    def test_GLM_sample_weight_half_zero_half_one(self):
        self.model = GLM(
            solver='IRLS', family=sm.families.Gamma(),
            fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        len_half = 16
        self.model.fit(self.X, self.Y,
                       sample_weight=np.array([1] * len_half +
                                              [0] * (self.X.shape[0] - len_half)))
        self.model_half = GLM(
            solver='IRLS', family=sm.families.Gamma(),
            fit_intercept=True, est_stderr=True,
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
    def test_glm_one_data_point(self):
        pass

    def test_ols_multicolinearty(self):
        self.model_col = GLM(
            solver='irls', family=sm.families.Gamma(),
            fit_intercept=False, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        X = np.hstack([self.X[:, 0:1], self.X[:, 0:1]])
        self.model_col.fit(X, self.Y, sample_weight=0.5)
        self.model = GLM(
            solver='IRLS', family=sm.families.Gamma(),
            fit_intercept=False, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.X[:, 0:1], self.Y, sample_weight=0.5)
        # coef
        np.testing.assert_array_almost_equal(
            self.model_col.coef, np.array([1.080e-05,   1.080e-05]), decimal=3)
        # stderr
        # scale
        np.testing.assert_array_almost_equal(
            self.model_col.dispersion, self.model.dispersion, decimal=3)
        # loglike_per_sample
        np.testing.assert_array_almost_equal(
            self.model_col.loglike_per_sample(X, self.Y),
            self.model.loglike_per_sample(self.X[:, 0:1],
                                          self.Y), decimal=3)
        np.testing.assert_array_almost_equal(
            self.model_col.predict(X),
            self.model.predict(self.X[:, 0:1]), decimal=3)


class GaussianTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = sm.datasets.longley.load()
        cls.X = cls.data.exog
        cls.Y = cls.data.endog

    def test_glm_IRLS(self):
        self.model = GLM(
            solver='IRLS', family=sm.families.Gaussian(),
            fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.X, self.Y)
        # coefficient
        self.assertEqual(self.model.coef.shape, (7, ))
        np.testing.assert_array_almost_equal(
            self.model.coef,
            np.array((-3.48225863e+06, 1.50618723e+01,  -3.58191793e-02,
                      -2.02022980e+00, -1.03322687e+00,  -5.11041057e-02,
                      1.82915146e+03)),
            decimal=2)
        # std.err of coefficient (calibrated by df_resid)
        self.assertEqual(self.model.stderr.shape, (7, ))
        np.testing.assert_array_almost_equal(
            self.model.stderr * np.sqrt(old_div(16., 9.)),
            np.array((8.90420384e+05, 8.49149258e+01, 3.34910078e-02, 4.88399682e-01,
                      2.14274163e-01, 2.26073200e-01, 4.55478499e+02)),
            decimal=3)
        # scale
        self.assertAlmostEqual(self.model.dispersion * 16. / 9., 92936.006167311629, places=6)
        # predict
        np.testing.assert_array_almost_equal(
            self.model.predict(self.X),
            np.array([60055.659970240202, 61216.013942398131,
                      60124.71283224225, 61597.114621930756, 62911.285409240052,
                      63888.31121532945, 65153.048956395127, 63774.180356866214,
                      66004.695227399934, 67401.605905447621,
                      68186.268927114084,  66552.055042522494,
                      68810.549973595422, 69649.67130804155, 68989.068486039061,
                      70757.757825193927]),
            decimal=3)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.X, self.Y),
            -109.61743480847952,
            places=3)

        self.assertAlmostEqual(
            self.model.loglike_per_sample(self.X, self.Y).sum(),
            -109.61743480847952,
            places=3)

        self.assertEqual(
            self.model.loglike_per_sample(self.X, self.Y).shape, (16,))
        # to_json
        json_dict = self.model.to_json('./tests/linear_models/GLM/Gaussian/')
        self.assertEqual(json_dict['properties']['solver'], 'IRLS')

        # from_json
        self.model_from_json = GLM.from_json(json_dict)
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

        np.testing.assert_array_almost_equal(
            self.model_from_json.predict(self.X),
            np.array([60055.659970240202, 61216.013942398131,
                      60124.71283224225, 61597.114621930756, 62911.285409240052,
                      63888.31121532945, 65153.048956395127, 63774.180356866214,
                      66004.695227399934, 67401.605905447621,
                      68186.268927114084,  66552.055042522494,
                      68810.549973595422, 69649.67130804155, 68989.068486039061,
                      70757.757825193927]),
            decimal=3)

    def test_glm_regularized(self):
        pass

    def test_glm_sample_weight_all_half(self):
        self.model = GLM(
            solver='IRLS', family=sm.families.Gaussian(),
            fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.X, self.Y, sample_weight=0.5)
        # coefficient
        self.assertEqual(self.model.coef.shape, (7, ))
        np.testing.assert_array_almost_equal(
            self.model.coef,
            np.array((-3.48225863e+06, 1.50618723e+01,  -3.58191793e-02,
                      -2.02022980e+00, -1.03322687e+00,  -5.11041057e-02,
                      1.82915146e+03)),
            decimal=2)
        # std.err of coefficient (calibrated by df_resid)
        self.assertEqual(self.model.stderr.shape, (7, ))
        np.testing.assert_array_almost_equal(
            self.model.stderr * np.sqrt(16. / 9. / 2.),
            np.array((8.90420384e+05, 8.49149258e+01, 3.34910078e-02, 4.88399682e-01,
                      2.14274163e-01, 2.26073200e-01, 4.55478499e+02)),
            decimal=3)
        # scale
        self.assertAlmostEqual(self.model.dispersion * 16. / 9., 92936.006167311629, places=6)
        # predict
        np.testing.assert_array_almost_equal(
            self.model.predict(self.X),
            np.array([60055.659970240202, 61216.013942398131,
                      60124.71283224225, 61597.114621930756, 62911.285409240052,
                      63888.31121532945, 65153.048956395127, 63774.180356866214,
                      66004.695227399934, 67401.605905447621,
                      68186.268927114084,  66552.055042522494,
                      68810.549973595422, 69649.67130804155, 68989.068486039061,
                      70757.757825193927]),
            decimal=3)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.X, self.Y, sample_weight=0.5),
            old_div(-109.61743480847952, 2.),
            places=3)

        self.assertAlmostEqual(
            self.model.loglike_per_sample(self.X, self.Y).sum(),
            -109.61743480847952,
            places=3)

        self.assertEqual(
            self.model.loglike_per_sample(self.X, self.Y).shape, (16,))

    def test_glm_sample_weight_all_zero(self):
        self.model = GLM(
            solver='IRLS', family=sm.families.Gaussian(),
            fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.assertRaises(ValueError, self.model.fit, self.X, self.Y, 0)

    def test_GLM_sample_weight_half_zero_half_one(self):
        self.model = GLM(
            solver='IRLS', family=sm.families.Gaussian(),
            fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        len_half = 8
        self.model.fit(self.X, self.Y,
                       sample_weight=np.array([1] * len_half +
                                              [0] * (self.X.shape[0] - len_half)))
        self.model_half = GLM(
            solver='IRLS', family=sm.families.Gaussian(),
            fit_intercept=True, est_stderr=True,
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
    def test_glm_one_data_point(self):
        pass

    def test_ols_multicolinearty(self):
        self.model_col = GLM(
            solver='irls', family=sm.families.Gaussian(),
            fit_intercept=False, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        X = np.hstack([self.X[:, 0:1], self.X[:, 0:1]])
        self.model_col.fit(X, self.Y, sample_weight=0.5)
        self.model = GLM(
            solver='IRLS', family=sm.families.Gaussian(),
            fit_intercept=False, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.X[:, 0:1], self.Y, sample_weight=0.5)
        # coef
        np.testing.assert_array_almost_equal(
            self.model_col.coef, np.array([319.48,  319.48]), decimal=3)
        # stderr
        # scale
        np.testing.assert_array_almost_equal(
            self.model_col.dispersion, self.model.dispersion, decimal=3)
        # loglike_per_sample
        np.testing.assert_array_almost_equal(
            self.model_col.loglike_per_sample(X, self.Y),
            self.model.loglike_per_sample(self.X[:, 0:1],
                                          self.Y), decimal=3)
        np.testing.assert_array_almost_equal(
            self.model_col.predict(X),
            self.model.predict(self.X[:, 0:1]), decimal=3)


class BinomialTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = sm.datasets.star98.load()
        cls.X = cls.data.exog
        cls.Y = cls.data.endog

    def test_glm_IRLS(self):
        self.model = GLM(
            solver='IRLS', family=sm.families.Binomial(),
            fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.X, self.Y)
        # coefficient
        self.assertEqual(self.model.coef.shape, (21, ))
        np.testing.assert_array_almost_equal(
            self.model.coef,
            np.array((2.9588779262, -0.0168150366,  0.0099254766, -0.0187242148,
                      -0.0142385609, 0.2544871730,  0.2406936644,  0.0804086739,
                      -1.9521605027, -0.3340864748, -0.1690221685,  0.0049167021,
                      -0.0035799644, -0.0140765648, -0.0040049918, -0.0039063958,
                      0.0917143006,  0.0489898381,  0.0080407389,  0.0002220095,
                      -0.0022492486)),
            decimal=3)
        # std.err of coefficient (calibrated by df_resid)
        self.assertEqual(self.model.stderr.shape, (21, ))
        np.testing.assert_array_almost_equal(
            self.model.stderr,
            np.array((1.546712e+00, 4.339467e-04, 6.013714e-04, 7.435499e-04, 4.338655e-04,
                      2.994576e-02, 5.713824e-02, 1.392359e-02, 3.168109e-01,
                      6.126411e-02, 3.270139e-02, 1.253877e-03, 2.254633e-04,
                      1.904573e-03, 4.739838e-04, 9.623650e-04, 1.450923e-02,
                      7.451666e-03, 1.499497e-03, 2.988794e-05, 3.489838e-04)),
            decimal=2)
        # scale
        self.assertEqual(self.model.dispersion, 1)
        # predict
        pred = np.array([0.5833118,  0.75144661,  0.50058272, 0.68534524,  0.32251021,
                         0.68693601,  0.33299827,  0.65624766, 0.49851481,  0.506736,
                         0.23954874,  0.86631452,  0.46432936,  0.44171873,  0.66797935,
                         0.73988491,  0.51966014,  0.42442446,  0.5649369,  0.59251634,
                         0.34798337,  0.56415024,  0.49974355,  0.3565539,  0.20752309,
                         0.18269097,  0.44932642,  0.48025128,  0.59965277,  0.58848671,
                         0.36264203,  0.33333196,  0.74253352,  0.5081886,  0.53421878,
                         0.56291445,  0.60205239,  0.29174423,  0.2954348,  0.32220414,
                         0.47977903,  0.23687535,  0.11776464,  0.1557423,  0.27854799,
                         0.22699533,  0.1819439,  0.32554433,  0.22681989,  0.15785389,
                         0.15268609,  0.61094772,  0.20743222,  0.51649059,  0.46502006,
                         0.41031788,  0.59523288,  0.65733285,  0.27835336,  0.2371213,
                         0.25137045,  0.23953942,  0.27854519,  0.39652413,  0.27023163,
                         0.61411863,  0.2212025,  0.42005842,  0.55940397,  0.35413774,
                         0.45724563,  0.57399437,  0.2168918,  0.58308738,  0.17181104,
                         0.49873249,  0.22832683,  0.14846056,  0.5028073,  0.24513863,
                         0.48202096,  0.52823155,  0.5086262,  0.46295993,  0.57869402,
                         0.78363217,  0.21144435,  0.2298366,  0.17954825,  0.32232586,
                         0.8343015,  0.56217006,  0.47367315,  0.52535649,  0.60350746,
                         0.43210701,  0.44712008,  0.35858239,  0.2521347,  0.19787004,
                         0.63256553,  0.51386532,  0.64997027,  0.13402072,  0.81756174,
                         0.74543642,  0.30825852,  0.23988707,  0.17273125,  0.27880599,
                         0.17395893,  0.32052828,  0.80467697,  0.18726218,  0.23842081,
                         0.19020381,  0.85835388,  0.58703615,  0.72415106,  0.64433695,
                         0.68766653,  0.32923663,  0.16352185,  0.38868816,  0.44980444,
                         0.74810044,  0.42973792,  0.53762581,  0.72714996,  0.61229484,
                         0.30267667,  0.24713253,  0.65086008,  0.48957265,  0.54955545,
                         0.5697156,  0.36406211,  0.48906545,  0.45919413,  0.4930565,
                         0.39785555,  0.5078719,  0.30159626,  0.28524393,  0.34687707,
                         0.22522042,  0.52947159,  0.29277287,  0.8585002,  0.60800389,
                         0.75830521,  0.35648175,  0.69508796,  0.45518355,  0.21567675,
                         0.39682985,  0.49042948,  0.47615798,  0.60588234,  0.62910299,
                         0.46005639,  0.71755165,  0.48852156,  0.47940661,  0.60128813,
                         0.16589699,  0.68512861,  0.46305199,  0.68832227,  0.7006721,
                         0.56564937,  0.51753941,  0.54261733,  0.56072214,  0.34545715,
                         0.30226104,  0.3572956,  0.40996287,  0.33517519,  0.36248407,
                         0.33937041,  0.34140691,  0.2627528,  0.29955161,  0.38581683,
                         0.24840026,  0.15414272,  0.40415991,  0.53936252,  0.52111887,
                         0.28060168,  0.45600958,  0.51110589,  0.43757523,  0.46891953,
                         0.39425249,  0.5834369,  0.55817308,  0.32051259,  0.43567448,
                         0.34134195,  0.43016545,  0.4885413,  0.28478325,  0.2650776,
                         0.46784606,  0.46265983,  0.42655938,  0.18972234,  0.60448491,
                         0.211896,  0.37886032,  0.50727577,  0.39782309,  0.50427121,
                         0.35882898,  0.39596807,  0.49160806,  0.35618002,  0.6819922,
                         0.36871093,  0.43079679,  0.67985516,  0.41270595,  0.68952767,
                         0.52587734,  0.32042126,  0.39120123,  0.56870985,  0.32962349,
                         0.32168989,  0.54076251,  0.4592907,  0.48480182,  0.4408386,
                         0.431178,  0.47078232,  0.55911605,  0.30331618,  0.50310393,
                         0.65036038,  0.45078895,  0.62354291,  0.56435463,  0.50034281,
                         0.52693538,  0.57217285,  0.49221472,  0.40707122,  0.44226533,
                         0.3475959,  0.54746396,  0.86385832,  0.48402233,  0.54313657,
                         0.61586824,  0.27097185,  0.69717808,  0.52156974,  0.50401189,
                         0.56724181,  0.6577178,  0.42732047,  0.44808396,  0.65435634,
                         0.54766225,  0.38160648,  0.49890847,  0.50879037,  0.5875452,
                         0.45101593,  0.5709704,  0.3175516,  0.39813159,  0.28305688,
                         0.40521062,  0.30120578,  0.26400428,  0.44205496,  0.40545798,
                         0.39366599,  0.55288196,  0.14104184,  0.17550155,  0.1949095,
                         0.40255144,  0.21016822,  0.09712017,  0.63151487,  0.25885514,
                         0.57323748,  0.61836898,  0.43268601,  0.67008878,  0.75801989,
                         0.50353406,  0.64222315,  0.29925757,  0.32592036,  0.39634977,
                         0.39582747,  0.41037006,  0.34174944])
        np.testing.assert_array_almost_equal(
            self.model.predict(self.X), pred, decimal=3)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.X, self.Y),
            -2998.61255899391,
            places=3)

        self.assertAlmostEqual(
            self.model.loglike_per_sample(self.X, self.Y).sum(),
            -2998.61255899391,
            places=3)

        self.assertEqual(
            self.model.loglike_per_sample(self.X, self.Y).shape, (303,))

        self.assertEqual(
            self.model.loglike_per_sample(self.X[:5], self.Y[:5]).shape, (5,))
        # to_json
        json_dict = self.model.to_json('./tests/linear_models/GLM/Binomial/')
        self.assertEqual(json_dict['properties']['solver'], 'IRLS')

        # from_json
        self.model_from_json = GLM.from_json(json_dict)
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

        np.testing.assert_array_almost_equal(
            self.model_from_json.predict(self.X), pred, decimal=3)

    def test_glm_regularized(self):
        # not supported by statsmodels
        pass

    def test_glm_sample_weight_all_half(self):
        self.model = GLM(
            solver='IRLS', family=sm.families.Binomial(),
            fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.X, self.Y, sample_weight=0.5)
        # coefficient
        self.assertEqual(self.model.coef.shape, (21, ))
        np.testing.assert_array_almost_equal(
            self.model.coef,
            np.array((2.9588779262, -0.0168150366,  0.0099254766, -0.0187242148,
                      -0.0142385609, 0.2544871730,  0.2406936644,  0.0804086739,
                      -1.9521605027, -0.3340864748, -0.1690221685,  0.0049167021,
                      -0.0035799644, -0.0140765648, -0.0040049918, -0.0039063958,
                      0.0917143006,  0.0489898381,  0.0080407389,  0.0002220095,
                      -0.0022492486)),
            decimal=3)
        # std.err of coefficient (calibrated by df_resid)
        self.assertEqual(self.model.stderr.shape, (21, ))
        np.testing.assert_array_almost_equal(
            old_div(self.model.stderr, np.sqrt(2)),
            np.array((1.546712e+00, 4.339467e-04, 6.013714e-04, 7.435499e-04, 4.338655e-04,
                      2.994576e-02, 5.713824e-02, 1.392359e-02, 3.168109e-01,
                      6.126411e-02, 3.270139e-02, 1.253877e-03, 2.254633e-04,
                      1.904573e-03, 4.739838e-04, 9.623650e-04, 1.450923e-02,
                      7.451666e-03, 1.499497e-03, 2.988794e-05, 3.489838e-04)),
            decimal=2)
        # scale
        self.assertEqual(self.model.dispersion, 1)
        # predict
        pred = np.array([0.5833118,  0.75144661,  0.50058272, 0.68534524,  0.32251021,
                         0.68693601,  0.33299827,  0.65624766, 0.49851481,  0.506736,
                         0.23954874,  0.86631452,  0.46432936,  0.44171873,  0.66797935,
                         0.73988491,  0.51966014,  0.42442446,  0.5649369,  0.59251634,
                         0.34798337,  0.56415024,  0.49974355,  0.3565539,  0.20752309,
                         0.18269097,  0.44932642,  0.48025128,  0.59965277,  0.58848671,
                         0.36264203,  0.33333196,  0.74253352,  0.5081886,  0.53421878,
                         0.56291445,  0.60205239,  0.29174423,  0.2954348,  0.32220414,
                         0.47977903,  0.23687535,  0.11776464,  0.1557423,  0.27854799,
                         0.22699533,  0.1819439,  0.32554433,  0.22681989,  0.15785389,
                         0.15268609,  0.61094772,  0.20743222,  0.51649059,  0.46502006,
                         0.41031788,  0.59523288,  0.65733285,  0.27835336,  0.2371213,
                         0.25137045,  0.23953942,  0.27854519,  0.39652413,  0.27023163,
                         0.61411863,  0.2212025,  0.42005842,  0.55940397,  0.35413774,
                         0.45724563,  0.57399437,  0.2168918,  0.58308738,  0.17181104,
                         0.49873249,  0.22832683,  0.14846056,  0.5028073,  0.24513863,
                         0.48202096,  0.52823155,  0.5086262,  0.46295993,  0.57869402,
                         0.78363217,  0.21144435,  0.2298366,  0.17954825,  0.32232586,
                         0.8343015,  0.56217006,  0.47367315,  0.52535649,  0.60350746,
                         0.43210701,  0.44712008,  0.35858239,  0.2521347,  0.19787004,
                         0.63256553,  0.51386532,  0.64997027,  0.13402072,  0.81756174,
                         0.74543642,  0.30825852,  0.23988707,  0.17273125,  0.27880599,
                         0.17395893,  0.32052828,  0.80467697,  0.18726218,  0.23842081,
                         0.19020381,  0.85835388,  0.58703615,  0.72415106,  0.64433695,
                         0.68766653,  0.32923663,  0.16352185,  0.38868816,  0.44980444,
                         0.74810044,  0.42973792,  0.53762581,  0.72714996,  0.61229484,
                         0.30267667,  0.24713253,  0.65086008,  0.48957265,  0.54955545,
                         0.5697156,  0.36406211,  0.48906545,  0.45919413,  0.4930565,
                         0.39785555,  0.5078719,  0.30159626,  0.28524393,  0.34687707,
                         0.22522042,  0.52947159,  0.29277287,  0.8585002,  0.60800389,
                         0.75830521,  0.35648175,  0.69508796,  0.45518355,  0.21567675,
                         0.39682985,  0.49042948,  0.47615798,  0.60588234,  0.62910299,
                         0.46005639,  0.71755165,  0.48852156,  0.47940661,  0.60128813,
                         0.16589699,  0.68512861,  0.46305199,  0.68832227,  0.7006721,
                         0.56564937,  0.51753941,  0.54261733,  0.56072214,  0.34545715,
                         0.30226104,  0.3572956,  0.40996287,  0.33517519,  0.36248407,
                         0.33937041,  0.34140691,  0.2627528,  0.29955161,  0.38581683,
                         0.24840026,  0.15414272,  0.40415991,  0.53936252,  0.52111887,
                         0.28060168,  0.45600958,  0.51110589,  0.43757523,  0.46891953,
                         0.39425249,  0.5834369,  0.55817308,  0.32051259,  0.43567448,
                         0.34134195,  0.43016545,  0.4885413,  0.28478325,  0.2650776,
                         0.46784606,  0.46265983,  0.42655938,  0.18972234,  0.60448491,
                         0.211896,  0.37886032,  0.50727577,  0.39782309,  0.50427121,
                         0.35882898,  0.39596807,  0.49160806,  0.35618002,  0.6819922,
                         0.36871093,  0.43079679,  0.67985516,  0.41270595,  0.68952767,
                         0.52587734,  0.32042126,  0.39120123,  0.56870985,  0.32962349,
                         0.32168989,  0.54076251,  0.4592907,  0.48480182,  0.4408386,
                         0.431178,  0.47078232,  0.55911605,  0.30331618,  0.50310393,
                         0.65036038,  0.45078895,  0.62354291,  0.56435463,  0.50034281,
                         0.52693538,  0.57217285,  0.49221472,  0.40707122,  0.44226533,
                         0.3475959,  0.54746396,  0.86385832,  0.48402233,  0.54313657,
                         0.61586824,  0.27097185,  0.69717808,  0.52156974,  0.50401189,
                         0.56724181,  0.6577178,  0.42732047,  0.44808396,  0.65435634,
                         0.54766225,  0.38160648,  0.49890847,  0.50879037,  0.5875452,
                         0.45101593,  0.5709704,  0.3175516,  0.39813159,  0.28305688,
                         0.40521062,  0.30120578,  0.26400428,  0.44205496,  0.40545798,
                         0.39366599,  0.55288196,  0.14104184,  0.17550155,  0.1949095,
                         0.40255144,  0.21016822,  0.09712017,  0.63151487,  0.25885514,
                         0.57323748,  0.61836898,  0.43268601,  0.67008878,  0.75801989,
                         0.50353406,  0.64222315,  0.29925757,  0.32592036,  0.39634977,
                         0.39582747,  0.41037006,  0.34174944])
        np.testing.assert_array_almost_equal(
            self.model.predict(self.X), pred, decimal=3)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.X, self.Y, sample_weight=0.5),
            old_div(-2998.61255899391, 2.),
            places=3)

        self.assertAlmostEqual(
            self.model.loglike_per_sample(self.X, self.Y).sum(),
            -2998.61255899391,
            places=3)

        self.assertEqual(
            self.model.loglike_per_sample(self.X, self.Y).shape, (303,))

        self.assertEqual(
            self.model.loglike_per_sample(self.X[:5], self.Y[:5]).shape, (5,))

    def test_glm_sample_weight_all_zero(self):
        self.model = GLM(
            solver='IRLS', family=sm.families.Binomial(),
            fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.assertRaises(ValueError, self.model.fit, self.X, self.Y, 0)

    def test_GLM_sample_weight_half_zero_half_one(self):
        self.model = GLM(
            solver='IRLS', family=sm.families.Binomial(),
            fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        len_half = 160
        self.model.fit(self.X, self.Y,
                       sample_weight=np.array([1] * len_half +
                                              [0] * (self.X.shape[0] - len_half)))
        self.model_half = GLM(
            solver='IRLS', family=sm.families.Binomial(),
            fit_intercept=True, est_stderr=True,
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
            decimal=2)

        # scale
        np.testing.assert_array_almost_equal(
            self.model.dispersion,
            self.model_half.dispersion,
            decimal=3)

    # corner cases

    def test_glm_one_data_point(self):
        self.model = GLM(
            solver='IRLS', family=sm.families.Binomial(),
            fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.X[0:1, :],
                       self.Y[0:1, ], sample_weight=0.5)
        # coef
        self.assertEqual(self.model.coef.shape, (21, ))
        # scale
        self.assertEqual(self.model.dispersion, 1)
        # loglike_per_sample
        np.testing.assert_array_almost_equal(self.model.loglike_per_sample(
            self.X[0:1, :], self.Y[0:1, ]), np.array([-3.565]), decimal=3)
        np.testing.assert_array_almost_equal(self.model.loglike_per_sample(
            np.array(self.X[0:1, :].tolist() * 6),
            np.array([[452., 355.], [510., 235.], [422., 335.],
                      [454., 355.], [452., 355.], [422., 355.]])),
            np.array([-3.565, -27.641,  -3.545,  -3.568,  -3.565,  -4.004]), decimal=3)

    def test_ols_multicolinearty(self):
        self.model_col = GLM(
            solver='irls', family=sm.families.Binomial(),
            fit_intercept=False, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        X = np.hstack([self.X[:, 0:1], self.X[:, 0:1]])
        self.model_col.fit(X, self.Y, sample_weight=0.5)
        self.model = GLM(
            solver='IRLS', family=sm.families.Binomial(),
            fit_intercept=False, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.X[:, 0:1], self.Y, sample_weight=0.5)
        # coef
        np.testing.assert_array_almost_equal(
            self.model_col.coef, np.array([-0.006, -0.006]), decimal=3)
        # stderr
        np.testing.assert_array_almost_equal(
            self.model_col.stderr, np.array([5.684e-05, 5.684e-05]), decimal=3)
        # scale
        np.testing.assert_array_almost_equal(
            self.model_col.dispersion, self.model.dispersion, decimal=3)
        # loglike_per_sample
        np.testing.assert_array_almost_equal(
            self.model_col.loglike_per_sample(X, self.Y),
            self.model.loglike_per_sample(self.X[:, 0:1],
                                          self.Y), decimal=3)
        np.testing.assert_array_almost_equal(
            self.model_col.predict(X),
            self.model.predict(self.X[:, 0:1]), decimal=3)


class InverseGaussianTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        filename = 'tests/linear_models/GLM/InverseGaussian/inv_gaussian.csv'
        data = np.genfromtxt(open(filename, 'rb'), delimiter=",", dtype=float)[1:]
        cls.Y = data[:5000, 0]
        cls.X = data[:5000, 1:]
        cls.res = InvGauss()

    def test_glm_IRLS(self):
        self.model = GLM(
            solver='IRLS', family=sm.families.InverseGaussian(),
            fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.X, self.Y)
        # coefficient
        self.assertEqual(self.model.coef.shape, (3, ))
        np.testing.assert_array_almost_equal(
            self.model.coef,
            np.array((1.0359574, 0.4519770, -0.2508288)),
            decimal=3)
        # std.err of coefficient (calibrated by df_resid)
        self.assertEqual(self.model.stderr.shape, (3, ))
        np.testing.assert_array_almost_equal(
            self.model.stderr * np.sqrt(old_div(5000., 4997.)),
            np.array((0.03429943, 0.03148291, 0.02237211)),
            decimal=3)
        # scale
        self.assertAlmostEqual(self.model.dispersion * 5000. / 4997., 0.2867266359127567, places=6)
        # predict
        np.testing.assert_array_almost_equal(
            self.model.predict(self.X),
            self.res.fittedvalues,
            decimal=3)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.X, self.Y),
            -2525.70955823223,
            places=1)

        self.assertAlmostEqual(
            self.model.loglike_per_sample(self.X, self.Y).sum(),
            -2525.70955823223,
            places=1)

        self.assertEqual(
            self.model.loglike_per_sample(self.X, self.Y).shape, (5000,))
        # to_json
        json_dict = self.model.to_json('./tests/linear_models/GLM/InverseGaussian/')
        self.assertEqual(json_dict['properties']['solver'], 'IRLS')

        # from_json
        self.model_from_json = GLM.from_json(json_dict)
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

        np.testing.assert_array_almost_equal(
            self.model_from_json.predict(self.X),
            self.res.fittedvalues,
            decimal=3)

    def test_glm_regularized(self):
        pass

    def test_glm_sample_weight_all_half(self):
        self.model = GLM(
            solver='IRLS', family=sm.families.InverseGaussian(),
            fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.X, self.Y, sample_weight=0.5)
        # coefficient
        self.assertEqual(self.model.coef.shape, (3, ))
        np.testing.assert_array_almost_equal(
            self.model.coef,
            np.array((1.0359574, 0.4519770, -0.2508288)),
            decimal=3)
        # std.err of coefficient (calibrated by df_resid)
        self.assertEqual(self.model.stderr.shape, (3, ))
        np.testing.assert_array_almost_equal(
            self.model.stderr * np.sqrt(5000. / 4997. / 2.),
            np.array((0.03429943, 0.03148291, 0.02237211)),
            decimal=3)
        # scale
        self.assertAlmostEqual(self.model.dispersion * 5000. / 4997., 0.2867266359127567, places=6)
        # predict
        np.testing.assert_array_almost_equal(
            self.model.predict(self.X),
            self.res.fittedvalues,
            decimal=3)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.X, self.Y, sample_weight=0.5),
            old_div(-2525.70955823223, 2.),
            places=1)

        self.assertAlmostEqual(
            self.model.loglike_per_sample(self.X, self.Y).sum(),
            -2525.70955823223,
            places=1)

        self.assertEqual(
            self.model.loglike_per_sample(self.X, self.Y).shape, (5000,))

    def test_glm_sample_weight_all_zero(self):
        self.model = GLM(
            solver='IRLS', family=sm.families.InverseGaussian(),
            fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.assertRaises(ValueError, self.model.fit, self.X, self.Y, 0)

    def test_GLM_sample_weight_half_zero_half_one(self):
        self.model = GLM(
            solver='IRLS', family=sm.families.InverseGaussian(),
            fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        len_half = 2500
        self.model.fit(self.X, self.Y,
                       sample_weight=np.array([1] * len_half +
                                              [0] * (self.X.shape[0] - len_half)))
        self.model_half = GLM(
            solver='IRLS', family=sm.families.InverseGaussian(),
            fit_intercept=True, est_stderr=True,
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

    def test_glm_one_data_point(self):
        pass

    def test_ols_multicolinearty(self):
        self.model_col = GLM(
            solver='irls', family=sm.families.InverseGaussian(),
            fit_intercept=False, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        X = np.hstack([self.X[:, 0:1], self.X[:, 0:1]])
        self.model_col.fit(X, self.Y, sample_weight=0.5)
        self.model = GLM(
            solver='IRLS', family=sm.families.InverseGaussian(),
            fit_intercept=False, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.X[:, 0:1], self.Y, sample_weight=0.5)
        # coef
        np.testing.assert_array_almost_equal(
            self.model_col.coef, np.array([0.712,  0.712]), decimal=3)
        # stderr
        # scale
        np.testing.assert_array_almost_equal(
            self.model_col.dispersion, self.model.dispersion, decimal=3)
        # loglike_per_sample
        np.testing.assert_array_almost_equal(
            self.model_col.loglike_per_sample(X, self.Y),
            self.model.loglike_per_sample(self.X[:, 0:1],
                                          self.Y), decimal=3)
        np.testing.assert_array_almost_equal(
            self.model_col.predict(X),
            self.model.predict(self.X[:, 0:1]), decimal=3)


class NegativeBinomialTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data = sm.datasets.committee.load()
        data.exog[:, 2] = np.log(data.exog[:, 2])
        interaction = data.exog[:, 2] * data.exog[:, 1]
        data.exog = np.column_stack((data.exog, interaction))

        cls.Y = data.endog
        cls.X = data.exog

    def test_glm_IRLS(self):
        self.model = GLM(
            solver='IRLS', family=sm.families.NegativeBinomial(),
            fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.X, self.Y)
        # coefficient
        self.assertEqual(self.model.coef.shape, (7, ))
        np.testing.assert_array_almost_equal(
            self.model.coef,
            np.array([-6.44847076, -0.0268147,  1.25103364,  2.91070663,
                      -0.34799563,  0.00659808, -0.31303026]),
            decimal=2)
        # std.err of coefficient (calibrated by df_resid)
        self.assertEqual(self.model.stderr.shape, (7, ))
        np.testing.assert_array_almost_equal(
            self.model.stderr,
            np.array([3.21429775e+00, 3.22130435e-02, 7.68090529e-01,
                      1.04436390e+00, 6.73309516e-01, 2.27984343e-03, 1.73596557e-01]),
            decimal=3)
        # scale
        self.assertEqual(self.model.dispersion, 1)
        # predict
        np.testing.assert_array_almost_equal(
            self.model.predict(self.X),
            np.array([12.62019383,  30.18289514, 21.48377849, 496.74068604,
                      103.23024673,  219.94693494,  324.4301163,  110.82526477,
                      112.44244488,  219.86056381,   56.84399998,   61.19840382,
                      114.09290269,   75.29071944,   61.21994387,   21.05130889,
                      42.75939828,   55.56133536,    0.72532053,   18.14664665]),
            decimal=0)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.X, self.Y),
            -101.33286676188968,
            places=1)

        self.assertAlmostEqual(
            self.model.loglike_per_sample(self.X, self.Y).sum(),
            -101.33286676188968,
            places=1)

        self.assertEqual(
            self.model.loglike_per_sample(self.X, self.Y).shape, (20,))
        # to_json
        json_dict = self.model.to_json('./tests/linear_models/GLM/NegativeBinomial/')
        self.assertEqual(json_dict['properties']['solver'], 'IRLS')

        # from_json
        self.model_from_json = GLM.from_json(json_dict)
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

        np.testing.assert_array_almost_equal(
            self.model_from_json.predict(self.X),
            np.array([12.62019383,  30.18289514, 21.48377849, 496.74068604,
                      103.23024673,  219.94693494,  324.4301163,  110.82526477,
                      112.44244488,  219.86056381,   56.84399998,   61.19840382,
                      114.09290269,   75.29071944,   61.21994387,   21.05130889,
                      42.75939828,   55.56133536,    0.72532053,   18.14664665]),
            decimal=0)

    def test_glm_regularized(self):
        pass

    def test_glm_sample_weight_all_half(self):
        self.model = GLM(
            solver='IRLS', family=sm.families.NegativeBinomial(),
            fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.X, self.Y, sample_weight=0.5)
        # coefficient
        self.assertEqual(self.model.coef.shape, (7, ))
        np.testing.assert_array_almost_equal(
            self.model.coef,
            np.array([-6.44847076, -0.0268147,  1.25103364,  2.91070663,
                      -0.34799563,  0.00659808, -0.31303026]),
            decimal=2)
        # std.err of coefficient (calibrated by df_resid)
        self.assertEqual(self.model.stderr.shape, (7, ))
        np.testing.assert_array_almost_equal(
            self.model.stderr,
            np.array([4.54570348e+00, 4.55561229e-02, 1.08624404e+00,
                      1.47695359e+00, 9.52203449e-01, 3.22418550e-03, 2.45502605e-01]),
            decimal=3)
        # scale
        self.assertAlmostEqual(self.model.dispersion, 1, places=4)
        # predict
        np.testing.assert_array_almost_equal(
            self.model.predict(self.X),
            np.array([12.62019383,  30.18289514, 21.48377849, 496.74068604,
                      103.23024673,  219.94693494,  324.4301163,  110.82526477,
                      112.44244488,  219.86056381,   56.84399998,   61.19840382,
                      114.09290269,   75.29071944,   61.21994387,   21.05130889,
                      42.75939828,   55.56133536,    0.72532053,   18.14664665]),
            decimal=0)
        # loglike/_per_sample
        self.assertAlmostEqual(
            self.model.loglike(self.X, self.Y, sample_weight=0.5),
            old_div(-101.33286676188968, 2.),
            places=1)

        self.assertAlmostEqual(
            self.model.loglike_per_sample(self.X, self.Y).sum(),
            -101.33286676188968,
            places=1)

        self.assertEqual(
            self.model.loglike_per_sample(self.X, self.Y).shape, (20,))

    def test_glm_sample_weight_all_zero(self):
        self.model = GLM(
            solver='IRLS', family=sm.families.NegativeBinomial(),
            fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.assertRaises(ValueError, self.model.fit, self.X, self.Y, 0)

    def test_GLM_sample_weight_half_zero_half_one(self):
        self.model = GLM(
            solver='IRLS', family=sm.families.NegativeBinomial(),
            fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        len_half = 10
        self.model.fit(self.X, self.Y,
                       sample_weight=np.array([1] * len_half +
                                              [0] * (self.X.shape[0] - len_half)))
        self.model_half = GLM(
            solver='IRLS', family=sm.families.NegativeBinomial(),
            fit_intercept=True, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model_half.fit(self.X[:len_half], self.Y[:len_half])
        # coefficient
        np.testing.assert_array_almost_equal(
            self.model.coef,
            self.model_half.coef,
            decimal=2)
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
    def test_glm_one_data_point(self):
        pass

    def test_ols_multicolinearty(self):
        self.model_col = GLM(
            solver='irls', family=sm.families.NegativeBinomial(),
            fit_intercept=False, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        X = np.hstack([self.X[:, 0:1], self.X[:, 0:1]])
        self.model_col.fit(X, self.Y, sample_weight=0.5)
        self.model = GLM(
            solver='IRLS', family=sm.families.NegativeBinomial(),
            fit_intercept=False, est_stderr=True,
            reg_method=None,  alpha=0, l1_ratio=0,  tol=1e-4, max_iter=100,
            coef=None, stderr=None,  dispersion=None)
        self.model.fit(self.X[:, 0:1], self.Y, sample_weight=0.5)
        # coef
        np.testing.assert_array_almost_equal(
            self.model_col.coef, np.array([0.059,  0.059]), decimal=3)
        # stderr
        # scale
        np.testing.assert_array_almost_equal(
            self.model_col.dispersion, self.model.dispersion, decimal=3)
        # loglike_per_sample
        np.testing.assert_array_almost_equal(
            self.model_col.loglike_per_sample(X, self.Y),
            self.model.loglike_per_sample(self.X[:, 0:1],
                                          self.Y), decimal=3)
        np.testing.assert_array_almost_equal(
            self.model_col.predict(X),
            self.model.predict(self.X[:, 0:1]), decimal=3)
