import unittest


import numpy as np
import statsmodels.api as sm


from IOHMM import DiscreteMNL


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
