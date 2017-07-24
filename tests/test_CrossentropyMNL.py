import unittest


import numpy as np
from sklearn.preprocessing import label_binarize
import statsmodels.api as sm


from IOHMM import DiscreteMNL, CrossEntropyMNL


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
        json_dict = self.model.to_json('./tests/linear_models/CrossentropyMNL/Binary/')
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
        json_dict = self.model.to_json('./tests/linear_models/CrossentropyMNL/Multinomial/')
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
