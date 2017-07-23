# import unittest

# import numpy as np
# from statsmodels.genmod.families.family import (Poisson,
#                                                 Gaussian,
#                                                 Gamma,
#                                                 Binomial,
#                                                 InverseGaussian,
#                                                 NegativeBinomial)

# from IOHMM import (ForwardingPoisson,
#                    ForwardingGaussian,
#                    ForwardingGamma,
#                    ForwardingBinomial,
#                    ForwardingInverseGaussian,
#                    ForwardingNegativeBinomial)


# # Things to test
# # 1. dimension
# # 2. sum equal to loglike is sample_weight is all ones
# # 3. when scale is zero


# class ForwardingPoissonTests(unittest.TestCase):

#     @classmethod
#     def setUpClass(cls):
#         cls.forwarding_family = ForwardingPoisson()
#         cls.family = Poisson()

#     def test_init(self):
#         self.assertEqual(self.forwarding_family.family.variance, self.family.variance)

#     def test_starting_mu(self):
#         y = np.arange(0)
#         np.testing.assert_array_equal(
#             self.forwarding_family.starting_mu(y), self.family.starting_mu(y))
#         y = np.arange(1)
#         np.testing.assert_array_equal(
#             self.forwarding_family.starting_mu(y), self.family.starting_mu(y))
#         y = np.arange(10)
#         np.testing.assert_array_equal(
#             self.forwarding_family.starting_mu(y), self.family.starting_mu(y))
#         y = np.random.rand(100,)
#         np.testing.assert_array_equal(
#             self.forwarding_family.starting_mu(y), self.family.starting_mu(y))

#     def test_weights(self):
#         mu = np.arange(0)
#         np.testing.assert_array_equal(
#             self.forwarding_family.weights(mu), self.family.weights(mu))
#         mu = np.arange(1)
#         np.testing.assert_array_equal(
#             self.forwarding_family.weights(mu), self.family.weights(mu))
#         mu = np.arange(10)
#         np.testing.assert_array_equal(
#             self.forwarding_family.weights(mu), self.family.weights(mu))
#         mu = np.random.rand(100,)
#         np.testing.assert_array_equal(
#             self.forwarding_family.weights(mu), self.family.weights(mu))

#     def test_deviance(self):
#         endog = np.arange(1, 11)
#         mu = np.arange(1, 11)
#         freq_weights = 1.
#         scale = 1.
#         self.assertEqual(
#             self.forwarding_family.deviance(endog, mu, freq_weights, scale),
#             self.family.deviance(endog, mu, freq_weights, scale))
#         freq_weights = np.random.rand(10)
#         self.assertEqual(
#             self.forwarding_family.deviance(endog, mu, freq_weights, scale),
#             self.family.deviance(endog, mu, freq_weights, scale))
#         mu = np.random.rand(10)
#         self.assertEqual(
#             self.forwarding_family.deviance(endog, mu, freq_weights, scale),
#             self.family.deviance(endog, mu, freq_weights, scale))
#         endog = np.arange(1, 2)
#         mu = np.arange(1, 2)
#         self.assertEqual(
#             self.forwarding_family.deviance(endog, mu, freq_weights, scale),
#             self.family.deviance(endog, mu, freq_weights, scale))
#         endog = np.arange(0)
#         mu = np.arange(0)
#         freq_weights = np.arange(0)
#         self.assertEqual(
#             self.forwarding_family.deviance(endog, mu, freq_weights, scale),
#             self.family.deviance(endog, mu, freq_weights, scale))

#     def test_resid_dev(self):
#         endog = np.arange(1, 11)
#         mu = np.arange(1, 11)
#         scale = 1.
#         np.testing.assert_array_equal(
#             self.forwarding_family.resid_dev(endog, mu, scale),
#             self.family.resid_dev(endog, mu, scale))
#         np.testing.assert_array_equal(
#             self.forwarding_family.resid_dev(endog, mu, scale),
#             self.family.resid_dev(endog, mu, scale))
#         mu = np.random.rand(10)
#         np.testing.assert_array_equal(
#             self.forwarding_family.resid_dev(endog, mu, scale),
#             self.family.resid_dev(endog, mu, scale))
#         endog = np.arange(1, 2)
#         mu = np.arange(1, 2)
#         np.testing.assert_array_equal(
#             self.forwarding_family.resid_dev(endog, mu, scale),
#             self.family.resid_dev(endog, mu, scale))
#         endog = np.arange(0)
#         mu = np.arange(0)
#         np.testing.assert_array_equal(
#             self.forwarding_family.resid_dev(endog, mu, scale),
#             self.family.resid_dev(endog, mu, scale))

#     def test_fitted(self):
#         lin_pred = np.arange(0)
#         np.testing.assert_array_equal(
#             self.forwarding_family.fitted(lin_pred), self.family.fitted(lin_pred))
#         lin_pred = np.arange(1)
#         np.testing.assert_array_equal(
#             self.forwarding_family.fitted(lin_pred), self.family.fitted(lin_pred))
#         lin_pred = np.arange(10)
#         np.testing.assert_array_equal(
#             self.forwarding_family.fitted(lin_pred), self.family.fitted(lin_pred))
#         lin_pred = np.random.rand(100,)
#         np.testing.assert_array_equal(
#             self.forwarding_family.fitted(lin_pred), self.family.fitted(lin_pred))

#     def test_predict(self):
#         mu = np.arange(0)
#         np.testing.assert_array_equal(
#             self.forwarding_family.predict(mu), self.family.predict(mu))
#         mu = np.arange(1)
#         np.testing.assert_array_equal(
#             self.forwarding_family.predict(mu), self.family.predict(mu))
#         mu = np.arange(10)
#         np.testing.assert_array_equal(
#             self.forwarding_family.predict(mu), self.family.predict(mu))
#         mu = np.random.rand(100,)
#         np.testing.assert_array_equal(
#             self.forwarding_family.predict(mu), self.family.predict(mu))

#     def test_loglike(self):
#         endog = np.arange(1, 11)
#         mu = np.arange(1, 11)
#         freq_weights = 1.
#         scale = 1.
#         self.assertEqual(
#             self.forwarding_family.loglike(endog, mu, freq_weights, scale),
#             self.family.loglike(endog, mu, freq_weights, scale))
#         freq_weights = np.random.rand(10)
#         self.assertEqual(
#             self.forwarding_family.loglike(endog, mu, freq_weights, scale),
#             self.family.loglike(endog, mu, freq_weights, scale))
#         mu = np.random.rand(10)
#         self.assertEqual(
#             self.forwarding_family.loglike(endog, mu, freq_weights, scale),
#             self.family.loglike(endog, mu, freq_weights, scale))
#         endog = np.arange(1, 2)
#         mu = np.arange(1, 2)
#         self.assertEqual(
#             self.forwarding_family.loglike(endog, mu, freq_weights, scale),
#             self.family.loglike(endog, mu, freq_weights, scale))
#         endog = np.arange(0)
#         mu = np.arange(0)
#         freq_weights = np.arange(0)
#         self.assertEqual(
#             self.forwarding_family.loglike(endog, mu, freq_weights, scale),
#             self.family.loglike(endog, mu, freq_weights, scale))

#     def test_resid_anscombe(self):
#         endog = np.arange(10)
#         mu = np.arange(10)
#         np.testing.assert_array_equal(
#             self.forwarding_family.resid_dev(endog, mu),
#             self.family.resid_dev(endog, mu))
#         np.testing.assert_array_equal(
#             self.forwarding_family.resid_dev(endog, mu),
#             self.family.resid_dev(endog, mu))
#         mu = np.random.rand(10)
#         np.testing.assert_array_equal(
#             self.forwarding_family.resid_dev(endog, mu),
#             self.family.resid_dev(endog, mu))
#         endog = np.arange(1)
#         mu = np.arange(1)
#         np.testing.assert_array_equal(
#             self.forwarding_family.resid_dev(endog, mu),
#             self.family.resid_dev(endog, mu))
#         endog = np.arange(0)
#         mu = np.arange(0)
#         np.testing.assert_array_equal(
#             self.forwarding_family.resid_dev(endog, mu),
#             self.family.resid_dev(endog, mu))

#     def test_loglike_per_sample(self):
#         endog = np.arange(1, 11)
#         mu = np.arange(1, 11)
#         scale = 1.
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         self.assertEqual(
#             np.sum(lps),
#             self.family.loglike(endog, mu, 1., scale))
#         mu = np.random.rand(10)
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         self.assertEqual(
#             np.sum(lps),
#             self.family.loglike(endog, mu, 1., scale))
#         endog = np.arange(1, 2)
#         mu = np.arange(1, 2)
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         self.assertEqual(
#             np.sum(lps),
#             self.family.loglike(endog, mu, 1., scale))
#         endog = np.arange(0)
#         mu = np.arange(0)
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         self.assertEqual(
#             np.sum(lps),
#             self.family.loglike(endog, mu, 1., scale))


# class ForwardingGaussianTests(unittest.TestCase):

#     @classmethod
#     def setUpClass(cls):
#         cls.forwarding_family = ForwardingGaussian()
#         cls.family = Gaussian()

#     def test_init(self):
#         self.assertEqual(self.forwarding_family.family.variance, self.family.variance)

#     def test_loglike_per_sample(self):
#         endog = np.arange(1, 11)
#         mu = np.arange(1, 11)
#         scale = 1.
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         np.testing.assert_array_almost_equal(lps, np.array([-0.918939] * 10), decimal=6)
#         endog = np.arange(1, 2)
#         mu = np.arange(1, 2)
#         scale = 5.
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         np.testing.assert_array_almost_equal(lps, np.array([-1.723657]), decimal=6)
#         endog = np.arange(0)
#         mu = np.arange(0)
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         np.testing.assert_array_equal(lps, np.array([]))

#         endog = np.arange(1, 11)
#         mu = np.arange(1, 11)
#         scale = 0.
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         np.testing.assert_array_equal(lps, np.array([0] * 10))
#         mu = np.array([1, 2, 3, 4, 5, 7, 8, 9, 10, 11])
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         np.testing.assert_array_equal(lps, np.array([0] * 5 + [-np.Infinity] * 5))

#         endog = np.arange(0)
#         mu = np.arange(0)
#         scale = 0.
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         np.testing.assert_array_equal(lps, np.array([]))


# class ForwardingGammaTests(unittest.TestCase):

#     @classmethod
#     def setUpClass(cls):
#         cls.forwarding_family = ForwardingGamma()
#         cls.family = Gamma()

#     def test_init(self):
#         self.assertEqual(self.forwarding_family.family.variance, self.family.variance)

#     def test_loglike_per_sample(self):
#         endog = np.arange(1, 11)
#         mu = np.arange(1, 11)
#         scale = 1.
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         self.assertEqual(
#             np.sum(lps),
#             self.family.loglike(endog, mu, 1., scale))
#         mu = np.random.rand(10)
#         scale = 2.
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         self.assertEqual(
#             np.sum(lps),
#             self.family.loglike(endog, mu, 1., scale))
#         endog = np.arange(1, 2)
#         mu = np.arange(1, 2)
#         scale = 5.
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         self.assertEqual(
#             np.sum(lps),
#             self.family.loglike(endog, mu, 1., scale))
#         endog = np.arange(0)
#         mu = np.arange(0)
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         self.assertEqual(
#             np.sum(lps),
#             self.family.loglike(endog, mu, 1., scale))

#         endog = np.arange(1, 11)
#         mu = np.arange(1, 11)
#         scale = 0.
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         np.testing.assert_array_equal(lps, np.array([0] * 10))
#         mu = np.array([1, 2, 3, 4, 5, 7, 8, 9, 10, 11])
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         np.testing.assert_array_equal(lps, np.array([0] * 5 + [-np.Infinity] * 5))

#         endog = np.arange(0)
#         mu = np.arange(0)
#         scale = 0.
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         np.testing.assert_array_equal(lps, np.array([]))


# class ForwardingBinomialTests(unittest.TestCase):

#     @classmethod
#     def setUpClass(cls):
#         cls.forwarding_family = ForwardingBinomial()
#         cls.family = Binomial()

#     def test_init(self):
#         self.assertEqual(self.forwarding_family.n, self.family.n)

#     def test_loglike_per_sample(self):
#         endog = np.random.randint(0, high=1, size=10)
#         mu = np.random.uniform(0, high=1, size=10)
#         scale = 1.
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         self.assertEqual(
#             np.sum(lps),
#             self.family.loglike(endog, mu, 1., scale))

#         endog = np.array([1])
#         mu = np.array([0.2])
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         np.testing.assert_array_almost_equal(lps, np.array([-1.609438]), decimal=6)

#         endog = np.array([1])
#         mu = np.array([0.98])
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         np.testing.assert_array_almost_equal(lps, np.array([-0.020203]), decimal=6)

#         endog = np.arange(0)
#         mu = np.arange(0)
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         np.testing.assert_array_equal(lps, np.array([]))


# class ForwardingInverseGaussianTests(unittest.TestCase):

#     @classmethod
#     def setUpClass(cls):
#         cls.forwarding_family = ForwardingInverseGaussian()
#         cls.family = InverseGaussian()

#     def test_init(self):
#         self.assertEqual(self.forwarding_family.family.variance, self.family.variance)

#     def test_loglike_per_sample(self):
#         endog = np.arange(1, 11)
#         mu = np.arange(1, 11)
#         scale = 1.
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         self.assertEqual(
#             np.sum(lps),
#             self.family.loglike(endog, mu, 1., scale))
#         mu = np.random.rand(10)
#         scale = 2.
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         self.assertEqual(
#             np.sum(lps),
#             self.family.loglike(endog, mu, 1., scale))
#         endog = np.arange(1, 2)
#         mu = np.arange(1, 2)
#         scale = 5.
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         self.assertEqual(
#             np.sum(lps),
#             self.family.loglike(endog, mu, 1., scale))
#         endog = np.arange(0)
#         mu = np.arange(0)
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         self.assertEqual(
#             np.sum(lps),
#             self.family.loglike(endog, mu, 1., scale))

#         endog = np.arange(1, 11)
#         mu = np.arange(1, 11)
#         scale = 0.
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         np.testing.assert_array_equal(lps, np.array([0] * 10))
#         mu = np.array([1, 2, 3, 4, 5, 7, 8, 9, 10, 11])
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         np.testing.assert_array_equal(lps, np.array([0] * 5 + [-np.Infinity] * 5))

#         endog = np.arange(0)
#         mu = np.arange(0)
#         scale = 0.
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         np.testing.assert_array_equal(lps, np.array([]))


# class ForwardingNegativeBinomialTests(unittest.TestCase):

#     @classmethod
#     def setUpClass(cls):
#         cls.forwarding_family = ForwardingNegativeBinomial()
#         cls.family = NegativeBinomial()

#     def test_init(self):
#         self.assertEqual(self.forwarding_family.alpha, self.family.alpha)

#     def test_loglike_per_sample(self):
#         endog = np.arange(1, 11)
#         mu = np.arange(1, 11)
#         scale = 1.
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         self.assertEqual(
#             np.sum(lps),
#             self.family.loglike(endog, mu, 1., scale))
#         mu = np.random.rand(10)
#         scale = 2.
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         self.assertEqual(
#             np.sum(lps),
#             self.family.loglike(endog, mu, 1., scale))
#         endog = np.arange(1, 2)
#         mu = np.arange(1, 2)
#         scale = 5.
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         self.assertEqual(
#             np.sum(lps),
#             self.family.loglike(endog, mu, 1., scale))
#         endog = np.arange(0)
#         mu = np.arange(0)
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         self.assertEqual(
#             np.sum(lps),
#             self.family.loglike(endog, mu, 1., scale))

#         endog = np.arange(1, 11)
#         mu = np.arange(1, 11)
#         scale = 0.
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         np.testing.assert_array_equal(lps, np.array([0] * 10))
#         mu = np.array([1, 2, 3, 4, 5, 7, 8, 9, 10, 11])
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         np.testing.assert_array_equal(lps, np.array([0] * 5 + [-np.Infinity] * 5))

#         endog = np.arange(0)
#         mu = np.arange(0)
#         scale = 0.
#         lps = self.forwarding_family.loglike_per_sample(endog, mu, scale)
#         self.assertEqual(lps.ndim, 1)
#         np.testing.assert_array_equal(lps, np.array([]))
