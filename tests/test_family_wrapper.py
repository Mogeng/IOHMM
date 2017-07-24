# import unittest

# import numpy as np
# from statsmodels.genmod.families.family import (Poisson,
#                                                 Gaussian,
#                                                 Gamma,
#                                                 Binomial,
#                                                 InverseGaussian,
#                                                 NegativeBinomial)

# from IOHMM import (PoissonWrapper,
#                    GaussianWrapper,
#                    GammaWrapper,
#                    BinomialWrapper,
#                    InverseGaussianWrapper,
#                    NegativeBinomialWrapper)


# # Things to test
# # 1. dimension
# # 2. sum equal to loglike is sample_weight is all ones
# # 3. when scale is zero


# class PoissonTests(unittest.TestCase):

#     @classmethod
#     def setUpClass(cls):
#         cls.forwarding_family = PoissonWrapper()
#         cls.family = Poisson()

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


# class GaussianTests(unittest.TestCase):

#     @classmethod
#     def setUpClass(cls):
#         cls.forwarding_family = GaussianWrapper()
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


# class GammaTests(unittest.TestCase):

#     @classmethod
#     def setUpClass(cls):
#         cls.forwarding_family = GammaWrapper()
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


# class BinomialTests(unittest.TestCase):

#     @classmethod
#     def setUpClass(cls):
#         cls.forwarding_family = BinomialWrapper()
#         cls.family = Binomial()

#     def test_init(self):
#         self.assertEqual(self.forwarding_family.family.n, self.family.n)

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


# class InverseGaussianTests(unittest.TestCase):

#     @classmethod
#     def setUpClass(cls):
#         cls.forwarding_family = InverseGaussianWrapper()
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


# class NegativeBinomialTests(unittest.TestCase):

#     @classmethod
#     def setUpClass(cls):
#         cls.forwarding_family = NegativeBinomialWrapper()
#         cls.family = NegativeBinomial()

#     def test_init(self):
#         self.assertEqual(self.forwarding_family.family.alpha, self.family.alpha)

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
