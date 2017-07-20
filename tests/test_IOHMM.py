import json
import unittest


import numpy as np
import pandas as pd

from IOHMM import UnSupervisedIOHMM, SemiSupervisedIOHMM, SupervisedIOHMM
from IOHMM import UnivariateOLS, DiscreteMNL, CrossEntropyMNL


class UnSupervisedIOHMMTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_speed = pd.read_csv('examples/data/speed.csv')

    def setUp(self):
        np.random.seed(0)

    def test_train_no_covariates(self):
        self.model = UnSupervisedIOHMM(num_states=2, max_EM_iter=100, EM_tol=1e-6)
        self.model.set_models(
            model_initial=CrossEntropyMNL(solver='lbfgs', reg_method='l2'),
            model_transition=CrossEntropyMNL(solver='lbfgs', reg_method='l2'),
            model_emissions=[UnivariateOLS()])
        self.model.set_inputs(
            covariates_initial=[],
            covariates_transition=[],
            covariates_emissions=[[]])
        self.model.set_outputs([['rt']])
        self.model.set_data([self.data_speed])
        self.model.train()

        # emission coefficients
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[0][0].coef,
            np.array([5.5]), decimal=1)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[1][0].coef,
            np.array([6.4]), decimal=1)

        # emission dispersion
        self.assertAlmostEqual(
            self.model.model_emissions[0][0].dispersion, 0.037, places=2)
        self.assertAlmostEqual(
            self.model.model_emissions[1][0].dispersion, 0.063, places=2)

        # transition
        np.testing.assert_array_almost_equal(
            self.model.model_transition[1].coef,
            np.array([[2.4]]), decimal=1)
        np.testing.assert_array_almost_equal(
            self.model.model_transition[0].coef,
            np.array([[-2]]), decimal=1)
        np.testing.assert_array_almost_equal(
            np.exp(self.model.model_transition[1].predict_log_proba(np.array([[]]))),
            np.array([[0.08, 0.92]]), decimal=2)
        np.testing.assert_array_almost_equal(
            np.exp(self.model.model_transition[0].predict_log_proba(np.array([[]]))),
            np.array([[0.88, 0.12]]), decimal=2)

    def test_train_covariates_for_transition(self):
        self.model = UnSupervisedIOHMM(num_states=2, max_EM_iter=100, EM_tol=1e-6)
        self.model.set_models(
            model_initial=CrossEntropyMNL(solver='newton-cg', reg_method='l2'),
            model_transition=CrossEntropyMNL(solver='newton-cg', reg_method='l2'),
            model_emissions=[UnivariateOLS()])
        self.model.set_inputs(
            covariates_initial=[],
            covariates_transition=['Pacc'],
            covariates_emissions=[[]])
        self.model.set_outputs([['rt']])
        self.model.set_data([self.data_speed])
        self.model.train()
        # emission coefficients
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[0][0].coef,
            np.array([5.5]), decimal=1)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[1][0].coef,
            np.array([6.4]), decimal=1)

        # emission dispersion
        self.assertAlmostEqual(
            self.model.model_emissions[0][0].dispersion, 0.036, places=2)
        self.assertAlmostEqual(
            self.model.model_emissions[1][0].dispersion, 0.063, places=2)

        # transition
        np.testing.assert_array_almost_equal(
            np.exp(self.model.model_transition[0].predict_log_proba(
                self.model.inp_transitions_all_users)).sum(axis=0),
            np.array([312, 126]), decimal=0)
        np.testing.assert_array_almost_equal(
            np.exp(self.model.model_transition[1].predict_log_proba(
                self.model.inp_transitions_all_users)).sum(axis=0),
            np.array([112, 326]), decimal=0)

    def test_train_multivariate(self):
        self.model = UnSupervisedIOHMM(num_states=2, max_EM_iter=100, EM_tol=1e-6)
        self.model.set_models(
            model_initial=CrossEntropyMNL(solver='newton-cg', reg_method='l2'),
            model_transition=CrossEntropyMNL(solver='newton-cg', reg_method='l2'),
            model_emissions=[UnivariateOLS(), DiscreteMNL(reg_method='l2')])
        self.model.set_inputs(
            covariates_initial=[],
            covariates_transition=[],
            covariates_emissions=[[], ['Pacc']])
        self.model.set_outputs([['rt'], ['corr']])
        self.model.set_data([self.data_speed])
        self.model.train()

        # emission coefficients
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[0][0].coef,
            np.array([5.5]), decimal=1)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[1][0].coef,
            np.array([6.4]), decimal=1)

        # emission dispersion
        self.assertAlmostEqual(
            self.model.model_emissions[0][0].dispersion, 0.036, places=2)
        self.assertAlmostEqual(
            self.model.model_emissions[1][0].dispersion, 0.063, places=2)

        # transition
        np.testing.assert_array_almost_equal(
            np.exp(self.model.model_transition[0].predict_log_proba(
                self.model.inp_transitions_all_users)).sum(axis=0),
            np.array([387, 51]), decimal=0)
        np.testing.assert_array_almost_equal(
            np.exp(self.model.model_transition[1].predict_log_proba(
                self.model.inp_transitions_all_users)).sum(axis=0),
            np.array([37, 401.]), decimal=0)

        # to_json
        json_dict = self.model.to_json('tests/iohmm_models/unsupervised_iohmm/')
        self.assertEqual(json_dict['data_type'], 'UnSupervisedIOHMM')
        self.assertSetEqual(
            set(json_dict['properties'].keys()),
            set(['num_states', 'EM_tol', 'max_EM_iter',
                 'covariates_initial', 'covariates_transition',
                 'covariates_emissions', 'responses_emissions',
                 'model_initial', 'model_transition', 'model_emissions']))
        with open('tests/iohmm_models/unsupervised_iohmm/model.json', 'w') as outfile:
            json.dump(json_dict, outfile)

    def test_from_json(self):
        with open('tests/iohmm_models/unsupervised_iohmm/model.json') as json_data:
            json_dict = json.load(json_data)
        self.model = UnSupervisedIOHMM.from_json(json_dict)
        self.assertEqual(type(self.model), UnSupervisedIOHMM)
        # emission coefficients
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[0][0].coef,
            np.array([5.5]), decimal=1)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[1][0].coef,
            np.array([6.4]), decimal=1)

        # emission dispersion
        self.assertAlmostEqual(
            self.model.model_emissions[0][0].dispersion, 0.036, places=2)
        self.assertAlmostEqual(
            self.model.model_emissions[1][0].dispersion, 0.063, places=2)

        # transition
        self.model.set_data([self.data_speed])
        np.testing.assert_array_almost_equal(
            np.exp(self.model.model_transition[0].predict_log_proba(
                self.model.inp_transitions_all_users)).sum(axis=0),
            np.array([387, 51]), decimal=0)
        np.testing.assert_array_almost_equal(
            np.exp(self.model.model_transition[1].predict_log_proba(
                self.model.inp_transitions_all_users)).sum(axis=0),
            np.array([37, 401.]), decimal=0)

    def test_from_config(self):
        with open('tests/iohmm_models/unsupervised_iohmm/model.json') as json_data:
            json_dict = json.load(json_data)
        json_dict['properties'].update({
            'model_initial': {
                'data_type': 'CrossEntropyMNL',
                'properties': {
                    'reg_method': 'l2',
                    'solver': 'newton-cg'
                }
            },
            'model_transition': {
                'data_type': 'CrossEntropyMNL',
                'properties': {
                    'reg_method': 'l2',
                    'solver': 'newton-cg'
                }
            },
            'model_emissions': [
                {
                    'data_type': 'UnivariateOLS',
                    'properties': {}
                },
                {
                    'data_type': 'DiscreteMNL',
                    'properties': {'reg_method': 'l2'}
                }
            ]})
        self.model = UnSupervisedIOHMM.from_config(json_dict)
        self.assertEqual(type(self.model), UnSupervisedIOHMM)
        self.model.set_data([self.data_speed])
        self.model.train()
        # emission coefficients
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[0][0].coef,
            np.array([5.5]), decimal=1)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[1][0].coef,
            np.array([6.4]), decimal=1)

        # emission dispersion
        self.assertAlmostEqual(
            self.model.model_emissions[0][0].dispersion, 0.036, places=2)
        self.assertAlmostEqual(
            self.model.model_emissions[1][0].dispersion, 0.063, places=2)

        # transition
        self.model.set_data([self.data_speed])
        np.testing.assert_array_almost_equal(
            np.exp(self.model.model_transition[0].predict_log_proba(
                self.model.inp_transitions_all_users)).sum(axis=0),
            np.array([387, 51]), decimal=0)
        np.testing.assert_array_almost_equal(
            np.exp(self.model.model_transition[1].predict_log_proba(
                self.model.inp_transitions_all_users)).sum(axis=0),
            np.array([37, 401.]), decimal=0)


class SemiSupervisedIOHMMTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_speed = pd.read_csv('examples/data/speed.csv')
        cls.states = cls._mock_states()

    @classmethod
    def _mock_states(cls):
        states = {}
        corr = np.array(cls.data_speed['corr'])
        for i in range(int(len(corr) / 2)):
            if corr[i] == 'cor':
                states[i] = np.array([0, 1, 0, 0])
                cls.data_speed.set_value(i, 'rt', 1)
            else:
                states[i] = np.array([1, 0, 0, 0])
                cls.data_speed.set_value(i, 'rt', 0)
        return states

    def setUp(self):
        np.random.seed(0)

    def test_train_no_covariates(self):
        np.random.seed(0)
        self.model = SemiSupervisedIOHMM(num_states=4, max_EM_iter=100, EM_tol=1e-10)
        self.model.set_models(
            model_initial=CrossEntropyMNL(solver='newton-cg', reg_method='l2'),
            model_transition=CrossEntropyMNL(solver='newton-cg', reg_method='l2'),
            model_emissions=[UnivariateOLS()])
        self.model.set_inputs(covariates_initial=[], covariates_transition=[],
                              covariates_emissions=[[]])
        self.model.set_outputs([['rt']])
        self.model.set_data([[self.data_speed, self.states]])
        self.model.train()
        # emission coefficients
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[0][0].coef,
            np.array([0]), decimal=1)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[1][0].coef,
            np.array([1]), decimal=1)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[2][0].coef,
            np.array([6.4]), decimal=1)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[3][0].coef,
            np.array([5.5]), decimal=1)

        # emission dispersion
        self.assertAlmostEqual(
            self.model.model_emissions[0][0].dispersion, 0, places=2)
        self.assertAlmostEqual(
            self.model.model_emissions[1][0].dispersion, 0, places=2)
        self.assertAlmostEqual(
            self.model.model_emissions[2][0].dispersion, 0.051, places=2)
        self.assertAlmostEqual(
            self.model.model_emissions[3][0].dispersion, 0.032, places=2)

        # transition
        np.testing.assert_array_almost_equal(
            np.exp(self.model.model_transition[0].predict_log_proba(np.array([[]]))),
            np.array([[0.4, 0.6, 0, 0]]), decimal=1)
        np.testing.assert_array_almost_equal(
            np.exp(self.model.model_transition[1].predict_log_proba(np.array([[]]))),
            np.array([[0.19, 0.81, 0, 0]]), decimal=2)
        np.testing.assert_array_almost_equal(
            np.exp(self.model.model_transition[2].predict_log_proba(np.array([[]]))),
            np.array([[0, 0, 0.93, 0.07]]), decimal=2)
        np.testing.assert_array_almost_equal(
            np.exp(self.model.model_transition[3].predict_log_proba(np.array([[]]))),
            np.array([[0, 0, 0.11, 0.89]]), decimal=2)

        # to_json
        json_dict = self.model.to_json('tests/iohmm_models/semisupervised_iohmm/')
        self.assertEqual(json_dict['data_type'], 'SemiSupervisedIOHMM')
        self.assertSetEqual(
            set(json_dict['properties'].keys()),
            set(['num_states', 'EM_tol', 'max_EM_iter',
                 'covariates_initial', 'covariates_transition',
                 'covariates_emissions', 'responses_emissions',
                 'model_initial', 'model_transition', 'model_emissions']))
        with open('tests/iohmm_models/semisupervised_iohmm/model.json', 'w') as outfile:
            json.dump(json_dict, outfile)

    def test_from_json(self):
        with open('tests/iohmm_models/semisupervised_iohmm/model.json') as json_data:
            json_dict = json.load(json_data)
        self.model = SemiSupervisedIOHMM.from_json(json_dict)
        self.assertEqual(type(self.model), SemiSupervisedIOHMM)
        # emission coefficients
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[0][0].coef,
            np.array([0]), decimal=1)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[1][0].coef,
            np.array([1]), decimal=1)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[2][0].coef,
            np.array([6.4]), decimal=1)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[3][0].coef,
            np.array([5.5]), decimal=1)

        # emission dispersion
        self.assertAlmostEqual(
            self.model.model_emissions[0][0].dispersion, 0, places=2)
        self.assertAlmostEqual(
            self.model.model_emissions[1][0].dispersion, 0, places=2)
        self.assertAlmostEqual(
            self.model.model_emissions[2][0].dispersion, 0.051, places=2)
        self.assertAlmostEqual(
            self.model.model_emissions[3][0].dispersion, 0.032, places=2)

        # transition
        np.testing.assert_array_almost_equal(
            np.exp(self.model.model_transition[0].predict_log_proba(np.array([[]]))),
            np.array([[0.4, 0.6, 0, 0]]), decimal=1)
        np.testing.assert_array_almost_equal(
            np.exp(self.model.model_transition[1].predict_log_proba(np.array([[]]))),
            np.array([[0.19, 0.81, 0, 0]]), decimal=2)
        np.testing.assert_array_almost_equal(
            np.exp(self.model.model_transition[2].predict_log_proba(np.array([[]]))),
            np.array([[0, 0, 0.93, 0.07]]), decimal=2)
        np.testing.assert_array_almost_equal(
            np.exp(self.model.model_transition[3].predict_log_proba(np.array([[]]))),
            np.array([[0, 0, 0.11, 0.89]]), decimal=2)

    def test_from_config(self):
        with open('tests/iohmm_models/semisupervised_iohmm/model.json') as json_data:
            json_dict = json.load(json_data)
        json_dict['properties'].update({
            'model_initial': {
                'data_type': 'CrossEntropyMNL',
                'properties': {
                    'reg_method': 'l2',
                    'solver': 'newton-cg'
                }
            },
            'model_transition': {
                'data_type': 'CrossEntropyMNL',
                'properties': {
                    'reg_method': 'l2',
                    'solver': 'newton-cg'
                }
            },
            'model_emissions': [
                {
                    'data_type': 'UnivariateOLS',
                    'properties': {}
                },
            ]})
        print json_dict['properties']['model_initial']
        self.model = SemiSupervisedIOHMM.from_config(json_dict)
        self.assertEqual(type(self.model), SemiSupervisedIOHMM)
        self.model.set_data([[self.data_speed, self.states]])
        self.model.train()

        # emission coefficients
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[0][0].coef,
            np.array([0]), decimal=1)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[1][0].coef,
            np.array([1]), decimal=1)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[2][0].coef,
            np.array([6.4]), decimal=1)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[3][0].coef,
            np.array([5.5]), decimal=1)

        # emission dispersion
        self.assertAlmostEqual(
            self.model.model_emissions[0][0].dispersion, 0, places=2)
        self.assertAlmostEqual(
            self.model.model_emissions[1][0].dispersion, 0, places=2)
        self.assertAlmostEqual(
            self.model.model_emissions[2][0].dispersion, 0.051, places=2)
        self.assertAlmostEqual(
            self.model.model_emissions[3][0].dispersion, 0.032, places=2)

        # transition
        np.testing.assert_array_almost_equal(
            np.exp(self.model.model_transition[0].predict_log_proba(np.array([[]]))),
            np.array([[0.4, 0.6, 0, 0]]), decimal=1)
        np.testing.assert_array_almost_equal(
            np.exp(self.model.model_transition[1].predict_log_proba(np.array([[]]))),
            np.array([[0.19, 0.81, 0, 0]]), decimal=2)
        np.testing.assert_array_almost_equal(
            np.exp(self.model.model_transition[2].predict_log_proba(np.array([[]]))),
            np.array([[0, 0, 0.93, 0.07]]), decimal=2)
        np.testing.assert_array_almost_equal(
            np.exp(self.model.model_transition[3].predict_log_proba(np.array([[]]))),
            np.array([[0, 0, 0.11, 0.89]]), decimal=2)


class SupervisedIOHMMTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_speed = pd.read_csv('examples/data/speed.csv')
        cls.states = cls._mock_states()

    @classmethod
    def _mock_states(cls):
        states = {}
        corr = np.array(cls.data_speed['corr'])
        for i in range(len(corr)):
            if corr[i] == 'cor':
                states[i] = np.array([0, 1])
            else:
                states[i] = np.array([1, 0])
        return states

    def test_train_no_covariates(self):
        self._mock_states()
        self.model = SupervisedIOHMM(num_states=2)
        self.model.set_models(
            model_initial=CrossEntropyMNL(solver='newton-cg', reg_method='l2'),
            model_transition=CrossEntropyMNL(solver='newton-cg', reg_method='l2'),
            model_emissions=[UnivariateOLS()])
        self.model.set_inputs(covariates_initial=[], covariates_transition=[],
                              covariates_emissions=[[]])
        self.model.set_outputs([['rt']])
        self.model.set_data([[self.data_speed, self.states]])
        self.model.train()
        # emission coefficients
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[0][0].coef,
            np.array([5.705]), decimal=3)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[1][0].coef,
            np.array([6.137]), decimal=3)

        # emission dispersion
        self.assertAlmostEqual(
            self.model.model_emissions[0][0].dispersion, 0.128, places=3)
        self.assertAlmostEqual(
            self.model.model_emissions[1][0].dispersion, 0.224, places=3)

        # transition
        np.testing.assert_array_almost_equal(
            np.exp(self.model.model_transition[0].predict_log_proba(np.array([[]]))),
            np.array([[0.384, 0.616]]), decimal=3)
        np.testing.assert_array_almost_equal(
            np.exp(self.model.model_transition[1].predict_log_proba(np.array([[]]))),
            np.array([[0.212, 0.788]]), decimal=3)

        # to_json
        json_dict = self.model.to_json('tests/iohmm_models/supervised_iohmm/')
        self.assertEqual(json_dict['data_type'], 'SupervisedIOHMM')
        self.assertSetEqual(
            set(json_dict['properties'].keys()),
            set(['num_states',
                 'covariates_initial', 'covariates_transition',
                 'covariates_emissions', 'responses_emissions',
                 'model_initial', 'model_transition', 'model_emissions']))
        with open('tests/iohmm_models/supervised_iohmm/model.json', 'w') as outfile:
            json.dump(json_dict, outfile)

    def test_from_json(self):
        with open('tests/iohmm_models/supervised_iohmm/model.json') as json_data:
            json_dict = json.load(json_data)
        self.model = SupervisedIOHMM.from_json(json_dict)
        self.assertEqual(type(self.model), SupervisedIOHMM)
        # emission coefficients
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[0][0].coef,
            np.array([5.705]), decimal=3)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[1][0].coef,
            np.array([6.137]), decimal=3)

        # emission dispersion
        self.assertAlmostEqual(
            self.model.model_emissions[0][0].dispersion, 0.128, places=3)
        self.assertAlmostEqual(
            self.model.model_emissions[1][0].dispersion, 0.224, places=3)

        # transition
        np.testing.assert_array_almost_equal(
            np.exp(self.model.model_transition[0].predict_log_proba(np.array([[]]))),
            np.array([[0.384, 0.616]]), decimal=3)
        np.testing.assert_array_almost_equal(
            np.exp(self.model.model_transition[1].predict_log_proba(np.array([[]]))),
            np.array([[0.212, 0.788]]), decimal=3)

    def test_from_config(self):
        with open('tests/iohmm_models/supervised_iohmm/model.json') as json_data:
            json_dict = json.load(json_data)
        json_dict['properties'].update({
            'model_initial': {
                'data_type': 'CrossEntropyMNL',
                'properties': {
                    'reg_method': 'l2',
                    'solver': 'newton-cg'
                }
            },
            'model_transition': {
                'data_type': 'CrossEntropyMNL',
                'properties': {
                    'reg_method': 'l2',
                    'solver': 'newton-cg'
                }
            },
            'model_emissions': [
                {
                    'data_type': 'UnivariateOLS',
                    'properties': {}
                },
            ]})
        self.model = SupervisedIOHMM.from_config(json_dict)
        self.assertEqual(type(self.model), SupervisedIOHMM)
        self.model.set_data([[self.data_speed, self.states]])
        self.model.train()

        # emission coefficients
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[0][0].coef,
            np.array([5.705]), decimal=3)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[1][0].coef,
            np.array([6.137]), decimal=3)

        # emission dispersion
        self.assertAlmostEqual(
            self.model.model_emissions[0][0].dispersion, 0.128, places=3)
        self.assertAlmostEqual(
            self.model.model_emissions[1][0].dispersion, 0.224, places=3)

        # transition
        np.testing.assert_array_almost_equal(
            np.exp(self.model.model_transition[0].predict_log_proba(np.array([[]]))),
            np.array([[0.384, 0.616]]), decimal=3)
        np.testing.assert_array_almost_equal(
            np.exp(self.model.model_transition[1].predict_log_proba(np.array([[]]))),
            np.array([[0.212, 0.788]]), decimal=3)
