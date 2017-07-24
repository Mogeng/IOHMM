import json
import unittest


import numpy as np
import pandas as pd

from IOHMM import UnSupervisedIOHMM
from IOHMM import OLS, DiscreteMNL, CrossEntropyMNL


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
            model_emissions=[OLS()])
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
            np.array([[5.5]]), decimal=1)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[1][0].coef,
            np.array([[6.4]]), decimal=1)

        # emission dispersion
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[0][0].dispersion, np.array([[0.037]]), decimal=2)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[1][0].dispersion, np.array([[0.063]]), decimal=2)

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
            model_emissions=[OLS()])
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
            np.array([[5.5]]), decimal=1)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[1][0].coef,
            np.array([[6.4]]), decimal=1)

        # emission dispersion
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[0][0].dispersion, np.array([[0.036]]), decimal=2)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[1][0].dispersion, np.array([[0.063]]), decimal=2)

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
            model_emissions=[OLS(), DiscreteMNL(reg_method='l2')])
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
            np.array([[5.5]]), decimal=1)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[1][0].coef,
            np.array([[6.4]]), decimal=1)

        # emission dispersion
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[0][0].dispersion, np.array([[0.036]]), decimal=2)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[1][0].dispersion, np.array([[0.063]]), decimal=2)

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
        json_dict = self.model.to_json('tests/IOHMM_models/UnSupervisedIOHMM/')
        self.assertEqual(json_dict['data_type'], 'UnSupervisedIOHMM')
        self.assertSetEqual(
            set(json_dict['properties'].keys()),
            set(['num_states', 'EM_tol', 'max_EM_iter',
                 'covariates_initial', 'covariates_transition',
                 'covariates_emissions', 'responses_emissions',
                 'model_initial', 'model_transition', 'model_emissions']))
        with open('tests/IOHMM_models/UnSupervisedIOHMM/model.json', 'w') as outfile:
            json.dump(json_dict, outfile)

    def test_from_json(self):
        with open('tests/IOHMM_models/UnSupervisedIOHMM/model.json') as json_data:
            json_dict = json.load(json_data)
        self.model = UnSupervisedIOHMM.from_json(json_dict)
        self.assertEqual(type(self.model), UnSupervisedIOHMM)
        # emission coefficients
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[0][0].coef,
            np.array([[5.5]]), decimal=1)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[1][0].coef,
            np.array([[6.4]]), decimal=1)

        # emission dispersion
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[0][0].dispersion, np.array([[0.036]]), decimal=2)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[1][0].dispersion, np.array([[0.063]]), decimal=2)

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
        with open('tests/IOHMM_models/UnSupervisedIOHMM/model.json') as json_data:
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
                    'data_type': 'OLS',
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
            np.array([[5.5]]), decimal=1)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[1][0].coef,
            np.array([[6.4]]), decimal=1)

        # emission dispersion
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[0][0].dispersion, np.array([[0.036]]), decimal=2)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[1][0].dispersion, np.array([[0.063]]), decimal=2)

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
