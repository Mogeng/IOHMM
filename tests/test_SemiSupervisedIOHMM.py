from __future__ import print_function
from __future__ import division
from builtins import range
from past.utils import old_div
import json
import unittest


import numpy as np
import pandas as pd

from IOHMM import SemiSupervisedIOHMM
from IOHMM import OLS, CrossEntropyMNL


class SemiSupervisedIOHMMTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_speed = pd.read_csv('examples/data/speed.csv')
        cls.states = cls._mock_states()

    @classmethod
    def _mock_states(cls):
        states = {}
        corr = np.array(cls.data_speed['corr'])
        for i in range(int(old_div(len(corr), 2))):
            if corr[i] == 'cor':
                states[i] = np.array([0, 1, 0, 0])
                cls.data_speed.at[i, 'rt'] = 1
            else:
                states[i] = np.array([1, 0, 0, 0])
                cls.data_speed.at[i, 'rt'] = 0
        return states

    def setUp(self):
        np.random.seed(0)

    def test_train_no_covariates(self):
        np.random.seed(0)
        self.model = SemiSupervisedIOHMM(num_states=4, max_EM_iter=100, EM_tol=1e-10)
        self.model.set_models(
            model_initial=CrossEntropyMNL(solver='newton-cg', reg_method='l2'),
            model_transition=CrossEntropyMNL(solver='newton-cg', reg_method='l2'),
            model_emissions=[OLS()])
        self.model.set_inputs(covariates_initial=[], covariates_transition=[],
                              covariates_emissions=[[]])
        self.model.set_outputs([['rt']])
        self.model.set_data([[self.data_speed, self.states]])
        self.model.train()
        # emission coefficients
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[0][0].coef,
            np.array([[0]]), decimal=1)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[1][0].coef,
            np.array([[1]]), decimal=1)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[2][0].coef,
            np.array([[6.4]]), decimal=1)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[3][0].coef,
            np.array([[5.5]]), decimal=1)

        # emission dispersion
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[0][0].dispersion, np.array([[0]]), decimal=2)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[1][0].dispersion, np.array([[0]]), decimal=2)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[2][0].dispersion, np.array([[0.051]]), decimal=2)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[3][0].dispersion, np.array([[0.032]]), decimal=2)

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
        json_dict = self.model.to_json('tests/IOHMM_models/SemiSupervisedIOHMM/')
        self.assertEqual(json_dict['data_type'], 'SemiSupervisedIOHMM')
        self.assertSetEqual(
            set(json_dict['properties'].keys()),
            set(['num_states', 'EM_tol', 'max_EM_iter',
                 'covariates_initial', 'covariates_transition',
                 'covariates_emissions', 'responses_emissions',
                 'model_initial', 'model_transition', 'model_emissions']))
        with open('tests/IOHMM_models/SemiSupervisedIOHMM/model.json', 'w') as outfile:
            json.dump(json_dict, outfile, indent=4, sort_keys=True)

    def test_from_json(self):
        with open('tests/IOHMM_models/SemiSupervisedIOHMM/model.json') as json_data:
            json_dict = json.load(json_data)
        self.model = SemiSupervisedIOHMM.from_json(json_dict)
        self.assertEqual(type(self.model), SemiSupervisedIOHMM)
        # emission coefficients
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[0][0].coef,
            np.array([[0]]), decimal=1)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[1][0].coef,
            np.array([[1]]), decimal=1)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[2][0].coef,
            np.array([[6.4]]), decimal=1)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[3][0].coef,
            np.array([[5.5]]), decimal=1)

        # emission dispersion
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[0][0].dispersion, np.array([[0]]), decimal=2)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[1][0].dispersion, np.array([[0]]), decimal=2)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[2][0].dispersion, np.array([[0.051]]), decimal=2)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[3][0].dispersion, np.array([[0.032]]), decimal=2)

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
        with open('tests/IOHMM_models/SemiSupervisedIOHMM/model.json') as json_data:
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
            ]})
        print(json_dict['properties']['model_initial'])
        self.model = SemiSupervisedIOHMM.from_config(json_dict)
        self.assertEqual(type(self.model), SemiSupervisedIOHMM)
        self.model.set_data([[self.data_speed, self.states]])
        self.model.train()

        # emission coefficients
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[0][0].coef,
            np.array([[0]]), decimal=1)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[1][0].coef,
            np.array([[1]]), decimal=1)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[2][0].coef,
            np.array([[6.4]]), decimal=1)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[3][0].coef,
            np.array([[5.5]]), decimal=1)

        # emission dispersion
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[0][0].dispersion, np.array([[0]]), decimal=2)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[1][0].dispersion, np.array([[0]]), decimal=2)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[2][0].dispersion, np.array([[0.051]]), decimal=2)
        np.testing.assert_array_almost_equal(
            self.model.model_emissions[3][0].dispersion, np.array([[0.032]]), decimal=2)

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
