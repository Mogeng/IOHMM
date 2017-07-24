# import json
# import unittest


# import numpy as np
# import pandas as pd

# from IOHMM import SupervisedIOHMM
# from IOHMM import OLS, CrossEntropyMNL


# class SupervisedIOHMMTests(unittest.TestCase):

#     @classmethod
#     def setUpClass(cls):
#         cls.data_speed = pd.read_csv('examples/data/speed.csv')
#         cls.states = cls._mock_states()

#     @classmethod
#     def _mock_states(cls):
#         states = {}
#         corr = np.array(cls.data_speed['corr'])
#         for i in range(len(corr)):
#             if corr[i] == 'cor':
#                 states[i] = np.array([0, 1])
#             else:
#                 states[i] = np.array([1, 0])
#         return states

#     def test_train_no_covariates(self):
#         self._mock_states()
#         self.model = SupervisedIOHMM(num_states=2)
#         self.model.set_models(
#             model_initial=CrossEntropyMNL(solver='newton-cg', reg_method='l2'),
#             model_transition=CrossEntropyMNL(solver='newton-cg', reg_method='l2'),
#             model_emissions=[OLS()])
#         self.model.set_inputs(covariates_initial=[], covariates_transition=[],
#                               covariates_emissions=[[]])
#         self.model.set_outputs([['rt']])
#         self.model.set_data([[self.data_speed, self.states]])
#         self.model.train()
#         # emission coefficients
#         np.testing.assert_array_almost_equal(
#             self.model.model_emissions[0][0].coef,
#             np.array([[5.705]]), decimal=3)
#         np.testing.assert_array_almost_equal(
#             self.model.model_emissions[1][0].coef,
#             np.array([[6.137]]), decimal=3)

#         # emission dispersion
#         np.testing.assert_array_almost_equal(
#             self.model.model_emissions[0][0].dispersion, np.array([[0.128]]), decimal=3)
#         np.testing.assert_array_almost_equal(
#             self.model.model_emissions[1][0].dispersion, np.array([[0.224]]), decimal=3)

#         # transition
#         np.testing.assert_array_almost_equal(
#             np.exp(self.model.model_transition[0].predict_log_proba(np.array([[]]))),
#             np.array([[0.384, 0.616]]), decimal=3)
#         np.testing.assert_array_almost_equal(
#             np.exp(self.model.model_transition[1].predict_log_proba(np.array([[]]))),
#             np.array([[0.212, 0.788]]), decimal=3)

#         # to_json
#         json_dict = self.model.to_json('tests/iohmm_models/supervised_iohmm/')
#         self.assertEqual(json_dict['data_type'], 'SupervisedIOHMM')
#         self.assertSetEqual(
#             set(json_dict['properties'].keys()),
#             set(['num_states',
#                  'covariates_initial', 'covariates_transition',
#                  'covariates_emissions', 'responses_emissions',
#                  'model_initial', 'model_transition', 'model_emissions']))
#         with open('tests/iohmm_models/supervised_iohmm/model.json', 'w') as outfile:
#             json.dump(json_dict, outfile)

#     def test_from_json(self):
#         with open('tests/iohmm_models/supervised_iohmm/model.json') as json_data:
#             json_dict = json.load(json_data)
#         self.model = SupervisedIOHMM.from_json(json_dict)
#         self.assertEqual(type(self.model), SupervisedIOHMM)
#         # emission coefficients
#         np.testing.assert_array_almost_equal(
#             self.model.model_emissions[0][0].coef,
#             np.array([[5.705]]), decimal=3)
#         np.testing.assert_array_almost_equal(
#             self.model.model_emissions[1][0].coef,
#             np.array([[6.137]]), decimal=3)

#         # emission dispersion
#         np.testing.assert_array_almost_equal(
#             self.model.model_emissions[0][0].dispersion, np.array([[0.128]]), decimal=3)
#         np.testing.assert_array_almost_equal(
#             self.model.model_emissions[1][0].dispersion, np.array([[0.224]]), decimal=3)

#         # transition
#         np.testing.assert_array_almost_equal(
#             np.exp(self.model.model_transition[0].predict_log_proba(np.array([[]]))),
#             np.array([[0.384, 0.616]]), decimal=3)
#         np.testing.assert_array_almost_equal(
#             np.exp(self.model.model_transition[1].predict_log_proba(np.array([[]]))),
#             np.array([[0.212, 0.788]]), decimal=3)

#     def test_from_config(self):
#         with open('tests/iohmm_models/supervised_iohmm/model.json') as json_data:
#             json_dict = json.load(json_data)
#         json_dict['properties'].update({
#             'model_initial': {
#                 'data_type': 'CrossEntropyMNL',
#                 'properties': {
#                     'reg_method': 'l2',
#                     'solver': 'newton-cg'
#                 }
#             },
#             'model_transition': {
#                 'data_type': 'CrossEntropyMNL',
#                 'properties': {
#                     'reg_method': 'l2',
#                     'solver': 'newton-cg'
#                 }
#             },
#             'model_emissions': [
#                 {
#                     'data_type': 'OLS',
#                     'properties': {}
#                 },
#             ]})
#         self.model = SupervisedIOHMM.from_config(json_dict)
#         self.assertEqual(type(self.model), SupervisedIOHMM)
#         self.model.set_data([[self.data_speed, self.states]])
#         self.model.train()

#         # emission coefficients
#         np.testing.assert_array_almost_equal(
#             self.model.model_emissions[0][0].coef,
#             np.array([[5.705]]), decimal=3)
#         np.testing.assert_array_almost_equal(
#             self.model.model_emissions[1][0].coef,
#             np.array([[6.137]]), decimal=3)

#         # emission dispersion
#         np.testing.assert_array_almost_equal(
#             self.model.model_emissions[0][0].dispersion, np.array([[0.128]]), decimal=3)
#         np.testing.assert_array_almost_equal(
#             self.model.model_emissions[1][0].dispersion, np.array([[0.224]]), decimal=3)

#         # transition
#         np.testing.assert_array_almost_equal(
#             np.exp(self.model.model_transition[0].predict_log_proba(np.array([[]]))),
#             np.array([[0.384, 0.616]]), decimal=3)
#         np.testing.assert_array_almost_equal(
#             np.exp(self.model.model_transition[1].predict_log_proba(np.array([[]]))),
#             np.array([[0.212, 0.788]]), decimal=3)
