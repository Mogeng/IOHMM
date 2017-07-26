from __future__ import division
from copy import deepcopy
import logging
import os
import warnings


import numpy as np


from forward_backward import forward_backward
from linear_models import (GLM, OLS, DiscreteMNL, CrossEntropyMNL)


warnings.simplefilter("ignore")
np.random.seed(0)
EPS = np.finfo(float).eps
# example:


class LinearModelLoader(object):
    """The mapping from data_type of a linear model
       ('GLM', 'OLS', 'DiscreteMNL', 'CrossEntropyMNL')
       to the correct class.

    """
    GLM = GLM
    OLS = OLS
    DiscreteMNL = DiscreteMNL
    CrossEntropyMNL = CrossEntropyMNL


class BaseIOHMM(object):

    def __init__(self, num_states=2):
        self.num_states = num_states
        self.trained = False

    def set_models(self, model_emissions,
                   model_initial=CrossEntropyMNL(),
                   model_transition=CrossEntropyMNL(), trained=False):
        # initial model and transition model must be MNLP
        if trained:
            self.model_initial = model_initial
            self.model_transition = model_transition
            self.model_emissions = model_emissions
            self.trained = True
        else:
            self.model_initial = model_initial
            self.model_transition = [deepcopy(model_initial) for _ in range(self.num_states)]
            self.model_emissions = [deepcopy(model_emissions) for _ in range(self.num_states)]

    def set_inputs(self, covariates_initial, covariates_transition, covariates_emissions):
        self.covariates_initial = covariates_initial
        self.covariates_transition = covariates_transition
        self.covariates_emissions = covariates_emissions
        # input should be a list inidicating the columns of the dataframe

    def set_outputs(self, responses_emissions):
        # output should be a list inidicating the columns of the dataframe
        self.responses_emissions = responses_emissions
        self.num_emissions = len(responses_emissions)

    def set_data(self, dfs):
        raise NotImplementedError

    def _initialize(self, with_randomness=True):
        # initialize log_gammas
        def _initialize_log_gamma(df, log_state):
            log_gamma = np.log(np.zeros((df.shape[0], self.num_states)))
            for k in log_state:
                log_gamma[k, :] = log_state[k]
            return log_gamma

        def _initialize_log_epsilon(df, log_state):
            log_epsilon = np.log(np.zeros((df.shape[0] - 1, self.num_states, self.num_states)))
            for i in log_state:
                if i + 1 in log_state:
                    st = int(np.argmax(log_state[i]))
                    log_epsilon[i, st, :] = log_state[i + 1]
            return log_epsilon

        # initialize log_gammas and log_epsilons
        self.log_gammas = [_initialize_log_gamma(df, log_state)
                           for df, log_state in self.dfs_logStates]

        self.log_epsilons = [_initialize_log_epsilon(df, log_state)
                             for df, log_state in self.dfs_logStates]
        if with_randomness:
            for st in range(self.num_states):
                if np.exp(np.hstack([lg[:, st] for lg in self.log_gammas])).sum() < EPS:
                    for lg in self.log_gammas:
                        lg[:, st] = np.random.rand(lg.shape[0])
            for st in range(self.num_states):
                if np.exp(np.hstack([le[:, st, :] for le in self.log_epsilons])).sum() < EPS:
                    for le in self.log_epsilons:
                        le[:, st, :] = np.random.rand(le.shape[0], self.num_states)

        # initialize log_likelihood
        self.log_likelihood = -np.Infinity

        # initialize input/output covariates
        self.inp_initials = [np.array(df[self.covariates_initial].iloc[0]).reshape(
            1, -1).astype('float64') for df, log_state in self.dfs_logStates]
        self.inp_initials_all_users = np.vstack(self.inp_initials)

        self.inp_transitions = [np.array(df[self.covariates_transition].iloc[1:]).astype(
            'float64') for df, log_state in self.dfs_logStates]
        self.inp_transitions_all_users = np.vstack(self.inp_transitions)

        self.inp_emissions = [[np.array(df[cov]).astype('float64') for
                               cov in self.covariates_emissions]
                              for df, log_state in self.dfs_logStates]
        self.inp_emissions_all_users = [np.vstack([x[emis] for
                                                   x in self.inp_emissions]) for
                                        emis in range(self.num_emissions)]
        self.out_emissions = [[np.array(df[res]) for
                               res in self.responses_emissions]
                              for df, log_state in self.dfs_logStates]

        self.out_emissions_all_users = [np.vstack([x[emis] for
                                                   x in self.out_emissions]) for
                                        emis in range(self.num_emissions)]

    def E_step(self):
        self.log_gammas = []
        self.log_epsilons = []
        self.log_likelihoods = []
        for seq in range(self.num_seqs):
            n_records = self.dfs_logStates[seq][0].shape[0]
            log_prob_initial = self.model_initial.predict_log_proba(
                self.inp_initials[seq]).reshape(self.num_states,)
            log_prob_transition = np.zeros((n_records - 1, self.num_states, self.num_states))
            for st in range(self.num_states):
                log_prob_transition[:, st, :] = self.model_transition[st].predict_log_proba(
                    self.inp_transitions[seq])
            assert log_prob_transition.shape == (n_records - 1, self.num_states, self.num_states)
            log_Ey = np.zeros((n_records, self.num_states))
            for emis in range(self.num_emissions):
                model_collection = [models[emis] for models in self.model_emissions]
                log_Ey += np.vstack([model.loglike_per_sample(
                    np.array(self.inp_emissions[seq][emis]).astype('float64'),
                    np.array(self.out_emissions[seq][emis])) for model in model_collection]).T

            log_gamma, log_epsilon, log_likelihood = forward_backward(
                log_prob_initial, log_prob_transition, log_Ey, self.dfs_logStates[seq][1])
            self.log_gammas.append(log_gamma)
            self.log_epsilons.append(log_epsilon)
            self.log_likelihoods.append(log_likelihood)
        self.log_likelihood = sum(self.log_likelihoods)

    def M_step(self):
        # optimize initial model
        X = self.inp_initials_all_users
        Y = np.exp(np.vstack([lg[0, :].reshape(1, -1) for lg in self.log_gammas]))
        self.model_initial.fit(X, Y)

        # optimize transition models
        X = self.inp_transitions_all_users
        for st in range(self.num_states):
            Y = np.exp(np.vstack([eps[:, st, :] for eps in self.log_epsilons]))
            self.model_transition[st].fit(X, Y)

        # optimize emission models
        for emis in range(self.num_emissions):
            X = self.inp_emissions_all_users[emis]
            Y = self.out_emissions_all_users[emis]
            for st in range(self.num_states):
                sample_weight = np.exp(np.hstack([lg[:, st] for lg in self.log_gammas]))
                # now need to add a tolist so that sklearn works fine
                self.model_emissions[st][emis].fit(X, Y, sample_weight=sample_weight)

    def train(self):
        for it in range(self.max_EM_iter):
            log_likelihood_prev = self.log_likelihood
            self.M_step()
            self.E_step()
            logging.info('log likelihood of iteration {0}: {1:.4f}'.format(it, self.log_likelihood))
            if abs(self.log_likelihood - log_likelihood_prev) < self.EM_tol:
                break
        self.trained = True

    def to_json(self, path):
        json_dict = {
            'data_type': self.__class__.__name__,
            'properties': {
                'num_states': self.num_states,
                'covariates_initial': self.covariates_initial,
                'covariates_transition': self.covariates_transition,
                'covariates_emissions': self.covariates_emissions,
                'responses_emissions': self.responses_emissions,
                'model_initial': self.model_initial.to_json(
                    path=os.path.join(path, 'model_initial')),
                'model_transition': [self.model_transition[st].to_json(
                    path=os.path.join(path, 'model_transition', 'state_{}'.format(st))) for
                    st in range(self.num_states)],
                'model_emissions': [[self.model_emissions[st][emis].to_json(
                    path=os.path.join(
                        path, 'model_emissions', 'state_{}'.format(st), 'emission_{}'.format(emis))
                ) for emis in range(self.num_emissions)] for st in range(self.num_states)]
            }
        }
        return json_dict

    @classmethod
    def _from_setup(
            cls, json_dict, num_states,
            model_initial, model_transition, model_emissions,
            covariates_initial, covariates_transition, covariates_emissions,
            responses_emissions, trained):
        model = cls(num_states=num_states)
        model.set_models(
            model_initial=model_initial,
            model_transition=model_transition,
            model_emissions=model_emissions,
            trained=trained)
        model.set_inputs(covariates_initial=covariates_initial,
                         covariates_transition=covariates_transition,
                         covariates_emissions=covariates_emissions)
        model.set_outputs(responses_emissions=responses_emissions)
        return model

    @classmethod
    def from_config(cls, json_dict):
        return cls._from_setup(
            json_dict,
            num_states=json_dict['properties']['num_states'],
            model_initial=getattr(
                LinearModelLoader, json_dict['properties']['model_initial']['data_type'])(
                    **json_dict['properties']['model_initial']['properties']),
            model_transition=getattr(
                LinearModelLoader, json_dict['properties']['model_transition']['data_type'])(
                    **json_dict['properties']['model_transition']['properties']),
            model_emissions=[getattr(
                LinearModelLoader, model_emission['data_type'])(**model_emission['properties'])
                for model_emission in json_dict['properties']['model_emissions']],
            covariates_initial=json_dict['properties']['covariates_initial'],
            covariates_transition=json_dict['properties']['covariates_transition'],
            covariates_emissions=json_dict['properties']['covariates_emissions'],
            responses_emissions=json_dict['properties']['responses_emissions'],
            trained=False)

    @classmethod
    def from_json(cls, json_dict):
        return cls._from_setup(
            json_dict,
            num_states=json_dict['properties']['num_states'],
            model_initial=getattr(
                LinearModelLoader, json_dict['properties']['model_initial']['data_type']).from_json(
                json_dict['properties']['model_initial']),
            model_transition=[getattr(
                LinearModelLoader, model_transition_json['data_type']
            ).from_json(model_transition_json) for
                model_transition_json in json_dict['properties']['model_transition']],
            model_emissions=[[getattr(
                LinearModelLoader, model_emission_json['data_type']
            ).from_json(model_emission_json) for model_emission_json in model_emissions_json] for
                model_emissions_json in json_dict['properties']['model_emissions']],
            covariates_initial=json_dict['properties']['covariates_initial'],
            covariates_transition=json_dict['properties']['covariates_transition'],
            covariates_emissions=json_dict['properties']['covariates_emissions'],
            responses_emissions=json_dict['properties']['responses_emissions'],
            trained=True)


class UnSupervisedIOHMM(BaseIOHMM):

    def __init__(self, num_states=2, EM_tol=1e-4, max_EM_iter=100):
        super(UnSupervisedIOHMM, self).__init__(num_states=num_states)
        self.EM_tol = EM_tol
        self.max_EM_iter = max_EM_iter

    def set_data(self, dfs):
        assert all([df.shape[0] > 0 for df in dfs])
        self.num_seqs = len(dfs)
        self.dfs_logStates = map(lambda x: [x, {}], dfs)
        self._initialize(with_randomness=True)

    def to_json(self, path):
        json_dict = super(UnSupervisedIOHMM, self).to_json(path)
        json_dict['properties'].update(
            {
                'EM_tol': self.EM_tol,
                'max_EM_iter': self.max_EM_iter,
            }
        )
        return json_dict

    @classmethod
    def _from_setup(
            cls, json_dict, num_states,
            model_initial, model_transition, model_emissions,
            covariates_initial, covariates_transition, covariates_emissions,
            responses_emissions, trained):
        model = cls(num_states=num_states,
                    EM_tol=json_dict['properties']['EM_tol'],
                    max_EM_iter=json_dict['properties']['max_EM_iter'])
        model.set_models(
            model_initial=model_initial,
            model_transition=model_transition,
            model_emissions=model_emissions,
            trained=trained)
        model.set_inputs(covariates_initial=covariates_initial,
                         covariates_transition=covariates_transition,
                         covariates_emissions=covariates_emissions)
        model.set_outputs(responses_emissions=responses_emissions)
        return model


class SemiSupervisedIOHMM(UnSupervisedIOHMM):
    def set_data(self, dfs_states):
        self.num_seqs = len(dfs_states)
        self.dfs_logStates = map(lambda x: [x[0], {k: np.log(x[1][k]) for k in x[1]}], dfs_states)
        self._initialize(with_randomness=True)


class SupervisedIOHMM(BaseIOHMM):
    def __init__(self, num_states=2):
        super(SupervisedIOHMM, self).__init__(num_states=num_states)
        self.max_EM_iter = 1
        self.EM_tol = 0

    def set_data(self, dfs_states):
        self.num_seqs = len(dfs_states)
        self.dfs_logStates = map(lambda x: [x[0], {k: np.log(x[1][k]) for k in x[1]}], dfs_states)
        self._initialize(with_randomness=False)
