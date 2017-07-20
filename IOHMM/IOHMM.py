from __future__ import division
from copy import deepcopy
import logging
import os
import warnings


import numpy as np


from HMM_utils import cal_HMM
from linear_models import (GLM,
                           UnivariateOLS, MultivariateOLS,
                           DiscreteMNL, CrossEntropyMNL)


warnings.simplefilter("ignore")
np.random.seed(0)
# example:


class LinearModelLoader(object):
    """The mapping from data_type of a linear model
       ('GLM', 'UnivariateOLS', 'MultivariateOLS', 'DiscreteMNL', 'CrossEntropyMNL')
       to the correct class.

    """
    GLM = GLM
    UnivariateOLS = UnivariateOLS
    MultivariateOLS = MultivariateOLS
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

    def _initialize(self):
        # initialize log_gammas
        def _initialize_log_gamma(df, log_state):
            log_gamma = np.log(np.zeros((df.shape[0], self.num_states)))
            for k in log_state:
                log_gamma[k, :] = log_state[k]
            return log_gamma

        self.log_gammas = [_initialize_log_gamma(df, log_state)
                           for df, log_state in self.dfs_logStates]
        for st in range(self.num_states):
            if np.exp(np.hstack([lg[:, st] for lg in self.log_gammas])).sum() == 0:
                for lg in self.log_gammas:
                    lg[:, st] = np.random.rand(lg.shape[0])

        # initialize log_epsilons
        self.log_epsilons = [np.random.rand(df.shape[0] - 1, self.num_states, self.num_states) for
                             df, log_state in self.dfs_logStates]

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

            log_gamma, log_epsilon, log_likelihood = cal_HMM(
                log_prob_initial, log_prob_transition, log_Ey, self.dfs_logStates[seq][1])
            self.log_gammas.append(log_gamma)
            self.log_epsilons.append(log_epsilon)
            self.log_likelihoods.append(log_likelihood)
        self.log_likelihood = sum(self.log_likelihoods)

    def M_step(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

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
        self._initialize()

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
        self._initialize()


class SupervisedIOHMM(BaseIOHMM):
    # // TODO this is a redundant code as SemiSupervised IOHMM, don't know what to do
    def set_data(self, dfs_states):
        self.num_seqs = len(dfs_states)
        self.dfs_logStates = map(lambda x: [x[0], {k: np.log(x[1][k]) for k in x[1]}], dfs_states)
        self._initialize()

    def _initialize_labeled(self):
        def _initialize_labeled_input_for_initial(df, log_state):
            if 0 in log_state:
                return np.array(df[self.covariates_initial].iloc[0]).reshape(1, -1).astype('float64')
            else:
                return np.empty((0, len(self.covariates_initial)), dtype=float)

        def _initialize_labeled_output_for_initial(df, log_state):
            if 0 in log_state:
                return log_state[0].reshape(1, -1).astype('float64')
            else:
                return np.empty((0, self.num_states), dtype=float)

        def _initialize_labeled_input_for_transition(df, log_state):
            ind = {}
            inp = {}
            for i in range(self.num_states):
                ind[i] = []
            for i in log_state:
                if i + 1 in log_state:
                    st = int(np.argmax(log_state[i]))
                    ind[st].append(i + 1)
            for i in range(self.num_states):
                inp[i] = np.array(df.iloc[ind[i]][self.covariates_transition]).astype('float64')
            return inp

        def _initialize_labeled_output_for_transition(df, log_state):
            ind = {}
            out = {}
            for i in range(self.num_states):
                ind[i] = []
            for i in log_state:
                if i + 1 in log_state:
                    st = int(np.argmax(log_state[i]))
                    ind[st].append(i + 1)
            for i in range(self.num_states):
                out[i] = [log_state[k].reshape(1, -1).astype('float64') for k in ind[i]]
                if out[i] == []:
                    out[i] = [np.empty((0, self.num_states), dtype=float)]
            return out

        def _initialize_labeled_input_output_for_emission(df, log_state, cov):
            ind = {}
            inp = {}
            for i in range(self.num_states):
                ind[i] = []
            for i in log_state:
                st = int(np.argmax(log_state[i]))
                ind[st].append(i)
            for i in range(self.num_states):
                inp[i] = np.array(df.iloc[ind[i]][cov]).astype('float64')
            return inp

        inp_initials = [_initialize_labeled_input_for_initial(df, log_state)
                        for df, log_state in self.dfs_logStates]
        self.inp_initials_all_users_labeled = np.vstack(inp_initials)
        out_initials = [_initialize_labeled_output_for_initial(df, log_state)
                        for df, log_state in self.dfs_logStates]
        self.out_initials_all_users_labeled = np.vstack(out_initials)

        inp_transitions = [_initialize_labeled_input_for_transition(
            df, log_state) for df, log_state in self.dfs_logStates]
        self.inp_transitions_all_users_labeled = {i: np.vstack(
            [x[i] for x in inp_transitions]) for i in range(self.num_states)}
        out_transitions = [_initialize_labeled_output_for_transition(
            df, log_state) for df, log_state in self.dfs_logStates]
        self.out_transitions_all_users_labeled = {i: np.vstack(
            [item for sublist in out_transitions for item in sublist[i]])
            for i in range(self.num_states)}

        inp_emissions = []
        self.inp_emissions_all_users_labeled = []
        for cov in self.covariates_emissions:
            inp_emissions.append([_initialize_labeled_input_output_for_emission(
                df, log_state, cov) for df, log_state in self.dfs_logStates])
        for covs in inp_emissions:
            self.inp_emissions_all_users_labeled.append(
                {i: np.vstack([x[i] for x in covs]) for i in range(self.num_states)})

        out_emissions = []
        self.out_emissions_all_users_labeled = []
        for res in self.responses_emissions:
            out_emissions.append([_initialize_labeled_input_output_for_emission(
                df, log_state, res) for df, log_state in self.dfs_logStates])
        for ress in out_emissions:
            self.out_emissions_all_users_labeled.append(
                {i: np.vstack([x[i] for x in ress]) for i in range(self.num_states)})

    def M_step(self):
        # optimize initial model
        X = self.inp_initials_all_users_labeled
        Y = np.exp(self.out_initials_all_users_labeled)
        if X.shape[0] == 0:
            logging.error(('No initial activity is labeled, cannot perform supervised IOHMM,\
                           try un/semi-supervised IOHMM.'))
        else:
            self.model_initial.fit(X, Y)

        # optimize transition models
        X = self.inp_transitions_all_users_labeled
        for st in range(self.num_states):
            Y = np.exp(self.out_transitions_all_users_labeled[st])
            if X[st].shape[0] == 0:
                logging.error(('No initial activity is labeled, cannot perform supervised IOHMM,\
                           try un/semi-supervised IOHMM.'))
            else:
                self.model_transition[st].fit(X[st], Y)

        # optimize emission models
        # print self.log_gammas
        # print np.exp(self.log_gammas)
        for emis in range(self.num_emissions):
            X = self.inp_emissions_all_users_labeled[emis]
            Y = self.out_emissions_all_users_labeled[emis]
            for st in range(self.num_states):
                if X[st].shape[0] == 0:
                    logging.error(('No initial activity is labeled, cannot perform supervised IOHMM,\
                           try un/semi-supervised IOHMM.'))
                else:
                    self.model_emissions[st][emis].fit(X[st], Y[st])

    def train(self):
        self._initialize_labeled()
        self.M_step()
