'''
This module implements IOHMM models:

(1) UnSupervisedIOHMM:
    Standard IOHMM with no ground truth label of hidden states.

(2) SemiSupervisedIOHMM:
    With a little ground truth labels of hidden states, use these labels to
    direct the learning process during EM.

(3) SupervisedIOHMM:
    With some ground truth labels of hidden states,
    use only ground truth labels to train. There are no iterations of EM.

The structure of the code is inspired by
depmixS4: An R Package for Hidden Markov Models:
https://cran.r-project.org/web/packages/depmixS4/vignettes/depmixS4.pdf

Features:
1. Can take a list of dataframes each representing a sequence.
2. Forward Backward algorithm fully vectorized.
3. Support json-serialization of the model so that model can be saved and loaded easily.
'''

from __future__ import division
from __future__ import absolute_import
from builtins import range
from builtins import object
from copy import deepcopy
import logging
import os
import warnings


import numpy as np


from .forward_backward import forward_backward
from .linear_models import (GLM, OLS, DiscreteMNL, CrossEntropyMNL)


warnings.simplefilter("ignore")
np.random.seed(0)
EPS = np.finfo(float).eps


class LinearModelLoader(object):
    """
    The map from data_type of a linear model
    ('GLM', 'OLS', 'DiscreteMNL', 'CrossEntropyMNL')
    to the correct class.
    """
    GLM = GLM
    OLS = OLS
    DiscreteMNL = DiscreteMNL
    CrossEntropyMNL = CrossEntropyMNL


class BaseIOHMM(object):
    """
    Base class for IOHMM models. Should not be directly called.
    Intended for subclassing.
    """

    def __init__(self, num_states=2):
        """
        Constructor
        Parameters
        ----------
        num_states: the number of hidden states
        """
        self.num_states = num_states
        self.trained = False

    def set_models(self, model_emissions,
                   model_initial=CrossEntropyMNL(),
                   model_transition=CrossEntropyMNL(), trained=False):
        """
        Set the initial probability model, transition probability models,
        and emission models.
        (1) model_initial: a linear model
        (2) model_transitions: a list of linear models, one for each hidden state.
        (3) model_emissions: a list of list of linear models,
                             the outer list is for each hidden state,
                             the inner list is for each emission model.
        Parameters
        ----------
        trained: a boolean indicating whether the models are already trained.
                 If the models are already trained, set the models directly,
                 otherwise initialize from empty linear models.
        if trained models, then the parameters are:
            model_initial: a linear model
            model_transitions: a list of linear models, one for each hidden state.
            model_emissions: a list of list of linear models,
                             the outer list is for each hidden state,
                             the inner list is for each emission model.
        otherwise:
            model_initial: the initial probability model (simply indicates its type)
            model_transition: the transition probability model (simply indicates its type)
            model_emissions: list of linear models, one for each emission.

        Notes
        -------
        Initial model and transition model must be CrossEntropyMNL models
        """
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
        """
        Set input covariates for initial, transition and emission models
        Parameters
        ----------
        covariates_initial: list of strings,
                            indicates the field names in the dataframe
                            to use as the independent variables.
        covariates_transition: list of strings,
                               indicates the field names in the dataframe
                               to use as the independent variables.
        covariates_emissions: list of list of strings, each outer list is for one emission model
                              and each inner list of strings
                              indicates the field names in the dataframe
                              to use as the independent variables.
        """
        self.covariates_initial = covariates_initial
        self.covariates_transition = covariates_transition
        self.covariates_emissions = covariates_emissions

    def set_outputs(self, responses_emissions):
        """
        Set output covariates for emission models
        Parameters
        ----------
        responses_emissions: list of list of strings, each outer list is for one emission
                             and each inner list of strings
                             indicates the field names in the dataframe
                             to use as the dependent variables.
        Notes
        ----------
        Emission model such as Multivariate OLS, CrossEntropyMNL
        will have multiple strings (columns) in the inner list.
        """
        self.responses_emissions = responses_emissions
        self.num_emissions = len(responses_emissions)

    def set_data(self, dfs):
        """
        Set data for the model
        Parameters
        ----------
        dfs: a list of dataframes, each df represents a sequence.
        Notes
        ----------
        The column names of each df must contains the covariates and response fields
        specified above.
        """
        raise NotImplementedError

    def _initialize(self, with_randomness=True):
        """
        Initialize
        (1) log_gammas: list of arrays, the state posterior probability for each sequence
        (2) log_epsilons: list of arrays, the state posterior 'transition' probability
            (joint probability of two consecutive points) for each sequence
        based on the ground truth labels supplied.
        (3) log likelihood as negative inifinity
        (4) inp_initials: list of arrays of shape (1, len(covariates_initial)),
                          the independent variables for the initial activity of each sequence.
        (5) inp_initials_all_sequences: array of shape(len(sequences), len(covariates_initial)),
                                        the concatenation of inp_initials.
        (6) inp_transitions: list of arrays of shape (df.shape[0]-1, len(covariates_transition)),
                             the independent variables for the transition activity of each sequence.
                             (Note that the -1 is for the first activity,
                             there is no transition, it is used for initial independent variable.)
        (7) inp_transitions_all_sequences: array of shape
                                           (sum(df.shape[0]-1 for df in dfs),
                                           len(covariates_transition)),
                                           the concatenation of inp_transitions.
        (8) inp_emissions: list of list of arrays of shape
                           (df.shape[0], len(covariates_emission[i])).
                           The outer list is for each df (sequence).
                           The inner list is for each emission model.
        (9) inp_emissions_all_sequences: list of array of shape
                                         (sum(df.shape[0] for df in dfs),
                                         len(covariates_emission[i])).
                                         The list is for each emission model.
                                         This is the concatenation of all sequences
                                         for each emission model.
        (10) out_emissions: list of list of arrays of shape
                            (df.shape[0], len(response_emission[i])).
                            The outer list is for each df (sequence).
                            The inner list is for each emission model.
        (11) out_emissions_all_sequences: list of array of shape
                                          (sum(df.shape[0] for df in dfs),
                                          len(response_emission[i])).
                                          The list is for each emission model.
                                          This is the concatenation of all sequences
                                          for each emission model.


        Parameters
        ----------
        with_randomness: After initializing log_gammas and log_epsilons,
                         there might be some states that no sample is associated with it.
                         In this case, should we add some random posterior probability to it,
                         so as to start the EM iterations?

                         For UnsupervisedIOHMM and SemiSupervisedIOHMM this is set to True,
                         since we want to use EM iterations to figure out the true posterior.

                         For SupervisedIOHMM, this is set to False,
                         since we only want to use labeled data for training.
        """
        def _initialize_log_gamma(df, log_state):
            """
            Initialize posterior probability for a dataframe and the log_state provided.
            Parameters
            ----------
            df: The dataframe for a sequence, actually we only need its length.
            log_state: a dictionary (int -> array of shape (num_states, )).
                       The log_state[t] is the ground truth hidden state array of time stamp t.
                       log_state[t][k] is 0 and log_state[t][~k] is -np.Infinity
                       if the hidden state of timestamp t is k.
            Returns:
            ----------
            log_gamma: array of shape (df.shape[0], num_states).
                       The posterior probability of each timestamp.
                       log_gamma[t][k] is 0 and log_gamma[t][k] is -np.Infinity
                       if the hidden state of timestamp t is k.
                       If at time stamp t there is no ground truth,
                       log_gamma[t] will be all -np.Infinity.
            """
            log_gamma = np.log(np.zeros((df.shape[0], self.num_states)))
            for time_stamp in log_state:
                log_gamma[time_stamp, :] = log_state[time_stamp]
            return log_gamma

        def _initialize_log_epsilon(df, log_state):
            """
            Initialize posterior joint probability of two consecutive timestamp
            for a dataframe and the log_state provided.
            Parameters
            ----------
            df: The dataframe for a sequence, actually we only need its length.
            log_state: a dictionary (int -> array of shape (num_states, )).
                       The log_state[i] is the ground truth hidden state array of time stamp i.
                       log_state[i][k] is 0 and log_state[i][~k] is -np.Infinity
                       if the hidden state of timestamp i is k.
            Returns:
            ----------
            log_epsilon: array of shape (df.shape[0] - 1, num_states, num_states).
                         The posterior joint probability of two consecutive points.
                         log_epsilon[t][k][j] is 0 and log_epsilon[t][~k][~j] is -np.Infinity
                         if the hidden state of timestamp t is k and
                         hidden state of timestamp t+1 is j.
                         If at time stamp t or t+1 there is no ground truth,
                         log_epsilon[t] will be all -np.Infinity.

            """
            log_epsilon = np.log(np.zeros((df.shape[0] - 1, self.num_states, self.num_states)))
            for time_stamp in log_state:
                if time_stamp + 1 in log_state:
                    # actually should find the index of 1
                    st = int(np.argmax(log_state[time_stamp]))
                    log_epsilon[time_stamp, st, :] = log_state[time_stamp + 1]
            return log_epsilon

        # initialize log_gammas
        self.log_gammas = [_initialize_log_gamma(df, log_state)
                           for df, log_state in self.dfs_logStates]
        # initialize log_epsilons
        self.log_epsilons = [_initialize_log_epsilon(df, log_state)
                             for df, log_state in self.dfs_logStates]
        if with_randomness:
            for st in range(self.num_states):
                if np.exp(np.hstack([lg[:, st] for lg in self.log_gammas])).sum() < EPS:
                    # there is no any sample associated with this state
                    for lg in self.log_gammas:
                        lg[:, st] = np.random.rand(lg.shape[0])
            for st in range(self.num_states):
                if np.exp(np.vstack([le[:, st, :] for le in self.log_epsilons])).sum() < EPS:
                    # there is no any sample associated with this state
                    for le in self.log_epsilons:
                        le[:, st, :] = np.random.rand(le.shape[0], self.num_states)

        # initialize log_likelihood
        self.log_likelihoods = [-np.Infinity for _ in range(self.num_seqs)]
        self.log_likelihood = -np.Infinity

        # initialize input/output covariates
        self.inp_initials = [np.array(df[self.covariates_initial].iloc[0]).reshape(
            1, -1).astype('float64') for df, log_state in self.dfs_logStates]
        self.inp_initials_all_sequences = np.vstack(self.inp_initials)

        self.inp_transitions = [np.array(df[self.covariates_transition].iloc[1:]).astype(
            'float64') for df, log_state in self.dfs_logStates]
        self.inp_transitions_all_sequences = np.vstack(self.inp_transitions)

        self.inp_emissions = [[np.array(df[cov]).astype('float64') for
                               cov in self.covariates_emissions]
                              for df, log_state in self.dfs_logStates]
        self.inp_emissions_all_sequences = [np.vstack([seq[emis] for
                                                       seq in self.inp_emissions]) for
                                            emis in range(self.num_emissions)]
        self.out_emissions = [[np.array(df[res]) for
                               res in self.responses_emissions]
                              for df, log_state in self.dfs_logStates]

        self.out_emissions_all_sequences = [np.vstack([seq[emis] for
                                                       seq in self.out_emissions]) for
                                            emis in range(self.num_emissions)]

    def E_step(self):
        """
        The Expectation step, Update
        (1) log_gammas: list of arrays, state posterior probability for each sequence
        (2) log_epsilons: list of arrays, state posterior 'transition' probability
            (joint probability of two consecutive points) for each sequence
        (3) log likelihood
        based on the model coefficients from last iteration,
        with respect to the ground truth hidden states if any.
        """
        self.log_gammas = []
        self.log_epsilons = []
        self.log_likelihoods = []
        for seq in range(self.num_seqs):
            n_records = self.dfs_logStates[seq][0].shape[0]
            # initial probability
            log_prob_initial = self.model_initial.predict_log_proba(
                self.inp_initials[seq]).reshape(self.num_states,)
            # transition probability
            log_prob_transition = np.zeros((n_records - 1, self.num_states, self.num_states))
            for st in range(self.num_states):
                log_prob_transition[:, st, :] = self.model_transition[st].predict_log_proba(
                    self.inp_transitions[seq])
            assert log_prob_transition.shape == (n_records - 1, self.num_states, self.num_states)
            # emission probability
            log_Ey = np.zeros((n_records, self.num_states))
            for emis in range(self.num_emissions):
                model_collection = [models[emis] for models in self.model_emissions]
                log_Ey += np.vstack([model.loglike_per_sample(
                    np.array(self.inp_emissions[seq][emis]).astype('float64'),
                    np.array(self.out_emissions[seq][emis])) for model in model_collection]).T
            # forward backward to calculate posterior
            log_gamma, log_epsilon, log_likelihood = forward_backward(
                log_prob_initial, log_prob_transition, log_Ey, self.dfs_logStates[seq][1])
            self.log_gammas.append(log_gamma)
            self.log_epsilons.append(log_epsilon)
            self.log_likelihoods.append(log_likelihood)
        self.log_likelihood = sum(self.log_likelihoods)

    def M_step(self):
        """
        The Maximization step, Update
        (1) model_initial: a linear model
        (2) model_transitions: a list of linear models, one for each hidden state.
        (3) model_emissions: a list of list of linear models,
                             the outer list is for each hidden state,
                             the inner list is for each emission model.
        based on the posteriors, and dependent/independent covariates.
        Notes:
        ----------
        In the emission models, if the sum of sample weight is zero,
        the linear model will raise ValueError.
        """

        # optimize initial model
        X = self.inp_initials_all_sequences
        Y = np.exp(np.vstack([lg[0, :].reshape(1, -1) for lg in self.log_gammas]))
        self.model_initial.fit(X, Y)

        # optimize transition models
        X = self.inp_transitions_all_sequences
        for st in range(self.num_states):
            Y = np.exp(np.vstack([eps[:, st, :] for eps in self.log_epsilons]))
            self.model_transition[st].fit(X, Y)

        # optimize emission models
        for emis in range(self.num_emissions):
            X = self.inp_emissions_all_sequences[emis]
            Y = self.out_emissions_all_sequences[emis]
            for st in range(self.num_states):
                sample_weight = np.exp(np.hstack([lg[:, st] for lg in self.log_gammas]))
                self.model_emissions[st][emis].fit(X, Y, sample_weight=sample_weight)

    def train(self):
        """
        The ieratioin of EM step,
        Notes:
        ----------
        For SupervisedIOHMM, max_EM_iter is 1, thus will only go through one iteration of EM step,
        which means that it will only use the ground truth hidden states to train.
        """
        for it in range(self.max_EM_iter):
            log_likelihood_prev = self.log_likelihood
            self.M_step()
            self.E_step()
            logging.info('log likelihood of iteration {0}: {1:.4f}'.format(it, self.log_likelihood))
            if abs(self.log_likelihood - log_likelihood_prev) < self.EM_tol:
                break
        self.trained = True

    def to_json(self, path):
        """
        Generate json object of the IOHMM model
        Parameters
        ----------
        path : the path to save the model
        Returns
        -------
        json_dict: a dictionary containing the attributes of the model
        """
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
        """
        Helper function to construct the IOHMM model used by from_json and from_config.
        Parameters
        ----------
        json_dict : the dictionary that specifies the model
        num_states: number of hidden states
        trained: a boolean indicating whether the models are already trained.
                 If the models are already trained, set the models directly,
                 otherwise initialize from empty linear models.
        if trained models, then the parameters are:
            model_initial: a linear model
            model_transitions: a list of linear models, one for each hidden state.
            model_emissions: a list of list of linear models,
                             the outer list is for each hidden state,
                             the inner list is for each emission model.
        otherwise:
            model_initial: the initial probability model (simply indicates its type)
            model_transition: the transition probability model (simply indicates its type)
            model_emissions: list of linear models, one for each emission.
        covariates_initial: list of strings,
                            each indicates the field name in the dataframe
                            to use as the independent variables.
        covariates_transition: list of strings,
                               each indicates the field name in the dataframe
                               to use as the independent variables.
        covariates_emissions: list of list of strings, each outer list is for one emission model
                              and each inner list of strings
                              indicates the field names in the dataframe
                              to use as the independent variables.
        responses_emissions: list of list of strings, each outer list is for one emission
                             and each inner list of strings
                             indicates the field names in the dataframe
                             to use as the dependent variables.
        Returns
        -------
        IOHMM object: an IOHMM object specified by the json_dict and other arguments
        """
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
        """
        Construct an IOHMM object from a json dictionary which specifies the structure of the model.
        Parameters
        ----------
        json_dict: a json dictionary containing the config/structure of the IOHMM.
        Returns
        -------
        IOHMM: an IOHMM object specified by the json_dict
        """
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
        """
        Construct an IOHMM object from a saved json dictionary.
        Parameters
        ----------
        json_dict: a json dictionary containing the attributes of the IOHMM.
        Returns
        -------
        IOHMM: an IOHMM object specified by the json_dict
        """
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
    """
    Unsupervised IOHMM models.
    This model is intended to be used when no ground truth hidden states are available.
    """

    def __init__(self, num_states=2, EM_tol=1e-4, max_EM_iter=100):
        """
        Constructor
        Parameters
        ----------
        num_states: the number of hidden states
        EM_tol: the tolerance of the EM iteration convergence
        max_EM_iter: the maximum number of EM iterations
        -------
        """
        super(UnSupervisedIOHMM, self).__init__(num_states=num_states)
        self.EM_tol = EM_tol
        self.max_EM_iter = max_EM_iter

    def set_data(self, dfs):
        """
        Set data for the model
        Constructs:
        ----------
        (1) num_seqs: number of seqences
        (2) dfs_logStates: list of (dataframe, log_state)
        (3) posteriors with randomness and input/output covariates
        Parameters
        ----------
        dfs: a list of dataframes, each df represents a sequence.
        Notes
        ----------
        The column names of each df must contains the covariates and response fields
        specified above.

        Since there are no ground truth hidden states, all log_state should be empty {}.
        """
        assert all([df.shape[0] > 0 for df in dfs])
        self.num_seqs = len(dfs)
        self.dfs_logStates = [[x, {}] for x in dfs]
        self._initialize(with_randomness=True)

    def to_json(self, path):
        """
        Generate json object of the UnSupervisedIOHMM/SemiSupervisedIOHMM model
        Parameters
        ----------
        path : the path to save the model
        Returns
        -------
        json_dict: a dictionary containing the attributes of the model
        """
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
        """
        Helper function to construct the UnSupervisedIOHMM/SemiSupervisedIOHMM model
        used by from_json and from_config.
        Parameters
        ----------
        json_dict : the dictionary that specifies the model
        num_states: number of hidden states
        trained: a boolean indicating whether the models are already trained.
                 If the models are already trained, set the models directly,
                 otherwise initialize from empty linear models.
        if trained models, then the parameters are:
            model_initial: a linear model
            model_transitions: a list of linear models, one for each hidden state.
            model_emissions: a list of list of linear models,
                             the outer list is for each hidden state,
                             the inner list is for each emission model.
        otherwise:
            model_initial: the initial probability model (simply indicates its type)
            model_transition: the transition probability model (simply indicates its type)
            model_emissions: list of linear models, one for each emission.
        covariates_initial: list of strings,
                            each indicates the field name in the dataframe
                            to use as the independent variables.
        covariates_transition: list of strings,
                               each indicates the field name in the dataframe
                               to use as the independent variables.
        covariates_emissions: list of list of strings, each outer list is for one emission model
                              and each inner list of strings
                              indicates the field names in the dataframe
                              to use as the independent variables.
        responses_emissions: list of list of strings, each outer list is for one emission
                             and each inner list of strings
                             indicates the field names in the dataframe
                             to use as the dependent variables.
        Returns
        -------
        IOHMM object: an IOHMM object specified by the json_dict and other arguments
        """
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
    """
    SemiSupervised IOHMM models.
    This model is intended to be used when there are some ground truth hidden states,
    but the user don't want to solely use these labeled data to train.
    """

    def set_data(self, dfs_states):
        """
        Set data for the model
        Constructs:
        ----------
        (1) num_seqs: number of seqences
        (2) dfs_logStates: list of (dataframe, log_state)
        (3) posteriors with randomness and input/output covariates
        Parameters
        ----------
        dfs_states: a list of (dataframes, states), each dataframe represents a sequence.
                    and states is a dictionary of (timestamp -> array of shape (num_states, ))
                    states[t][k] is 1 and states[t][~k] is 0 if the hidden state is k at
                    timestamp t.
        Notes
        ----------
        The column names of each df must contain the covariates and response fields
        specified above.
        """
        self.num_seqs = len(dfs_states)
        self.dfs_logStates = [[x[0], {k: np.log(x[1][k]) for k in x[1]}] for x in dfs_states]
        self._initialize(with_randomness=True)


class SupervisedIOHMM(BaseIOHMM):
    """
    SemiSupervised IOHMM models.
    This model is intended to be used when the user
    simply want to use ground truth hidden states to train the model
    """

    def __init__(self, num_states=2):
        """
        Constructor
        Parameters
        ----------
        num_states: the number of hidden states
        -------
        """
        super(SupervisedIOHMM, self).__init__(num_states=num_states)
        self.max_EM_iter = 1
        self.EM_tol = 0

    def set_data(self, dfs_states):
        """
        Set data for the model
        Constructs:
        ----------
        (1) num_seqs: number of seqences
        (2) dfs_logStates: list of (dataframe, log_state)
        (3) posteriors withOUT randomness and input/output covariates
        Parameters
        ----------
        dfs_states: a list of (dataframes, states), each dataframe represents a sequence.
                    and states if a dictionary of (timestamp -> array of shape (num_states, ))
                    states[t][k] is 1 and states[t][~k] is 0 if the hidden state is k at
                    timestamp t.
        Notes
        ----------
        The column names of each df must contains the covariates and response fields
        specified above.
        """
        self.num_seqs = len(dfs_states)
        self.dfs_logStates = [[x[0], {k: np.log(x[1][k]) for k in x[1]}] for x in dfs_states]
        self._initialize(with_randomness=False)
