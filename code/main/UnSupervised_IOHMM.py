from __future__ import  division
import numpy as np
from copy import deepcopy
import pandas as pd
import sys
sys.path.append('../auxiliary')
from SupervisedModels import *
from HMM import *
import warnings
warnings.simplefilter("ignore")

## example:  python Unsupervised_IOHMM.py

class UnSupervisedIOHMM:
    def __init__(self, num_states = 2, EM_tol = 1e-4, max_EM_iter = 100):
        self.num_states = num_states
        self.EM_tol = EM_tol
        self.max_EM_iter = max_EM_iter
        
    def setModels(self, model_emissions, model_initial = MNLP(), model_transition = MNLP()):
        # initial model and transition model must be MNLP
        self.model_initial = model_initial
        self.model_transition = [deepcopy(model_transition) for i in range(self.num_states)]
        self.model_emissions = [deepcopy(model_emissions) for i in range(self.num_states)]
        self.num_emissions = len(model_emissions)
    
    def setData(self, dfs):
        self.num_seqs = len(dfs)
        self.dfs = dfs
        
    
    def setInputs(self, covariates_initial, covariates_transition, covariates_emissions):
        self.covariates_initial = covariates_initial
        self.covariates_transition = covariates_transition
        self.covariates_emissions = covariates_emissions
        # input should be a list inidicating the columns of the dataframe
        self.inp_initials = [np.array(df[covariates_initial].iloc[0]).reshape(1,-1).astype('float64') for df in self.dfs]
        self.inp_initials_all_users = np.vstack(self.inp_initials)
        self.model_initial.coef = np.zeros((self.inp_initials_all_users.shape[1]+self.model_initial.fit_intercept,self.num_states))
        self.model_initial.coef = np.random.rand(self.inp_initials_all_users.shape[1]+self.model_initial.fit_intercept,self.num_states)
        
        self.inp_transitions = [np.array(df[covariates_transition].iloc[1:]).astype('float64') for df in self.dfs]
        self.inp_transitions_all_users = np.vstack(self.inp_transitions)
        
        for st in range(self.num_states):
            self.model_transition[st].coef = np.zeros((self.inp_transitions_all_users.shape[1]+self.model_transition[st].fit_intercept,self.num_states))
            self.model_transition[st].coef = np.random.rand(self.inp_transitions_all_users.shape[1]+self.model_transition[st].fit_intercept,self.num_states)
        self.inp_emissions = []
        self.inp_emissions_all_users = []
        for cov in covariates_emissions:
            self.inp_emissions.append([np.array(df[cov]).astype('float64') for df in self.dfs])
        for covs in self.inp_emissions:
            self.inp_emissions_all_users.append(np.vstack(covs))
        
        
    
    def setOutputs(self, responses_emissions):
        # output should be a list inidicating the columns of the dataframe
        self.responses_emissions = responses_emissions
        self.out_emissions = []
        self.out_emissions_all_users = []
        for res in responses_emissions:
            self.out_emissions.append([np.array(df[res]) for df in self.dfs])
        for ress in self.out_emissions:
            self.out_emissions_all_users.append(np.vstack(ress))
        for i in range(self.num_states):
            for j in range(self.num_emissions):
                if isinstance(self.model_emissions[i][j], GLM):
                    self.model_emissions[i][j].coef = np.random.rand(self.inp_emissions_all_users[j].shape[1]+self.model_emissions[i][j].fit_intercept,)
                    self.model_emissions[i][j].dispersion = 1
                if isinstance(self.model_emissions[i][j], LM):
                    if len(responses_emissions[j]) == 1:
                        self.model_emissions[i][j].coef = np.random.rand(self.inp_emissions_all_users[j].shape[1]+self.model_emissions[i][j].fit_intercept,)
                        self.model_emissions[i][j].dispersion = 1
                    else:
                        self.model_emissions[i][j].coef = np.random.rand(self.inp_emissions_all_users[j].shape[1]+self.model_emissions[i][j].fit_intercept, len(responses_emissions[j]))
                        self.model_emissions[i][j].dispersion = np.eye(len(responses_emissions[j]))
                if isinstance(self.model_emissions[i][j], MNLD):
                    self.model_emissions[i][j].coef = np.random.rand(self.inp_emissions_all_users[j].shape[1]+self.model_emissions[i][j].fit_intercept,np.unique(self.out_emissions_all_users[j]).shape[0])
                    self.model_emissions[i][j].lb = LabelBinarizer().fit(self.out_emissions_all_users[j])
                if isinstance(self.model_emissions[i][j], MNLP):
                    self.model_emissions[i][j].coef = np.random.rand(self.inp_emissions_all_users[j].shape[1]+self.model_emissions[i][j].fit_intercept,len(responses_emissions[j]))
    


    def setParams(self, model_initial_coef, model_transition_coef, model_emissions_coef, model_emissions_dispersion):
        self.model_initial.coef = model_initial_coef
        for st in range(self.num_states):
            self.model_transition[st].coef = model_transition_coef[st]
        for i in range(self.num_states):
            for j in range(self.num_emissions):
                self.model_emissions[i][j].coef = model_emissions_coef[i][j]
                try:
                    self.model_emissions[i][j].dispersion = model_emissions_dispersion[i][j]
                except:
                    pass


    def EStep(self):
        self.log_gammas = []
        self.log_epsilons = []
        self.lls = []
        
        for seq in range(self.num_seqs):
            n_records = self.dfs[seq].shape[0]
            log_prob_initial = self.model_initial.predict_log_probability(self.inp_initials[seq]).reshape(self.num_states,)
            assert log_prob_initial.shape == (self.num_states,)
            log_prob_transition = np.zeros((n_records - 1, self.num_states, self.num_states))
            for st in range(self.num_states):
                 log_prob_transition[:,st,:] = self.model_transition[st].predict_log_probability(self.inp_transitions[seq]) 
            assert log_prob_transition.shape == (n_records-1,self.num_states,self.num_states)
            
            log_Ey = np.zeros((n_records,self.num_states))
            for emis in range(self.num_emissions):
                model_collection = [models[emis] for models in self.model_emissions]
                log_Ey += np.vstack([model.log_probability(self.inp_emissions[emis][seq],
                                                           self.out_emissions[emis][seq]) for model in model_collection]).T

            
            log_gamma, log_epsilon, ll = calHMM(log_prob_initial, log_prob_transition, log_Ey)
            self.log_gammas.append(log_gamma)
            self.log_epsilons.append(log_epsilon)
            self.lls.append(ll)
            self.ll = sum(self.lls)

        
    def MStep(self):
        # optimize initial model
        X = self.inp_initials_all_users
        Y = np.exp(np.vstack([lg[0,:].reshape(1,-1) for lg in self.log_gammas]))
        logY = np.vstack([lg[0,:].reshape(1,-1) for lg in self.log_gammas])
        self.model_initial.fit(X, Y)
        
        # optimize transition models
        X = self.inp_transitions_all_users
        for st in range(self.num_states):
            Y = np.exp(np.vstack([eps[:,st,:] for eps in self.log_epsilons]))
            logY = np.vstack([eps[:,st,:] for eps in self.log_epsilons])
            self.model_transition[st].fit(X, Y)
        
        # optimize emission models
        for emis in range(self.num_emissions):
            X = self.inp_emissions_all_users[emis]
            Y = self.out_emissions_all_users[emis]
            for st in range(self.num_states):
                sample_weight = np.exp(np.hstack([lg[:,st] for lg in self.log_gammas]))
                self.model_emissions[st][emis].fit(X, Y, sample_weight = sample_weight)
        
    
    def train(self):
        self.EStep()
        for it in range(self.max_EM_iter):
            prev_ll = self.ll
            self.MStep()
            self.EStep()
            print self.ll
            if abs(self.ll-prev_ll) < self.EM_tol:
                break

        self.converged = it < self.max_EM_iter

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print >> sys.stderr, "Usage: UnSupervised_IOHMM<file>"
        exit(-1)


    speed = pd.read_csv('../data/speed.csv')
    dfs = [speed, speed]
    
    SHMM = UnSupervisedIOHMM(num_states=2, max_EM_iter=100, EM_tol=1e-4)
    SHMM.setData(dfs)
    SHMM.setModels(model_emissions = [LM()], model_transition=MNLP(solver='lbfgs'))
    SHMM.setInputs(covariates_initial = [], covariates_transition = [], covariates_emissions = [[]])
    SHMM.setOutputs([['rt']])
    SHMM.train()
    print 'done'


    
    