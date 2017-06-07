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

# example:

class UnSupervisedIOHMM(object):
    def __init__(self, num_states = 2, EM_tol = 1e-4, max_EM_iter = 100):
        self.num_states = num_states
        self.EM_tol = EM_tol
        self.max_EM_iter = max_EM_iter
        self.has_params = False
        
    def setModels(self, model_emissions, model_initial = MNLP(), model_transition = MNLP()):
        # initial model and transition model must be MNLP
        self.model_initial = model_initial
        self.model_transition = [deepcopy(model_transition) for i in range(self.num_states)]
        self.model_emissions = [deepcopy(model_emissions) for i in range(self.num_states)]
        self.num_emissions = len(model_emissions)

    
    def setInputs(self, covariates_initial, covariates_transition, covariates_emissions):
        self.covariates_initial = covariates_initial
        self.covariates_transition = covariates_transition
        self.covariates_emissions = covariates_emissions
        # input should be a list inidicating the columns of the dataframe

        
    def setOutputs(self, responses_emissions):
        # output should be a list inidicating the columns of the dataframe
        self.responses_emissions = responses_emissions
        

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
        self.has_params = True

    def setData(self, dfs):
        self.num_seqs = len(dfs)
        self.dfs_logStates = map(lambda x: [x, {}], dfs)
        self.initIO()


    def initIO(self):
        self.log_gammas = [LogGammaMap(df, log_state, self.num_states) for df, log_state in self.dfs_logStates]

        inp_initials = [np.array(df[self.covariates_initial].iloc[0]).reshape(1,-1).astype('float64') for df, log_state in self.dfs_logStates]
        self.inp_initials_all_users = np.vstack(inp_initials)
        
        inp_transitions = [np.array(df[self.covariates_transition].iloc[1:]).astype('float64') for df, log_state in self.dfs_logStates]
        self.inp_transitions_all_users = np.vstack(inp_transitions)

        inp_emissions = []
        self.inp_emissions_all_users = []
        for cov in self.covariates_emissions:
            inp_emissions.append([np.array(df[cov]).astype('float64') for df, log_state in self.dfs_logStates])
        for covs in inp_emissions:
            self.inp_emissions_all_users.append(np.vstack(covs))

        out_emissions = []
        self.out_emissions_all_users = []
        for res in self.responses_emissions:
            out_emissions.append([np.array(df[res]) for df, log_state in self.dfs_logStates])
        for ress in out_emissions:
            self.out_emissions_all_users.append(np.vstack(ress))


    
    def initParams(self):
        self.model_initial.coef = np.random.rand(len(self.covariates_initial)+self.model_initial.fit_intercept,self.num_states)
        for st in range(self.num_states):
            self.model_transition[st].coef = np.random.rand(len(self.covariates_transition)+self.model_transition[st].fit_intercept,self.num_states)
        for i in range(self.num_states):
            sample_weight = np.exp(np.hstack([lg[:,i] for lg in self.log_gammas]))
            if sample_weight.sum() > 0:
                for j in range(self.num_emissions):
                    X = self.inp_emissions_all_users[j]
                    Y = self.out_emissions_all_users[j]
                    self.model_emissions[i][j].fit(X, Y, sample_weight = sample_weight)
            for j in range(self.num_emissions):
                if isinstance(self.model_emissions[i][j], GLM):
                    self.model_emissions[i][j].coef = np.random.rand(len(self.covariates_emissions[j])+self.model_emissions[i][j].fit_intercept,)
                    self.model_emissions[i][j].dispersion = 1
                if isinstance(self.model_emissions[i][j], LM):
                    if len(self.responses_emissions[j]) == 1:
                        self.model_emissions[i][j].coef = np.random.rand(len(self.covariates_emissions[j])+self.model_emissions[i][j].fit_intercept,)
                        self.model_emissions[i][j].dispersion = 1
                    else:
                        self.model_emissions[i][j].coef = np.random.rand(len(self.covariates_emissions[j])+self.model_emissions[i][j].fit_intercept, len(self.responses_emissions[j]))
                        self.model_emissions[i][j].dispersion = np.eye(len(self.responses_emissions[j]))
                if isinstance(self.model_emissions[i][j], MNLD):
                    self.model_emissions[i][j].coef = np.random.rand(len(self.covariates_emissions[j])+self.model_emissions[i][j].fit_intercept,np.unique(self.out_emissions_all_users[j]).shape[0])
                    self.model_emissions[i][j].lb = LabelBinarizer().fit(self.out_emissions_all_users[j])
                if isinstance(self.model_emissions[i][j], MNLP):
                    self.model_emissions[i][j].coef = np.random.rand(len(self.covariates_emissions[j])+self.model_emissions[i][j].fit_intercept,len(self.responses_emissions[j]))
        self.has_params = True

    def EStep(self):

        posteriors = [EStepMap(df, self.model_initial, self.model_transition, self.model_emissions, 
            self.covariates_initial, self.covariates_transition, self.covariates_emissions, 
            self.responses_emissions, self.num_states, self.num_emissions, log_state) for df, log_state in self.dfs_logStates]

        self.log_gammas= [x[0] for x in posteriors]
        self.log_epsilons= [x[1] for x in posteriors]
        self.lls = [x[2] for x in posteriors]
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
        if not self.has_params:
            self.initParams()
        self.EStep()
        for it in range(self.max_EM_iter):
            prev_ll = self.ll
            self.out_initials_all_users = np.exp(np.vstack([lg[0,:].reshape(1,-1) for lg in self.log_gammas]))
            self.out_transitions_all_users = [np.exp(np.vstack([eps[:,st,:] for eps in self.log_epsilons])) for st in range(self.num_states)]
            self.MStep()
            self.EStep()
            print self.ll
            if abs(self.ll-prev_ll) < self.EM_tol:
                break

        self.converged = it < self.max_EM_iter


class SemiSupervisedIOHMM(UnSupervisedIOHMM):
    def setData(self, dfs_states):
        self.num_seqs = len(dfs_states)
        self.dfs_logStates = map(lambda x: [x[0], {k: np.log(x[1][k]) for k in x[1]}], dfs_states)
        self.initIO()


class SupervisedIOHMM(SemiSupervisedIOHMM):
    def __init__(self, num_states = 2):
        self.num_states = num_states

    def setData(self, dfs_states):
        # here the rdd is the rdd with (k, (df, state)) pairs that df is a dataframe, state is a dictionary
        self.num_seqs = len(dfs_states)
        self.dfs_logStates = map(lambda x: [x[0], {k: np.log(x[1][k]) for k in x[1]}], dfs_states)
        self.initIOLabeled()
        # for labeled data
    def initIOLabeled(self):
        inp_initials = [inpInitialsLabeled(df, log_state, self.covariates_initial) for df, log_state in self.dfs_logStates]
        self.inp_initials_all_users_labeled = np.vstack(inp_initials)
        out_initials = [outInitialsLabeled(df, log_state, self.num_states) for df, log_state in self.dfs_logStates]
        self.out_initials_all_users_labeled = np.vstack(out_initials)

        inp_transitions = [inpTransitionsLabeled(df, log_state, self.covariates_transition, self.num_states) for df, log_state in self.dfs_logStates]
        self.inp_transitions_all_users_labeled = {i: np.vstack([x[i] for x in inp_transitions]) for i in range(self.num_states)}
        out_transitions = [outTransitionsLabeled(df, log_state, self.num_states) for df, log_state in self.dfs_logStates]
        self.out_transitions_all_users_labeled = {i: np.vstack([item for sublist in out_transitions for item in sublist[i]]) for i in range(self.num_states)}

        inp_emissions = []
        self.inp_emissions_all_users_labeled = []
        for cov in self.covariates_emissions:
            inp_emissions.append([inpEmissionsLabeled(df, log_state, cov, self.num_states) for df, log_state in self.dfs_logStates])
        for covs in inp_emissions:
            self.inp_emissions_all_users_labeled.append({i: np.vstack([x[i] for x in covs]) for i in range(self.num_states)})
        
        out_emissions = []
        self.out_emissions_all_users_labeled = []
        for res in self.responses_emissions:
            out_emissions.append([outEmissionsLabeled(df, log_state, res, self.num_states) for df, log_state in self.dfs_logStates])
        for ress in out_emissions:
            self.out_emissions_all_users_labeled.append({i: np.vstack([x[i] for x in ress]) for i in range(self.num_states)})

    
    def MStep(self):
        # optimize initial model
        X = self.inp_initials_all_users_labeled
        Y = np.exp(self.out_initials_all_users_labeled)
        if X.shape[0] == 0:
            self.model_initial.coef = np.zeros((self.inp_initials_all_users_labeled.shape[1]+self.model_initial.fit_intercept,self.num_states))
        else:
            self.model_initial.fit(X, Y)
        
        # optimize transition models
        X = self.inp_transitions_all_users_labeled
        for st in range(self.num_states):
            Y = np.exp(self.out_transitions_all_users_labeled[st])
            if X[st].shape[0] == 0:
                self.model_transition[st].coef = np.zeros((self.inp_transitions_all_users_labeled[st].shape[1]+self.model_transition[st].fit_intercept,self.num_states))
            else:
                self.model_transition[st].fit(X[st], Y)
        
        # optimize emission models
        # print self.log_gammas
        # print np.exp(self.log_gammas)
        for emis in range(self.num_emissions):
            X = self.inp_emissions_all_users_labeled[emis]
            Y = self.out_emissions_all_users_labeled[emis]
            for st in range(self.num_states):
                if X[st].shape[0] != 0:
                    self.model_emissions[st][emis].fit(X[st], Y[st])
    
    def train(self):
        self.MStep()


class UnSupervisedIOHMMMapReduce(UnSupervisedIOHMM):
    def setData(self, rdd_dfs):
        self.num_seqs = rdd_dfs.count()
        self.dfs_logStates = rdd_dfs.mapValues(lambda v: (v,{}))
        self.initIO()

    def initIO(self):
        num_states = self.num_states
        covariates_initial, covariates_transition = self.covariates_initial, self.covariates_transition 
        self.log_gammas = self.dfs_logStates.map(lambda (k, v): LogGammaMap(v[0], v[1], num_states)).collect()

        self.inp_initials_all_users = self.dfs_logStates.map(lambda (k, v): np.array(v[0][covariates_initial].iloc[0]).reshape(1,-1).astype('float64')).reduce(lambda a, b: np.vstack((a, b)))
        self.inp_transitions_all_users = self.dfs_logStates.map(lambda (k, v): np.array(v[0][covariates_transition].iloc[1:]).astype('float64')).reduce(lambda a, b: np.vstack((a, b)))
        self.inp_emissions_all_users = []
        for cov in self.covariates_emissions:
            self.inp_emissions_all_users.append(self.dfs_logStates.map(lambda (k, v): np.array(v[0][cov]).astype('float64')).reduce(lambda a, b: np.vstack((a, b))))
        
        self.out_emissions_all_users = []
        for res in self.responses_emissions:
            self.out_emissions_all_users.append(self.dfs_logStates.map(lambda (k, v): np.array(v[0][res])).reduce(lambda a, b: np.vstack((a, b))))
    
    def EStep(self):        
        model_initial, model_transition, model_emissions = self.model_initial, self.model_transition, self.model_emissions
        covariates_initial, covariates_transition, covariates_emissions = self.covariates_initial, self.covariates_transition, self.covariates_emissions
        responses_emissions, num_states, num_emissions = self.responses_emissions, self.num_states, self.num_emissions
        posteriors = self.dfs_logStates.map(lambda (k, v): EStepMap(v[0], model_initial, model_transition, model_emissions, 
            covariates_initial, covariates_transition, covariates_emissions, 
            responses_emissions, num_states, num_emissions, v[1])).collect()
        self.log_gammas= [x[0] for x in posteriors]
        self.log_epsilons= [x[1] for x in posteriors]
        self.lls = [x[2] for x in posteriors] 
        self.ll = sum(self.lls)


class SemiSupervisedIOHMMMapReduce(UnSupervisedIOHMMMapReduce):
    def setData(self, rdd_dfs_states):
        self.num_seqs = rdd_dfs_states.count()
        self.dfs_logStates = rdd_dfs_states.mapValues(lambda v: (v[0],{k: np.log(v[1][k]) for k in v[1]}))
        self.initIO()
    

class SupervisedIOHMMMapReduce(SemiSupervisedIOHMMMapReduce, SupervisedIOHMM):
    def __init__(self, num_states = 2):
        self.num_states = num_states

    def setData(self, rdd_dfs_states):
        self.num_seqs = rdd_dfs_states.count()
        self.dfs_logStates = rdd_dfs_states.mapValues(lambda v: (v[0],{k: np.log(v[1][k]) for k in v[1]}))
        self.initIOLabeled()

    def initIOLabeled(self):
        num_states = self.num_states
        covariates_initial, covariates_transition = self.covariates_initial, self.covariates_transition
        self.inp_initials_all_users_labeled = self.dfs_logStates.map(lambda (k, v): inpInitialsLabeled(v[0], v[1], covariates_initial)).reduce(lambda a, b: np.vstack((a, b)))
        self.out_initials_all_users_labeled = self.dfs_logStates.map(lambda (k, v): outInitialsLabeled(v[0], v[1], num_states)).reduce(lambda a, b: np.vstack((a, b)))
        self.inp_transitions_all_users_labeled = self.dfs_logStates.map(lambda (k, v): inpTransitionsLabeled(v[0], v[1], covariates_transition, num_states)).reduce(lambda a, b: {i: np.vstack((a[i], b[i])) for i in range(num_states)})
        self.out_transitions_all_users_labeled = self.dfs_logStates.map(lambda (k, v): outTransitionsLabeled(v[0], v[1], num_states)).reduce(lambda a, b: {i: np.vstack(([item for sublist in a[i] for item in sublist], [item for sublist in b[i] for item in sublist])) for i in range(num_states)})
        
        self.inp_emissions_all_users_labeled = []
        for cov in self.covariates_emissions:
            self.inp_emissions_all_users_labeled.append(self.dfs_logStates.map(lambda (k, v): inpEmissionsLabeled(v[0], v[1], cov, num_states)).reduce(lambda a, b: {i: np.vstack((a[i], b[i])) for i in range(num_states)}))
        
        self.out_emissions_all_users_labeled = []
        for res in self.responses_emissions:
            self.out_emissions_all_users_labeled.append(self.dfs_logStates.map(lambda (k, v): outEmissionsLabeled(v[0], v[1], res, num_states)).reduce(lambda a, b: {i: np.vstack((a[i], b[i])) for i in range(num_states)}))
    
    def EStep(self):
        SemiSupervisedIOHMMMapReduce.EStep(self)

    def train(self):
        SupervisedIOHMM.train(self)

def LogGammaMap(df, log_state, num_states):
    log_gamma = np.log(np.zeros((df.shape[0], num_states)))
    for k in log_state:
        log_gamma[k,:] = log_state[k]
    return log_gamma

def EStepMap(df, model_initial, model_transition, model_emissions, 
    covariates_initial, covariates_transition, covariates_emissions, 
    responses_emissions, num_states, num_emissions, log_state={}):


    n_records = df.shape[0]
    log_prob_initial = model_initial.predict_log_probability(np.array(df[covariates_initial].iloc[0]).reshape(1,-1).astype('float64')).reshape(num_states,)
    log_prob_transition = np.zeros((n_records - 1, num_states, num_states))
    for st in range(num_states):
         log_prob_transition[:,st,:] = model_transition[st].predict_log_probability(np.array(df[covariates_transition].iloc[1:]).astype('float64')) 
    assert log_prob_transition.shape == (n_records-1,num_states,num_states)   
    log_Ey = np.zeros((n_records,num_states))
    for emis in range(num_emissions):
        model_collection = [models[emis] for models in model_emissions]
        log_Ey += np.vstack([model.log_probability(np.array(df[covariates_emissions[emis]]).astype('float64'),
                                                       np.array(df[responses_emissions[emis]])) for model in model_collection]).T

    log_gamma, log_epsilon, ll = calHMM(log_prob_initial, log_prob_transition, log_Ey, log_state)

    return [log_gamma, log_epsilon, ll]


def inpInitialsLabeled(df, log_state, covariates_initial):
    if 0 in log_state:
        return np.array(df[covariates_initial].iloc[0]).reshape(1,-1).astype('float64')
    else:
        return np.empty((0, len(covariates_initial)), dtype=float)

def outInitialsLabeled(df, log_state, num_states):
    if 0 in log_state:
        return log_state[0].reshape(1,-1).astype('float64')
    else:
        return np.empty((0, num_states), dtype=float)

def inpTransitionsLabeled(df, log_state, covariates_transition, num_states):
    ind = {}
    inp = {}
    for i in range(num_states):
        ind[i] = []
    for i in log_state:
        if i+1 in log_state:
            st = int(np.argmax(log_state[i]))
            ind[st].append(i+1)
    
    for i in range(num_states):
        inp[i] = np.array(df.ix[ind[i], covariates_transition]).astype('float64')

    return inp

def outTransitionsLabeled(df, log_state, num_states):
    ind = {}
    out = {}
    for i in range(num_states):
        ind[i] = []
    for i in log_state:
        if i+1 in log_state:
            st = int(np.argmax(log_state[i]))
            ind[st].append(i+1)
    for i in range(num_states):
        out[i] = [log_state[k].reshape(1,-1).astype('float64') for k in ind[i]]
        if out[i] == []:
            out[i] = [np.empty((0, num_states), dtype=float)]

    return out

def inpEmissionsLabeled(df, log_state, cov, num_states):
    ind = {}
    inp = {}
    for i in range(num_states):
        ind[i] = []
    for i in log_state:
        st = int(np.argmax(log_state[i]))
        ind[st].append(i)
    for i in range(num_states):
        inp[i] = np.array(df.ix[ind[i], cov]).astype('float64')

    return inp

def outEmissionsLabeled(df, log_state, res, num_states):
    ind = {}
    out = {}
    for i in range(num_states):
        ind[i] = []
    for i in log_state:
        st = int(np.argmax(log_state[i]))
        ind[st].append(i)
    for i in range(num_states):
        out[i] = np.array(df.ix[ind[i], res]).astype('float64')

    return out


if __name__ == "__main__":
    if len(sys.argv) != 1:
        print >> sys.stderr, "Usage: UnSupervised_IOHMM<file>"
        exit(-1)


    speed = pd.read_csv('../data/speed.csv')
    dfs = [speed, speed]
    
    SHMM = UnSupervisedIOHMM(num_states=2, max_EM_iter=100, EM_tol=1e-4)
    SHMM.setModels(model_emissions = [LM()], model_transition=MNLP(solver='lbfgs'))
    SHMM.setInputs(covariates_initial = [], covariates_transition = [], covariates_emissions = [[]])
    SHMM.setOutputs([['rt']])
    SHMM.setData(dfs)
    SHMM.train()
    print 'done'


    
    