from __future__ import  division
import numpy as np
from copy import deepcopy
from operator import add
import pandas as pd
from pyspark import SparkContext
import sys
sys.path.append('../auxiliary')
from SupervisedModels import *
from SemiHMM import *
import warnings
warnings.simplefilter("ignore")

## example:  ./bin/spark-submit Supervised_IOHMM_MapReduce.py

def EStepMap(df, log_state, model_initial, model_transition, model_emissions, 
    covariates_initial, covariates_transition, covariates_emissions, 
    responses_emissions, num_states, num_emissions):


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

    return [[log_gamma, log_epsilon, ll]]

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


class SupervisedIOHMMMapReduce:
    def __init__(self, num_states = 2):
        self.num_states = num_states

        
    def setModels(self, model_emissions, model_initial = MNLP(), model_transition = MNLP()):
        # initial model and transition model must be MNLP
        self.model_initial = model_initial
        self.model_transition = [deepcopy(model_transition) for i in range(self.num_states)]
        self.model_emissions = [deepcopy(model_emissions) for i in range(self.num_states)]
        self.num_emissions = len(model_emissions)
    
    def setData(self, rdd_dfs_states):
        # here the rdd is the rdd with (k, (df, state)) pairs that df is a dataframe, state is a dictionary
        self.num_seqs = rdd_dfs_states.count()
        self.dfs_logStates = rdd_dfs_states.mapValues(lambda v: (v[0],{k: np.log(v[1][k]) for k in v[1]}))
        
    
    def setInputs(self, covariates_initial, covariates_transition, covariates_emissions):

        num_states = self.num_states

        self.covariates_initial = covariates_initial
        self.covariates_transition = covariates_transition
        self.covariates_emissions = covariates_emissions
        # x = self.dfs_logStates.take(1)[0]

        self.inp_initials_all_users_labeled = self.dfs_logStates.map(lambda (k, v): inpInitialsLabeled(v[0], v[1], covariates_initial)).reduce(lambda a, b: np.vstack((a, b)))
        self.out_initials_all_users_labeled = self.dfs_logStates.map(lambda (k, v): outInitialsLabeled(v[0], v[1], num_states)).reduce(lambda a, b: np.vstack((a, b)))

        self.model_initial.coef = np.random.rand(self.inp_initials_all_users_labeled.shape[1]+self.model_initial.fit_intercept,self.num_states)
        
        self.inp_transitions_all_users_labeled = self.dfs_logStates.map(lambda (k, v): inpTransitionsLabeled(v[0], v[1], covariates_transition, num_states)).reduce(lambda a, b: {i: np.vstack((a[i], b[i])) for i in range(num_states)})
        self.out_transitions_all_users_labeled = self.dfs_logStates.map(lambda (k, v): outTransitionsLabeled(v[0], v[1], num_states)).reduce(lambda a, b: {i: np.vstack(([item for sublist in a[i] for item in sublist], [item for sublist in b[i] for item in sublist])) for i in range(num_states)})
        
        for st in range(self.num_states):
            self.model_transition[st].coef = np.random.rand(self.inp_transitions_all_users_labeled[st].shape[1]+self.model_transition[st].fit_intercept,self.num_states)
        self.inp_emissions_all_users_labeled = []
        for cov in covariates_emissions:
            self.inp_emissions_all_users_labeled.append(self.dfs_logStates.map(lambda (k, v): inpEmissionsLabeled(v[0], v[1], cov, num_states)).reduce(lambda a, b: {i: np.vstack((a[i], b[i])) for i in range(num_states)}))
        
        
    
    def setOutputs(self, responses_emissions):
        # output should be a list inidicating the columns of the dataframe
        num_states = self.num_states
        self.responses_emissions = responses_emissions
        self.out_emissions_all_users_labeled = []
        for res in responses_emissions:
            self.out_emissions_all_users_labeled.append(self.dfs_logStates.map(lambda (k, v): outEmissionsLabeled(v[0], v[1], res, num_states)).reduce(lambda a, b: {i: np.vstack((a[i], b[i])) for i in range(num_states)}))
        for i in range(self.num_states):
            for j in range(self.num_emissions):
                if isinstance(self.model_emissions[i][j], GLM):
                    self.model_emissions[i][j].coef = np.zeros((self.inp_emissions_all_users_labeled[j][i].shape[1]+self.model_emissions[i][j].fit_intercept,))
                    self.model_emissions[i][j].coef = np.random.rand(self.inp_emissions_all_users_labeled[j][i].shape[1]+self.model_emissions[i][j].fit_intercept,)
                    self.model_emissions[i][j].dispersion = 1
                if isinstance(self.model_emissions[i][j], LM):
                    if len(responses_emissions[j]) == 1:
                        self.model_emissions[i][j].coef = np.zeros((self.inp_emissions_all_users_labeled[j][i].shape[1]+self.model_emissions[i][j].fit_intercept,))
                        self.model_emissions[i][j].coef = np.random.rand(self.inp_emissions_all_users_labeled[j][i].shape[1]+self.model_emissions[i][j].fit_intercept,)
                        self.model_emissions[i][j].dispersion = 1
                    else:
                        self.model_emissions[i][j].coef = np.zeros((self.inp_emissions_all_users_labeled[j][i].shape[1]+self.model_emissions[i][j].fit_intercept, len(responses_emissions[j])))
                        self.model_emissions[i][j].coef = np.random.rand(self.inp_emissions_all_users_labeled[j][i].shape[1]+self.model_emissions[i][j].fit_intercept, len(responses_emissions[j]))
                        self.model_emissions[i][j].dispersion = np.eye(len(responses_emissions[j]))
                if isinstance(self.model_emissions[i][j], MNLD):
                    self.model_emissions[i][j].coef = np.zeros((self.inp_emissions_all_users_labeled[j][i].shape[1]+self.model_emissions[i][j].fit_intercept,np.unique(self.out_emissions_all_users[j]).shape[0]))
                    self.model_emissions[i][j].coef = np.random.rand(self.inp_emissions_all_users_labeled[j][i].shape[1]+self.model_emissions[i][j].fit_intercept,np.unique(self.out_emissions_all_users[j]).shape[0])
                    self.model_emissions[i][j].lb = LabelBinarizer().fit(self.out_emissions_all_users[j])
#                     self.model_emissions[i][j].n_targets = len(np.unique(self.out_emissions_all_users[j]))
                if isinstance(self.model_emissions[i][j], MNLP):
                    self.model_emissions[i][j].coef = np.zeros((self.inp_emissions_all_users_labeled[j][i].shape[1]+self.model_emissions[i][j].fit_intercept,len(responses_emissions[j])))
                    self.model_emissions[i][j].coef = np.random.rand(self.inp_emissions_all_users_labeled[j][i].shape[1]+self.model_emissions[i][j].fit_intercept,len(responses_emissions[j]))

    
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

        model_initial = self.model_initial
        model_transition = self.model_transition
        model_emissions = self.model_emissions
        covariates_initial = self.covariates_initial
        covariates_transition = self.covariates_transition
        covariates_emissions = self.covariates_emissions
        responses_emissions = self.responses_emissions
        num_states = self.num_states
        num_emissions = self.num_emissions


        rdd_E = self.dfs_logStates.mapValues(lambda v: EStepMap(v[0],v[1], model_initial, model_transition, model_emissions, 
            covariates_initial, covariates_transition, covariates_emissions, 
            responses_emissions, num_states, num_emissions))
        posteriors = rdd_E.map(lambda x: x[1]).reduce(add)

        # rdd_E = self.dfs.map(lambda (k, v): EStepMap(v, model_initial, model_transition, model_emissions, 
        #     covariates_initial, covariates_transition, covariates_emissions, 
        #     responses_emissions, num_states, num_emissions))

        # posteriors = rdd_E.collect()

        self.log_gammas= [x[0] for x in posteriors]
        self.log_epsilons= [x[1] for x in posteriors]
        self.lls = [x[2] for x in posteriors]
        self.ll = sum(self.lls)

        
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

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print >> sys.stderr, "Usage: Supervised_IOHMM_MapReduce<file>"
        exit(-1)

    sc = SparkContext(appName="Python_Supervised_IOHMM_MapReduce", pyFiles=[
        '../auxiliary/SemiHMM.py',
        '../auxiliary/SupervisedModels.py',
        '../auxiliary/family.py'])


    speed = pd.read_csv('../data/speed.csv')
    states = {}
    corr = np.array(speed['corr'])
    for i in range(len(corr)):
        state = np.zeros((2,))
        if corr[i] == 'cor':
            states[i] = np.array([0,1])
        else:
            states[i] = np.array([1,0])

    indexes = [(1,1), (2,1)]
    RDD = sc.parallelize(indexes)
    dfs_states = RDD.mapValues(lambda v: [speed, states])
    
    SHMM = SupervisedIOHMMMapReduce(num_states=2)
    SHMM.setData(dfs_states)
    SHMM.setModels(model_emissions = [LM()], model_transition=MNLP(solver='lbfgs'))
    SHMM.setInputs(covariates_initial = [], covariates_transition = [], covariates_emissions = [[]])
    SHMM.setOutputs([['rt']])


    SHMM.train()
    SHMM.EStep()
    print SHMM.ll
    print 'done'



