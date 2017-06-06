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




## adapted code to MapReduce
## will be improved if Spark MLLIB support linear models with sample weights
class SemiSupervisedIOHMMMapReduce:
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
    
    def setData(self, rdd_dfs_states):
        # here the rdd is the rdd with (k, (df, state)) pairs that df is a dataframe, state is a dictionary
        self.num_seqs = rdd_dfs_states.count()
        self.dfs_logStates = rdd_dfs_states.mapValues(lambda v: (v[0],{k: np.log(v[1][k]) for k in v[1]}))
        dfs_logStates = self.dfs_logStates.collect()
        self.log_gammas = [np.log(np.zeros((df.shape[0], self.num_states))) for df, log_state in dfs_logStates]
        for i, (df, log_state) in enumerate(dfs_logStates):
            for k in log_state:
                self.log_gammas[i][k,:] = log_state[k]
    
    def setInputs(self, covariates_initial, covariates_transition, covariates_emissions):

        self.covariates_initial = covariates_initial
        self.covariates_transition = covariates_transition
        self.covariates_emissions = covariates_emissions
        # x = self.dfs_logStates.take(1)[0]

        self.inp_initials_all_users = self.dfs_logStates.map(lambda (k, v): np.array(v[0][covariates_initial].iloc[0]).reshape(1,-1).astype('float64')).reduce(lambda a, b: np.vstack((a, b)))

        self.model_initial.coef = np.random.rand(self.inp_initials_all_users.shape[1]+self.model_initial.fit_intercept,self.num_states)
        
        self.inp_transitions_all_users = self.dfs_logStates.map(lambda (k, v): np.array(v[0][covariates_transition].iloc[1:]).astype('float64')).reduce(lambda a, b: np.vstack((a, b)))
        
        for st in range(self.num_states):
            self.model_transition[st].coef = np.random.rand(self.inp_transitions_all_users.shape[1]+self.model_transition[st].fit_intercept,self.num_states)
        self.inp_emissions_all_users = []
        for cov in covariates_emissions:
            self.inp_emissions_all_users.append(self.dfs_logStates.map(lambda (k, v): np.array(v[0][cov]).astype('float64')).reduce(lambda a, b: np.vstack((a, b))))
        
        
    
    def setOutputs(self, responses_emissions):
        # output should be a list inidicating the columns of the dataframe
        self.responses_emissions = responses_emissions
        self.out_emissions_all_users = []
        for res in responses_emissions:
            self.out_emissions_all_users.append(self.dfs_logStates.map(lambda (k, v): np.array(v[0][res])).reduce(lambda a, b: np.vstack((a, b))))
        for i in range(self.num_states):
            sample_weight = np.exp(np.hstack([lg[:,i] for lg in self.log_gammas]))
            if sample_weight.sum() > 0:
                for j in range(self.num_emissions):
                    X = self.inp_emissions_all_users[j]
                    Y = self.out_emissions_all_users[j]
                    self.model_emissions[i][j].fit(X, Y, sample_weight = sample_weight)
            else:
                for j in range(self.num_emissions):
                    if isinstance(self.model_emissions[i][j], GLM):
                        self.model_emissions[i][j].coef = np.zeros((self.inp_emissions_all_users[j].shape[1]+self.model_emissions[i][j].fit_intercept,))
                        self.model_emissions[i][j].coef = np.random.rand(self.inp_emissions_all_users[j].shape[1]+self.model_emissions[i][j].fit_intercept,)
                        self.model_emissions[i][j].dispersion = 1
                    if isinstance(self.model_emissions[i][j], LM):
                        if len(responses_emissions[j]) == 1:
                            self.model_emissions[i][j].coef = np.zeros((self.inp_emissions_all_users[j].shape[1]+self.model_emissions[i][j].fit_intercept,))
                            self.model_emissions[i][j].coef = np.random.rand(self.inp_emissions_all_users[j].shape[1]+self.model_emissions[i][j].fit_intercept,)
                            self.model_emissions[i][j].dispersion = 1
                        else:
                            self.model_emissions[i][j].coef = np.zeros((self.inp_emissions_all_users[j].shape[1]+self.model_emissions[i][j].fit_intercept, len(responses_emissions[j])))
                            self.model_emissions[i][j].coef = np.random.rand(self.inp_emissions_all_users[j].shape[1]+self.model_emissions[i][j].fit_intercept, len(responses_emissions[j]))
                            self.model_emissions[i][j].dispersion = np.eye(len(responses_emissions[j]))
                    if isinstance(self.model_emissions[i][j], MNLD):
                        self.model_emissions[i][j].coef = np.zeros((self.inp_emissions_all_users[j].shape[1]+self.model_emissions[i][j].fit_intercept,np.unique(self.out_emissions_all_users[j]).shape[0]))
                        self.model_emissions[i][j].coef = np.random.rand(self.inp_emissions_all_users[j].shape[1]+self.model_emissions[i][j].fit_intercept,np.unique(self.out_emissions_all_users[j]).shape[0])
                        self.model_emissions[i][j].lb = LabelBinarizer().fit(self.out_emissions_all_users[j])
    #                     self.model_emissions[i][j].n_targets = len(np.unique(self.out_emissions_all_users[j]))
                    if isinstance(self.model_emissions[i][j], MNLP):
                        self.model_emissions[i][j].coef = np.zeros((self.inp_emissions_all_users[j].shape[1]+self.model_emissions[i][j].fit_intercept,len(responses_emissions[j])))
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
        # print self.log_gammas
        # print np.exp(self.log_gammas)
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
        print >> sys.stderr, "Usage: SemiSupervised_IOHMM_MapReduce<file>"
        exit(-1)

    sc = SparkContext(appName="Python_SemiSupervised_IOHMM_MapReduce", pyFiles=[
        '../auxiliary/SemiHMM.py',
        '../auxiliary/SupervisedModels.py',
        '../auxiliary/family.py'])


    speed = pd.read_csv('../data/speed.csv')
    states = {}
    corr = np.array(speed['corr'])
    for i in range(int(len(corr)/2)):
        state = np.zeros((2,))
        if corr[i] == 'cor':
            states[i] = np.array([0,1])
        else:
            states[i] = np.array([1,0])

    indexes = [(1,1), (2,1)]
    RDD = sc.parallelize(indexes)
    dfs_states = RDD.mapValues(lambda v: [speed, states])
    

    SHMM = SemiSupervisedIOHMMMapReduce(num_states=2, max_EM_iter=100, EM_tol=1e-4)
    SHMM.setData(dfs_states)
    SHMM.setModels(model_emissions = [LM()], model_transition=MNLP(solver='lbfgs'))
    SHMM.setInputs(covariates_initial = [], covariates_transition = [], covariates_emissions = [[]])
    SHMM.setOutputs([['rt']])
    SHMM.train()
    print 'done'


