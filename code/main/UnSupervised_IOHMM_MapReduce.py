from __future__ import division
from copy import deepcopy
import numpy as np
from operator import add
import pandas as pd
from pyspark import SparkContext
import sys
sys.path.append('../auxiliary')
from SupervisedModels import *
from HMM import *


import warnings
warnings.simplefilter("ignore")

## example:  ./bin/spark-submit UnSupervised_IOHMM_MapReduce.py


def EStepMap(df, model_initial, model_transition, model_emissions, 
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

    log_gamma, log_epsilon, ll = calHMM(log_prob_initial, log_prob_transition, log_Ey)


    return [[log_gamma, log_epsilon, ll]]




## adapted code to MapReduce
## will be improved if Spark MLLIB support linear models with sample weights
class UnSupervisedIOHMMMapReduce:
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
    
    def setData(self, rdd_dfs):
    	# here the rdd is the rdd with (k, v) pairs that v is a dataframe
        self.num_seqs = rdd_dfs.count()
        self.dfs = rdd_dfs
        
    
    def setInputs(self, covariates_initial, covariates_transition, covariates_emissions):

        self.covariates_initial = covariates_initial
        self.covariates_transition = covariates_transition
        self.covariates_emissions = covariates_emissions
        
        self.inp_initials_all_users = self.dfs.map(lambda x: np.array(x[1][covariates_initial].iloc[0]).reshape(1,-1).astype('float64')).reduce(lambda a, b: np.vstack((a, b)))
        self.model_initial.coef = np.random.rand(self.inp_initials_all_users.shape[1]+self.model_initial.fit_intercept,self.num_states)
        
        self.inp_transitions_all_users = self.dfs.map(lambda x: np.array(x[1][covariates_transition].iloc[1:]).astype('float64')).reduce(lambda a, b: np.vstack((a, b)))
        
        for st in range(self.num_states):
            self.model_transition[st].coef = np.random.rand(self.inp_transitions_all_users.shape[1]+self.model_transition[st].fit_intercept,self.num_states)
        self.inp_emissions_all_users = []
        for cov in covariates_emissions:
            self.inp_emissions_all_users.append(self.dfs.map(lambda x: np.array(x[1][cov]).astype('float64')).reduce(lambda a, b: np.vstack((a, b))))
        
        
    
    def setOutputs(self, responses_emissions):
        # output should be a list inidicating the columns of the dataframe
        self.responses_emissions = responses_emissions
        self.out_emissions_all_users = []
        for res in responses_emissions:
            self.out_emissions_all_users.append(self.dfs.map(lambda x: np.array(x[1][res])).reduce(lambda a, b: np.vstack((a, b))))
        for i in range(self.num_states):
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


        rdd_E = self.dfs.mapValues(lambda v: EStepMap(v, model_initial, model_transition, model_emissions, 
            covariates_initial, covariates_transition, covariates_emissions, 
            responses_emissions, num_states, num_emissions))

        posteriors = rdd_E.map(lambda x: x[1]).reduce(add)

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
        print >> sys.stderr, "Usage: UnSupervised_IOHMM_MapReduce<file>"
        exit(-1)

    sc = SparkContext(appName="Python_UnSupervised_IOHMM_MapReduce", pyFiles=[
        '../auxiliary/HMM.py',
        '../auxiliary/SupervisedModels.py',
        '../auxiliary/family.py'])


    speed = pd.read_csv('../data/speed.csv')
    indexes = [(1,1), (2,1)]
    RDD = sc.parallelize(indexes)
    dfs = RDD.mapValues(lambda v: speed)
    
    SHMM = UnSupervisedIOHMMMapReduce(num_states=2, max_EM_iter=100, EM_tol=1e-4)
    SHMM.setData(dfs)
    SHMM.setModels(model_emissions = [LM()], model_transition=MNLP(solver='lbfgs'))
    SHMM.setInputs(covariates_initial = [], covariates_transition = [], covariates_emissions = [[]])
    SHMM.setOutputs([['rt']])
    SHMM.train()
    print 'done'













    