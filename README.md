# IO-HMM
This is the repository that includes my python implementation of IO-HMM

My implementation is easy to use. Please see examples and code in the notebook.

Example usage:


`SHMM = SupervisedHMM(num_states=2, max_EM_iter=1000, EM_tol=1e-4)` <br/>
`SHMM.setData([speed])`<br/>
`SHMM.setModels(model_emissions = [LM(), MNLD()], model_transition=MNLP(solver='lbfgs'))`<br/>
`SHMM.setInputs(covariates_initial = [], covariates_transition = [], covariates_emissions = [[],[]])`<br/>
`SHMM.setOutputs([['rt'],['corr']])`<br/>
