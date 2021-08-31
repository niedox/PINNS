
# Neural Network based ion-channel model fitting
Implementation of PINNS (physically inspired neural networks) to model ion-channels. A combination of  experimental data and Hodgkin-Huxley ODEs is used to train the PINNs-model.  

## Repository structure

* "replication/" contains the original code (Tensorflow) from Raissi's paper and the replication in PyTorch  
    * Raissi_original/Burgers.py is the original code
    * Raissi_replication/burgers_rep3.py is the replication. It learns the PDE parameters both for noiseless and noisy data.
    * the working directory should be replication/

* "ion_channel_fit/" contains the data and code related to the modeling of ion-channels with PINNS. 
   * fit_all_act.py contains the class for the model fitting on data with multiple activation voltages
   * fit_one_act.py contains the class for the model fitting on data with one selected activation voltage
   * fit_one_notebook.ipynb is used to train the single-activation-voltage model, plot and store the results
   * fit_one_reps.ipynb is used to train the single-activation-voltage model on several experimental repetitions, plot and store the results
   * fit_all_notebook.ipynb is used to train the multiple-activation-voltage model, plot and store the results
   * fit_all_reps.ipynb is used to train the multiple-activation-voltage model on several experimental repetitions, plot and store the results
   * "surrogate/" contains a notebook that can generate artificial data with selected levels of noise
   * "plots/" is the directory where the results' plots are stored
   * the working directory must be ion_channel_fit/
   
## PARAMETERS

In the replication part, the following parameters can be tuned
* PLOT: whether to plot results
* MAX_ITER_LBFGS: maximal number of iterations per LBFGS-optimization step
* NOISE: noise percentage for the fitting on noisy data

In the ion-channel fitting part, the following parameters can be tuned:
* PLOT: whether to plot results
* MAX_ITER_LBFGS: maximal number of iterations per LBFGS-optimization step
* LINESEARCH:  line search function (either "strong_wolfe" or None)
* NN_DEPTH:  depth of the neural nets (number of layers)
* NN_BREADTH:  breadth of the neural nets (number of neurons per layer)
* E_K: channel reversial potential (here, potassium)
* G_K: channel ref conductance (here, potassium)
* ADAM_IT: number of ADAM iteration to perform
* WHOLE_DATA: whether or not to use the whole data as training set
* DEVICE: "cpu" or "cuda". Whether to train on cpu or gpu
* EXP_REP: repetition (activation voltage) to consider (only for fit_one_act)


