## Label Noise Robustness of Conformal Prediction

This repository contains code accompanying the following paper: "Label Noise Robustness of Conformal Prediction".
The contents of this repository include a Python package implementing the experiments with synthetic and real data presented in the paper.

### Abstract

We study the robustness of conformal prediction, a powerful tool for uncertainty quantification, to label noise. Our analysis tackles both regression and classification problems, characterizing when and how it is possible to construct uncertainty sets that correctly cover the unobserved noiseless ground truth labels. We further extend our theory and formulate the requirements for correctly controlling a general loss function, such as the false negative proportion, with noisy labels. 
Our theory and experiments suggest that conformal prediction and risk-controlling techniques with noisy labels attain conservative risk over the clean ground truth labels whenever the noise is dispersive and increases variability.
In such cases, we can also correct for noise of bounded size in the conformal prediction algorithm in order to ensure achieving the correct risk of the ground truth labels without score or data regularity. 



### Contents
•	`AVA/` : \Code for reproducing results of experiments with AVA dataset.\
•	`cifar-10/` : \Code for reproducing results of experiments with CIFAR-10H dataset.\
•	`synthetic_classification/` : \Code for reproducing results of synthetic classification experiment.\
•	`synthetic_regression/` : \Code for reproducing results of synthetic regression experiment.\
Run 'main' file in order to run all experiments in parallel on a computing cluster and reproduce results from the paper. 'Run_all' and 'submit' files run the main file with different seeds or varying parameters on the clusters. 'Show_results' notebook visualizes the results achieved in these experiments and create the figures presented in the paper.\
•	`risk_control/` : \Code for reproducing results of risk control experiments.

### Prerequisites
Python package dependencies:

•	numpy\
•	torch\
•	tqdm\
•	panda\
•	matplotlib\
•	sys\
•	os\
•	sklearn\
•	random\
•	seaborn\
•	scikit-garden\
•	scipy

The code for the numerical experiments was written to be run on a computing cluster using the SLURM scheduler.
