## Conformal Prediction is Robust to Label Noise

This repository contains code accompanying the following paper: "Conformal Prediction is Robust to Label Noise".
The contents of this repository include a Python package implementing the experiments with synthetic and real data presented in the paper.

### Abstract

We study the robustness of conformal prediction—a powerful tool for uncertainty quantification—to label noise. Our analysis tackles both regression and classification problems, characterizing when and how it is possible to construct uncertainty sets that correctly cover the unobserved noiseless ground truth labels. Through stylized theoretical examples and practical experiments, we argue that na¨ıve conformal prediction covers the noiseless ground truth label unless the noise distribution is adversarially designed. This leads us to believe that correcting for label noise is unnecessary except for pathological data distributions or noise sources. In such cases, we can also correct for noise of bounded size in the conformal prediction algorithm in order to ensure correct coverage of the ground truth labels without score or data regularity.



### Contents
•	`AVA/` : \Code for reproducing results of experiments with AVA dataset.\n
•	`cifar-10/` : \Code for reproducing results of experiments with CIFAR-10H dataset.\n
•	`synthetic_classification/` : \Code for reproducing results of synthetic classification experiment.\n
•	`synthetic_regression/` : \Code for reproducing results of synthetic regression experiment.\n
Run 'main' file in order to run all experiments in parallel on a computing cluster and reproduce results from the paper. 'Run_all' and 'submit' files run the main file with different seeds or varying parameters on the clusters. 'Show_results' notebook visualizes the results achieved in these experiments and create the figures presented in the paper.

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
