# MC-SQ: A Highly Accurate Ensemble for Multi-class Quantification

This repository provides the code used in the experiments of our paper

*MC-SQ: A Highly Accurate Ensemble for Multi-class Quantification*  
*Zahra Donyavi, Adriane Serapião, Gustavo Batista*



In the following, we describe how one can reproduce our results.


## Experiment

This experiment has been run with Python 3.9.11. An environment including all packages required to run our code is given by the ```Environment.yaml``` file. Some packages like ```numpy```, ```pandas```, ```scikit-learn``` and ```cvxpy``` are used in most algorithms.


##### Loading Datasets

For dataset preparation and sampling part of the codes, we used ```https://github.com/tobiasschumacher/quantification_paper.git```. Each dataset has a ```prep.py``` file to prepare. After preparing all datasets, the parameter ```load_from_disk=False``` in line 167 in ```run.py``` can be set to the ```True``` value.


#### Main Experiments

To reproduce all our experiments with all datasets, all of our five ensemble models (EnsembleDyS, EnsembleEM, EnsembleGAC, EnsembleGPAC, and EnsembleFM), and implemented single quantifiers (EMQ, GAC, GPAC, FM, and HDy), one can simply run our main script via 

```bash
    python3 -m run.py -a {algorithms} -d {datasets} --mc {} --cal {} --seeds {seeds} 
```

where algorithms and datasets to run on can be specified by their respective names as listed in ```alg_index.csv``` and ```data/data_index.csv```. When none of the arguments are specified, all experiments will be executed. The 10 default seeds we used are specified in ```run.py```.

By using ```--mc 0``` or ```--mc 1```, all binary or multi-class datasets will be run at once, and if ```--mc``` do not be specified, experiments will run on all binary and multi-class datasets at once.

```--cal``` argument specifies whether the base classifiers should be calibrated or not.


### Algorithms

The implementations of methods can be found in the ```Ensemble_Quantifier.py``` file. 






