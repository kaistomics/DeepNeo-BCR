# DeepNeo-BCR
Predict linear B cell epitopes of fixed length (12-, 15-, and 16-mer)

## DeepNeo-BCR: a tool for predicting linear B cell epitopes 

Despite recent advances in bioinformatics, prediction of B cell epitopes has remained challenging. Here, we developed a convolutional neural network-based models to accurately predict B cell–binding epitopes agianst the general B cell population. Sixty-three independent models were created for representative IGHV alleles in humans and mice, and these models were then combined into an ensemble using linear regression.

References: Manuscript in press


## Download and install:

Please download this github repo.

The code can be run on Python>3.6 and Keras with tensorflow backend.
Necessary packages are listed below.
Typical installation time on a Linux machine is ~1m if requirements are met, and ~15m if starting from a new conda environment.
If you wish to install on conda environment, please refer to conda_env.yml for required packages


The input file of DeepNeo-BCR is a single column file with query peptide list.
An example data is provided within this repo.

```
python predict_63.py GPU_NUM MODE INPUTFILE
```
is the basic command line for DeepNeo-BCR.

Users can test the code using

```
python predict_63.py 0 all Example/example.txt
```

Although GPU is not necessary to run the code, it will be helpful in prompt prediction.
If no GPU is available, please enter '0'.
Estimated run time for example data on a non-GPU desktop computer is ~10sec.

There are four modes available : all, human, human_reduced, mouse

'all' includes all mouse and human alleles.

'human' includes all human alleles (N=48)

'human_reduced' includes representative human alleles (N=25) and can be used if computational power is limited.

'mouse' includes mouse alleles.

We suggest using >0.3 to interpret B cell epitopes.

## Version
This software was tested on Ubuntu 20.04 machine with following software versions:


Python=3.7

Tensorflow=2.11

Pandas=1.3.5

scikit-learn=1.0.2

scipy=1.7.3

