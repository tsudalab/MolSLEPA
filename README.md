# MolSLEPA
Interpretable Fragment-based Molecule Design with Self-learning Entropic Population Annealing


## Environment Setup
An environment for rationale can be easily setup via Anaconda:
```
git clone https://github.com/tsudalab/MolSLEPA.git
cd MolSLEPA
conda env create -f environment.yml
conda activate molslepa
```

## Workflow

All command line executables are under the folder 'cli':
```
cd cli
```

The molecule generation in MolSLEPA refers to [MoLeR](https://github.com/microsoft/molecule-generation). 
To run MolSLEPA, follow four steps:

### Step 1: Preprocessing
- Preprocess data using the 'preprocess.py' script. This script takes a plain text list of SMILES strings and turns it into '*.pkl' files containing descriptions of the molecular graphs and generation traces. You need to provide train, valid and test datasets. Each file contains SMILES strings, one per line. The folder and file name must match the name in  'preprocess.py'. To run
```
python preprocess.py
```

### Step 2: Training
- Train MoLeR on the preprocessed data using the 'train.py' script. This script trains MoLeR until convergence, run
```
python train.py
```
### Step 3: Sampling
- Sample fragment-based chemical space using the class 'Sample' in 'MolSLEPA' script. This script generates a set of weighted samples of molecules, run
```
python molslepa.py
```
### Step 4: Dos Estimation
- Calculate the saliency of fragments using the class 'MultiHistogram' in 'MolSLEPA' script. Thi script approximate the density of states (DoS), which is determined by the weights obtained in last step. 

'ploy.ipynb' provides a reproduction of the resulting figure in the paper.

