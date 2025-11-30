# Parts of Speech tagging using a Hidden Markov Model (HMM)

--------

This repository is the result of a project we did as part of our computational syntax course, which is part of the syllabus of the [Master in Language Analysis and Processing on EHU/UPV](https://www.ehu.eus/en/web/master/master-language-analysis-processing). 

The objective of this project was to implement Part of Speech tagging from scratch using a Hidden Markov Model and experiment on said model. The use of high level machine learning libraries was obviously forbidden.

--------

## Achievements of this project

### Manual implementation of:

- Full HMM training loop
- Viterbi algorithm for tagging
- Evaluation loop (accuracy, macro&micro F1)

### Functionality of the program

- The model is designed to be trained on the [Universal Dependencies](https://universaldependencies.org) data format. Therefore, it can be trained with any language, independently of its script or syntactic structure.   
- Using the currently accessible data as of 30.11.2025, it achieves the following accuracy when trained on the UD datasets: 
	- Spanish - 94.3%
	- Persian - 92.7%
	- Czech - 91.0%
	- Urdu - 89.5%
	- English - 89.1%
	- German - 88.5%
	- Slovak - 72.35%

For comparison, since there are 17 tags in the UD system, the chance of getting the correct one by random chance is about 5.8%.

## How to use the program? 

Assuming you are on Linux, to simply test the code, clone the repo and run:

```sh
chmod +x scripts/download_data.sh
./scripts/download_data.sh

python main.py
# Or python3, but you probably know which one you need to use
```

For Windows users, clone the repository and run this in PowerShell: 

```sh
bash scripts/download_data.sh

python main.py
# Or python3, but you probably know which one you need to use
```

To use the actual functionality, you can follow and adapt the following template:

```python
# Import the needed functions
from hmm.dataio import load_treebank  
from hmm.model import HMMTagger

# Load the training data
train_sentences = load_treebank("path/to/data/train.conllu")

# Initiate the class
tagger = HMMTagger()

# Do the actual training
tagger.train(train_sentences)

# Tag something
result = tagger.tag("This mode is awesome!")

# And show it!
print(result)
```

## Libraries used for the model

- Collections
- Conllu
- Numpy
- Random
- Re
- Typing
