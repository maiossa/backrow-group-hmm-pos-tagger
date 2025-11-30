# A script to test  how different masking rates affect the quality of the model.

from hmm.dataio import load_treebank
from hmm.model import HMMTagger

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_sentences = load_treebank("data/english/gum/train.conllu")
test_sentences =  load_treebank("data/english/ewt/test.conllu")

results = []

for mask_rate in np.arange(0, 0.06, 0.01):
    tagger = HMMTagger()
    tagger.train(train_sentences, mask_rate=mask_rate)
    accuracy = tagger.evaluate(test_sentences)[0]
    results.append({"mask_rate": mask_rate, "accuracy": accuracy})

df = pd.DataFrame(results)

# Plot
plt.figure()
plt.plot(df["mask_rate"], df["accuracy"], marker="o")
plt.xlabel("Mask rate")
plt.ylabel("Accuracy")
plt.title("HMM Tagger Accuracy vs Mask Rate")
plt.grid(True)
plt.show()



# My question is as follows:

# The tagger is obviously going to lose some accuracy when switching domains (gum -> ewt)
# Will masking out random words help it generalize better?

# This script trains the tagger with different rates of random masking and extracts the efficiency
